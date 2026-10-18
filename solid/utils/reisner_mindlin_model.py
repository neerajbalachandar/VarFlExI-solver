from dolfin import *
import numpy as np

# from FINAL.solid.elastodynamics_reisner_span_for_validation import left

from .generalized_alpha import GeneralizedAlphaIntegrator


class ReissnerMindlinModel:
    """Mixed Reissner--Mindlin plate model with generalized-alpha time stepping.

    When ``geometric_nonlinearity`` is enabled, membrane kinematics use the
    Green--Lagrange mid-surface metric strain.  Rotations remain the
    independent, leading-order Reissner--Mindlin rotations: this is not a
    geometrically exact finite-rotation shell or a materially nonlinear model.

    Credits to fenics_shells for the original Reissner--Mindlin implementation.
    """

    def __init__(
        self,
        domain,
        E,
        nu,
        rho_s,
        eta_m,
        eta_k,
        kappa_shear,
        dt,
        alpha_m=0.10,
        alpha_f=0.20,
        geometric_nonlinearity=True,
        include_quadratic_inplane_strain=True,
    ):
        self.domain = domain
        self.E = float(E)
        self.nu = float(nu)
        self.rho_s = float(rho_s)
        self.rho = Constant(self.rho_s)
        self.eta_m = Constant(float(eta_m))
        self.eta_k = Constant(float(eta_k))
        self.kappa_shear = Constant(float(kappa_shear))
        self.h = domain.h
        self.dt = dt
        # Kept configurable so the previous linear formulation remains an
        # available baseline without changing the mixed-space/FSI interfaces.
        self.geometric_nonlinearity = bool(geometric_nonlinearity)
        # False retains the preceding von Karman membrane approximation; true
        # (the default) adds 1/2 grad(u)^T grad(u) for the Green--Lagrange
        # mid-surface metric strain.
        self.include_quadratic_inplane_strain = bool(include_quadratic_inplane_strain)
        self.integrator = GeneralizedAlphaIntegrator(dt, Constant(alpha_m), Constant(alpha_f))
        self.gamma = self.integrator.gamma
        self.beta = self.integrator.beta
        self.I2 = Identity(2)

        self._build_spaces()
        self._build_state()
        self._build_bcs()
        # self._build_forms() # Commented and check out what is the fault here, added the below line to correct as it was similar. where is the defn?
        self.build_residual_forms()

    def _build_spaces(self):
        mesh = self.domain.mesh
        U_el = VectorElement("CG", mesh.ufl_cell(), 2, dim=2)
        W_el = FiniteElement("CG", mesh.ufl_cell(), 2)
        T_el = VectorElement("CG", mesh.ufl_cell(), 2, dim=2)
        mixed_element = MixedElement([U_el, W_el, T_el])
        self.V = FunctionSpace(mesh, mixed_element)
        self.Vt = VectorFunctionSpace(mesh, "CG", 1, dim=3)
        self.Vsig = TensorFunctionSpace(mesh, "DG", 0, shape=(2, 2))
        self.t_aero = Function(self.Vt, name="AerodynamicTraction")

    def _build_state(self):
        self.dq_trial = TrialFunction(self.V)
        self.q_test = TestFunction(self.V)
        self.q = Function(self.V, name="PlateState")
        self.q_old = Function(self.V)
        self.v_old = Function(self.V)
        self.a_old = Function(self.V)
        self.dq_newton = Function(self.V)
        self.linear_solver = LUSolver("mumps")
        self.sig = Function(self.Vsig, name="MembraneStress")

    def _build_bcs(self):
        u_zero_2d = Constant((0.0, 0.0))

        # print("========== BC DEBUG ==========")
        # print("V              :", self.V)
        # print("V.sub(0)       :", self.V.sub(0))
        # print("V.sub(1)       :", self.V.sub(1))
        # print("V.sub(2)       :", self.V.sub(2))
        # print("u_zero_2d      :", u_zero_2d)
        # print("is_left_boundary:", self.domain.is_left_boundary)
        # print("type:", type(self.domain.is_left_boundary))
        # print("==============================")
        
        
        # self.bc_u = DirichletBC(self.V.sub(0), u_zero_2d, self.domain.is_left_boundary)     
        # self.bc_w = DirichletBC(self.V.sub(1), Constant(0.0), self.domain.is_left_boundary)
        # self.bc_theta = DirichletBC(self.V.sub(2), u_zero_2d, self.domain.is_left_boundary)

# Why is left BC import failing previously?
        left = CompiledSubDomain("near(x[1], 0.0) && on_boundary")

        self.bc_u = DirichletBC(self.V.sub(0), Constant((0.0, 0.0)), left)
        self.bc_w = DirichletBC(self.V.sub(1), Constant(0.0), left)
        self.bc_theta = DirichletBC(self.V.sub(2), Constant((0.0, 0.0)), left)

        self.bcs = [self.bc_u, self.bc_w, self.bc_theta]

    def split_state(self, q_fun):
        if isinstance(q_fun, tuple):
            return q_fun
        return split(q_fun)

    def membrane_strain(self, u_mem):
        """Small (linear) membrane strain, retained for reference stiffness."""
        return sym(grad(u_mem))

    def nonlinear_membrane_strain(self, u_mem, w):
        """Green--Lagrange mid-surface membrane strain at the current state.

        With the default settings this is
        sym(grad(u)) + 1/2 grad(u).T*grad(u) + 1/2 grad(w) outer grad(w).
        Disabling ``include_quadratic_inplane_strain`` recovers the prior
        von Karman membrane strain.
        """
        grad_u = grad(u_mem)
        grad_w = grad(w)
        eps = self.membrane_strain(u_mem) + 0.5 * outer(grad_w, grad_w)
        if self.include_quadratic_inplane_strain:
            eps += 0.5 * dot(grad_u.T, grad_u)
        return eps

    def nonlinear_membrane_variation(self, u_mem, w, delta_u, delta_w):
        """Consistent virtual von Karman membrane strain.

        The transverse part contains the current ``w`` and virtual ``delta_w``;
        evaluating ``nonlinear_membrane_strain(delta_u, delta_w)`` here would
        be mathematically incorrect.
        """
        eps_virtual = self.membrane_strain(delta_u) + sym(
            outer(grad(w), grad(delta_w))
        )
        if self.include_quadratic_inplane_strain:
            eps_virtual += sym(dot(grad(u_mem).T, grad(delta_u)))
        return eps_virtual

    def curvature(self, theta):
        return sym(grad(theta))

    def shear_strain(self, theta, w):
        return grad(w) - theta

    def membrane_stress(self, u_mem):
        """Linear membrane resultant used by the reference stiffness form."""
        eps = self.membrane_strain(u_mem)
        return self.membrane_stress_from_strain(eps)

    def membrane_stress_from_strain(self, eps):
        coeff = self.E * self.h / (1.0 - self.nu ** 2)
        return coeff * ((1.0 - self.nu) * eps + self.nu * tr(eps) * self.I2)

    def nonlinear_membrane_stress(self, u_mem, w):
        """Membrane resultant corresponding to the selected nonlinear strain."""
        return self.membrane_stress_from_strain(self.nonlinear_membrane_strain(u_mem, w))

    def bending_moment(self, theta):
        kap = self.curvature(theta)
        coeff = self.E * self.h ** 3 / (12.0 * (1.0 - self.nu ** 2))
        return coeff * ((1.0 - self.nu) * kap + self.nu * tr(kap) * self.I2)

    def displacement_3d(self, q_fun):
        u_mem, w, _theta = self.split_state(q_fun)
        return as_vector((u_mem[0], u_mem[1], w))

    def mass_form(self, q_trial, q_test):
        if isinstance(q_trial, tuple):
            u_t, w_t, theta_t = q_trial
        else:
            u_t, w_t, theta_t = split(q_trial)

        if isinstance(q_test, tuple):
            u_x, w_x, theta_x = q_test
        else:
            u_x, w_x, theta_x = split(q_test)

        inertia_rot = self.rho * self.h ** 3 / 12.0
        return self.rho * self.h * (inner(u_t, u_x) + w_t * w_x) * dx + inertia_rot * inner(
            theta_t, theta_x
        ) * dx

    def stiffness_form(self, q_trial, q_test):
        """Linear elastic bilinear form and Rayleigh reference stiffness."""
        if isinstance(q_trial, tuple):
            u_t, w_t, theta_t = q_trial
        else:
            u_t, w_t, theta_t = split(q_trial)

        if isinstance(q_test, tuple):
            u_x, w_x, theta_x = q_test
        else:
            u_x, w_x, theta_x = split(q_test)

        eps_x = self.membrane_strain(u_x)
        kap_x = self.curvature(theta_x)
        gam_t = self.shear_strain(theta_t, w_t)
        gam_x = self.shear_strain(theta_x, w_x)

        N_t = self.membrane_stress(u_t)
        M_t = self.bending_moment(theta_t)
        G_shear = Constant(self.E / (2.0 * (1.0 + self.nu)))
        K_shear = self.kappa_shear * G_shear * self.h
        return (
            inner(N_t, eps_x) * dx
            + inner(M_t, kap_x) * dx
            + K_shear * inner(gam_t, gam_x) * dx
        )

    def damping_form(self, q_trial, q_test):
        """Reference/initial-stiffness Rayleigh damping.

        ``eta_k`` multiplies the linear reference stiffness, not the nonlinear
        internal-force functional.  This keeps C = eta_m M + eta_k K_ref a
        velocity-linear damping operator with the original units/meaning.
        """
        return self.eta_m * self.mass_form(q_trial, q_test) + self.eta_k * self.stiffness_form(
            q_trial, q_test
        )

    def external_work(self, q_test):
        # The active FSI solver applies transferred nodal loads directly to its
        # algebraic right-hand side.  It intentionally does not add this form
        # to self.residual; doing both would double-count aerodynamic loading.
        v_disp = self.displacement_3d(q_test)
        return dot(v_disp, self.t_aero) * self.domain.ds_aero(1)

    def internal_work(self, q_state, q_test):
        """Internal virtual work at a displacement state.

        In the nonlinear branch this is
        int[N:delta(E_m) + M:delta(kappa) + Q.delta(gamma)] dOmega.
        It is a residual (linear in ``q_test``), rather than a bilinear
        stiffness form, so UFL can construct its complete consistent tangent.
        """
        u_mem, w, theta = self.split_state(q_state)
        delta_u, delta_w, delta_theta = self.split_state(q_test)

        if not self.geometric_nonlinearity:
            return self.stiffness_form(q_state, q_test)

        eps_virtual = self.nonlinear_membrane_variation(
            u_mem, w, delta_u, delta_w
        )
        N = self.nonlinear_membrane_stress(u_mem, w)
        M = self.bending_moment(theta)
        gamma = self.shear_strain(theta, w)
        delta_kappa = self.curvature(delta_theta)
        delta_gamma = self.shear_strain(delta_theta, delta_w)
        G_shear = Constant(self.E / (2.0 * (1.0 + self.nu)))
        K_shear = self.kappa_shear * G_shear * self.h
        return (
            inner(N, eps_virtual) * dx
            + inner(M, delta_kappa) * dx
            + K_shear * inner(gamma, delta_gamma) * dx
        )

    def build_residual_forms(self):
        q_u, q_w, q_th = split(self.q)
        qo_u, qo_w, qo_th = split(self.q_old)
        vo_u, vo_w, vo_th = split(self.v_old)
        ao_u, ao_w, ao_th = split(self.a_old)

        a_u_new = self.integrator.update_a(q_u, qo_u, vo_u, ao_u, ufl=True)
        a_w_new = self.integrator.update_a(q_w, qo_w, vo_w, ao_w, ufl=True)
        a_th_new = self.integrator.update_a(q_th, qo_th, vo_th, ao_th, ufl=True)

        v_u_new = self.integrator.update_v(a_u_new, qo_u, vo_u, ao_u, ufl=True)
        v_w_new = self.integrator.update_v(a_w_new, qo_w, vo_w, ao_w, ufl=True)
        v_th_new = self.integrator.update_v(a_th_new, qo_th, vo_th, ao_th, ufl=True)

        a_alpha = (
            self.integrator.average(ao_u, a_u_new, self.integrator.alpha_m),
            self.integrator.average(ao_w, a_w_new, self.integrator.alpha_m),
            self.integrator.average(ao_th, a_th_new, self.integrator.alpha_m),
        )
        v_alpha = (
            self.integrator.average(vo_u, v_u_new, self.integrator.alpha_f),
            self.integrator.average(vo_w, v_w_new, self.integrator.alpha_f),
            self.integrator.average(vo_th, v_th_new, self.integrator.alpha_f),
        )
        q_alpha = (
            self.integrator.average(qo_u, q_u, self.integrator.alpha_f),
            self.integrator.average(qo_w, q_w, self.integrator.alpha_f),
            self.integrator.average(qo_th, q_th, self.integrator.alpha_f),
        )

        self.residual = (
            self.mass_form(a_alpha, self.q_test)
            + self.damping_form(v_alpha, self.q_test)
            # Internal force is evaluated at q_{n+alpha_f}, consistently with
            # the existing generalized-alpha displacement interpolation.
            + self.internal_work(q_alpha, self.q_test)
        )
        # Differentiating the complete residual includes all material and
        # geometric tangent terms from both quadratic membrane contributions.
        self.jacobian = derivative(self.residual, self.q, self.dq_trial)
        return self.residual, self.jacobian

    def update_fields(self):
        self.integrator.update_fields(self.q, self.q_old, self.v_old, self.a_old)

    def compute_strain_energy(self):
        u_mem_t, w_t, theta_t = split(self.q_old)
        if self.geometric_nonlinearity:
            eps_mem = self.nonlinear_membrane_strain(u_mem_t, w_t)
            N_mem = self.membrane_stress_from_strain(eps_mem)
        else:
            eps_mem = self.membrane_strain(u_mem_t)
            N_mem = self.membrane_stress_from_strain(eps_mem)
        return assemble(
            0.5 * inner(N_mem, eps_mem) * dx
            + 0.5 * inner(self.bending_moment(theta_t), self.curvature(theta_t)) * dx
            + 0.5
            * self.kappa_shear
            * (self.E / (2.0 * (1.0 + self.nu)))
            * self.h
            * inner(self.shear_strain(theta_t, w_t), self.shear_strain(theta_t, w_t))
            * dx
        )

    def compute_surface_strain_rms(self):
        """Area RMS of the top/bottom plate-surface strain tensors."""
        u_mem, w, theta = split(self.q_old)
        eps_mem = (
            self.nonlinear_membrane_strain(u_mem, w)
            if self.geometric_nonlinearity
            else self.membrane_strain(u_mem)
        )
        eps_bend = 0.5 * self.h * self.curvature(theta)
        eps_top = eps_mem + eps_bend
        eps_bottom = eps_mem - eps_bend
        measure = dx(domain=self.domain.mesh)
        area = assemble(Constant(1.0) * measure)
        if area <= 0.0:
            raise RuntimeError("Cannot compute surface strain on a zero-area mesh")
        mean_square = (
            assemble(inner(eps_top, eps_top) * measure)
            + assemble(inner(eps_bottom, eps_bottom) * measure)
        ) / (2.0 * area)
        return float(np.sqrt(max(mean_square, 0.0)))

    def compute_kinetic_energy(self):
        return 0.5 * assemble(self.mass_form(self.v_old, self.v_old))

    def compute_damping_increment(self, dt_value):
        return float(dt_value) * assemble(self.damping_form(self.v_old, self.v_old))

    def extract_dof_maps(self):
        dofs_u_x = np.asarray(self.V.sub(0).sub(0).dofmap().dofs(), dtype=np.int64)
        dofs_u_y = np.asarray(self.V.sub(0).sub(1).dofmap().dofs(), dtype=np.int64)
        dofs_w = np.asarray(self.V.sub(1).dofmap().dofs(), dtype=np.int64)
        return dofs_u_x, dofs_u_y, dofs_w


__all__ = ["ReissnerMindlinModel"]
