"""Fixed-population coupled discrete-model MMS. It exercises the installed
FLOWVLM/rVPM/Reissner–Mindlin operators and the production socket chronology.
It is not a continuum Navier–Stokes MMS, does not verify native particle
lifecycle, and does not treat CRM/RBF moment conservation as verified.

Implements formulation sections "Weak form", "Generalized-α stage values",
"Independent target aerodynamic load on the structure", and "Structural
manufactured source coupled to the target force". The structural source is
Objective B (end-to-end transfer and FSI accuracy); the particle scope remains
fixed-population. If panel-resolved target force data are unavailable, the
reference functional below explicitly uses a smooth uniform-pressure
approximation rather than claiming independent force-law verification.
"""

import math
import numpy as np

from dolfin import *

def _scalar_expression_code(target, name, t, order=0):
    spec = target.structural.components[name]
    amp, omega, mode = float(spec["amplitude"]), float(spec["omega"]), spec["mode"]
    phase = f"({omega:.17g}*targ_t)"
    if mode == "cos":
        factors = (f"cos({phase})", f"(-{omega:.17g})*sin({phase})",
                   f"(-{omega**2:.17g})*cos({phase})")
    elif mode == "sin":
        factors = (f"sin({phase})", f"{omega:.17g}*cos({phase})",
                   f"(-{omega**2:.17g})*sin({phase})")
    elif mode == "ramp":
        factors = (f"(1-cos({phase}))", f"{omega:.17g}*sin({phase})",
                   f"{omega**2:.17g}*cos({phase})")
    else:
        raise ValueError(f"unsupported target mode {mode!r}")
    if order not in (0, 1, 2):
        raise ValueError("target time derivative order must be 0, 1, or 2")
    Lx, Ly = float(target.structural.Lx), float(target.structural.Ly)
    B = f"pow(sin(pi*x[0]/{Lx:.17g}),4)*pow(sin(pi*x[1]/{Ly:.17g}),4)"
    return f"{amp:.17g}*({B})*({factors[order]})"


def _target_expression(target, name, t, order=0, degree=8):
    """DOLFIN Expression for one exact target component or derivative.

    Formulation section: "Target structural field". The first and second time
    derivatives are analytical mode derivatives; finite differences are not
    used in this manufactured source.
    """
    return Expression(_scalar_expression_code(target, name, t, order), degree=degree,
                      targ_t=float(t))


def _subspace(spaces, index):
    value = spaces[index]
    return value[0] if isinstance(value, tuple) else value


def _target_function(model, target, t, order, spaces):
    """Interpolate analytic target components into the production mixed FE space."""
    Vu, Vw, Vtheta = (_subspace(spaces, i) for i in range(3))
    uexpr = Expression((_scalar_expression_code(target, "u", t, order),
                        _scalar_expression_code(target, "v", t, order)), degree=8,
                       targ_t=float(t))
    thetexpr = Expression((_scalar_expression_code(target, "theta_x", t, order),
                           _scalar_expression_code(target, "theta_y", t, order)), degree=8,
                          targ_t=float(t))
    u = interpolate(uexpr, Vu)
    w = interpolate(_target_expression(target, "w", t, order), Vw)
    theta = interpolate(thetexpr, Vtheta)
    out = Function(model.V)
    FunctionAssigner(model.V, [Vu, Vw, Vtheta]).assign(out, [u, w, theta])
    return out


class ReferenceForceFunctional:
    """Independent uniform-pressure virtual work on the plate surface.

    Formulation sections: "Independent target aerodynamic load on the
    structure" and "Geometry transfer, force transfer, and virtual work".
    The current manifest carries no sectional target-force history, so this
    class uses a smooth uniform pressure approximation. It deliberately does
    not call CouplingTransfer or claim independent target force-law accuracy.
    """
    def __init__(self, model, target):
        self.model = model
        self.target = target
        fs = target.freestream
        self.pressure_scale = 0.01 * 0.5 * float(fs["rho"]) * float(fs["vinf"])**2
        self.omega = 2.3
        self.panel_forces = None
        self.panel_edges = np.linspace(0.0, float(target.structural.Ly), int(target.n_span)+1)

    def set_panel_forces(self, forces):
        forces = np.asarray(forces, dtype=float).reshape(-1, 3)
        if forces.shape[0] != int(self.target.n_span) or not np.isfinite(forces).all():
            raise ValueError("target panel force data must be finite with n_span rows")
        self.panel_forces = forces.copy()

    def work(self, t, dq):
        if self.panel_forces is None:
            # Explicit smooth fallback for unit-level use without a fluid
            # target-force payload; production socket MMS sets panel_forces.
            pressure = self.pressure_scale * math.sin(self.omega * float(t))
            traction = as_vector((0.0, 0.0, pressure))
        else:
            class PanelPressure(UserExpression):
                def __init__(self, parent, **kwargs):
                    self.parent = parent
                    super().__init__(**kwargs)

                def eval(self, values, x):
                    j = min(max(int(np.searchsorted(self.parent.panel_edges, x[1], side="right")-1),
                                0), len(self.parent.panel_forces)-1)
                    dy = self.parent.panel_edges[j+1]-self.parent.panel_edges[j]
                    density = self.parent.panel_forces[j] / max(self.parent.target.chord*dy, 1e-15)
                    values[0], values[1], values[2] = density

                def value_shape(self):
                    return (3,)
            traction = PanelPressure(self, degree=0)
        return dot(self.model.displacement_3d(dq), traction) * self.model.dx

    def vector(self, t):
        return assemble(self.work(t, self.model.q_test))


def _newmark_target_states(model, target, t, spaces):
    """Return exact sampled target and its discrete Newmark stage history.

    Formulation section: "Generalized-α stage values". Previous velocity and
    acceleration are advanced with the installed generalized-alpha
    kinematic maps, beginning from the analytic initial values.
    """
    dt = float(model.dt)
    if dt <= 0:
        raise ValueError("production structural dt must be positive")
    steps = int(round(float(t) / dt))
    if steps < 0 or abs(steps * dt - float(t)) > 1e-8 * max(1.0, abs(float(t))):
        raise ValueError("target source time must lie on the structural time grid")
    qprev = _target_function(model, target, 0.0, 0, spaces)
    vprev = _target_function(model, target, 0.0, 1, spaces)
    aprev = _target_function(model, target, 0.0, 2, spaces)
    qnew = qprev
    for k in range(1, steps + 1):
        qnew = _target_function(model, target, k * dt, 0, spaces)
        anew = Function(model.V)
        vnew = Function(model.V)
        anew.vector()[:] = model.integrator.update_a(
            qnew.vector(), qprev.vector(), vprev.vector(), aprev.vector(), ufl=False
        )
        vnew.vector()[:] = model.integrator.update_v(
            anew.vector(), qprev.vector(), vprev.vector(), aprev.vector(), ufl=False
        )
        anew.vector().apply("insert")
        vnew.vector().apply("insert")
        qprev, vprev, aprev = qnew, vnew, anew
    return qnew, qprev if steps == 0 else _target_function(model, target, max(0, steps-1)*dt, 0, spaces), vprev, aprev


def add_coupled_mms_source(model, target, t, reference_force, spaces):
    """Assemble the target residual minus independent reference aerodynamic work.

    Formulation sections: "Structural manufactured source coupled to the
    target force" and "Generalized-α stage values". This vector is added to
    the production right-hand side, whose structural sign is -residual + load.
    It is computed on separate target Functions; numerical solver state is
    never assigned the target.
    """
    dt = float(model.dt)
    qnew = _target_function(model, target, t, 0, spaces)
    qold = _target_function(model, target, max(0.0, float(t)-dt), 0, spaces)
    # Use the actual discrete old-state values from the target trajectory.
    if float(t) <= 0:
        v_old = _target_function(model, target, 0.0, 1, spaces)
        a_old = _target_function(model, target, 0.0, 2, spaces)
    else:
        _, _, v_old, a_old = _newmark_target_states(model, target, float(t)-dt, spaces)
    a_new = Function(model.V)
    v_new = Function(model.V)
    a_new.vector()[:] = model.integrator.update_a(
        qnew.vector(), qold.vector(), v_old.vector(), a_old.vector(), ufl=False
    )
    v_new.vector()[:] = model.integrator.update_v(
        a_new.vector(), qold.vector(), v_old.vector(), a_old.vector(), ufl=False
    )
    a_new.vector().apply("insert")
    v_new.vector().apply("insert")
    am = float(model.integrator.alpha_m)
    af = float(model.integrator.alpha_f)
    a_alpha = (am * split(a_old)[0] + (1-am) * split(a_new)[0],
               am * split(a_old)[1] + (1-am) * split(a_new)[1],
               am * split(a_old)[2] + (1-am) * split(a_new)[2])
    v_alpha = (af * split(v_old)[0] + (1-af) * split(v_new)[0],
               af * split(v_old)[1] + (1-af) * split(v_new)[1],
               af * split(v_old)[2] + (1-af) * split(v_new)[2])
    q_alpha = (af * split(qold)[0] + (1-af) * split(qnew)[0],
               af * split(qold)[1] + (1-af) * split(qnew)[1],
               af * split(qold)[2] + (1-af) * split(qnew)[2])
    residual_form = (model.mass_form(a_alpha, model.q_test)
                     + model.damping_form(v_alpha, model.q_test)
                     + model.internal_work(q_alpha, model.q_test))
    source = assemble(residual_form)
    source.axpy(-1.0, reference_force.vector(float(t)))
    for bc in model.bcs:
        bc.apply(source)
    return source
