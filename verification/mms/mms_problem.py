"""mms_problem.py — Single mathematical definition of the VarFlExI MMS problem.

This module contains ALL manufactured analytical fields, their spatial and
temporal derivatives, independently derived structural sources, particle
targets, interface traction, and error-norm definitions.

INDEPENDENCE GUARANTEE
----------------------
No production solver residuals, weak forms, or numerical operators are called
to construct manufactured sources.  Every source is derived from the governing
differential equations applied analytically to the prescribed target.

FUNCTION SPACE DOCUMENTATION
-----------------------------
Let Ω = [0, Lx] × [0, Ly] be the mid-surface plate domain.

Admissible structural state spaces (continuous setting):
  Membrane translations:   (u₀, v₀) ∈ [H¹(Ω)]²
    - Required for membrane strain ε = ½(∇u + ∇uᵀ) + … ∈ [L²(Ω)]⁴
    - Required for nonlinear term ∇w ⊗ ∇w ∈ [L²(Ω)]⁴
  Transverse displacement:  w ∈ H¹(Ω)
    - Required for shear strain γ = ∇w - θ ∈ [L²(Ω)]²
  Rotations:               (θ_x, θ_y) ∈ [H¹(Ω)]²
    - Required for curvature κ = ½(∇θ + ∇θᵀ) ∈ [L²(Ω)]⁴
  Velocity:                (u̇₀, v̇₀, ẇ, θ̇_x, θ̇_y) ∈ [H¹(Ω)]⁵  (Rayleigh damping)
  Acceleration:            (ü₀, v̈₀, ẅ, θ̈_x, θ̈_y) ∈ [L²(Ω)]⁵
  Manufactured load:       f ∈ [L²(Ω)]⁵  (body-force density)

Finite-dimensional discrete spaces in production FEniCS model:
  (u₀, v₀) : CG2 vector element on triangular mesh
  w         : CG2 scalar element
  (θ_x, θ_y): CG2 vector element
  Test space : same as trial space (Galerkin)

The manufactured fields below are C^∞(Ω) ∩ C^∞([0,T]).  Every required norm
and derivative is bounded on Ω × [0,T].

Particle trajectory regularity:
  X_p*(t), Γ_p*(t), σ_p*(t) ∈ C^∞([0,T])  for all particles p.
  Position RHS G_x requires that the velocity field u evaluated at X_p*(t) is
  finite and smooth, which holds for smooth, finitely-supported vorticity fields
  with Gaussian-erf kernel.

Interface traction:
  t* ∈ [L²(Γ_interface)]³  for the weak force transfer.
  Smooth prescribed traction; independently integrated against FEM test functions.

COORDINATE SYSTEM
-----------------
  x : chordwise direction (along root chord Lx)
  y : spanwise direction  (along span Ly)
  z : normal to plate mid-surface (positive upward)
  t : physical time (seconds)

The plate is clamped at y=0 (root), free on remaining edges.

UNITS
-----
All quantities in SI:  m, kg, s, Pa, N, rad.
Amplitudes are set to small fractions of Lx/Ly to satisfy small-strain
assumptions consistent with the production physical configuration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
try:
    import sympy as sp
except ImportError:
    sp = None
import yaml

# ---------------------------------------------------------------------------
# 1. SPATIAL ENVELOPE  B(x, y) = sin⁴(πx/Lx) sin⁴(πy/Ly)
#    Fourth-order zeros ensure homogeneous Dirichlet + Neumann data at all edges.
# ---------------------------------------------------------------------------

def envelope(x: float, y: float, Lx: float, Ly: float) -> dict:
    """Return B and its first/second spatial partial derivatives.

    The choice of fourth power guarantees:
      B = 0, B_x = 0, B_xx = 0  at x = 0, Lx
      B = 0, B_y = 0, B_yy = 0  at y = 0, Ly
    so both Dirichlet and normal-derivative (Neumann) boundary conditions are
    automatically satisfied by any field of the form A * B(x,y) * g(t).
    """
    kx = math.pi / float(Lx)
    ky = math.pi / float(Ly)
    sx = math.sin(kx * x)
    cx = math.cos(kx * x)
    sy = math.sin(ky * y)
    cy = math.cos(ky * y)
    fx = sx ** 4
    fy = sy ** 4
    fx1 = 4.0 * kx * sx ** 3 * cx
    fy1 = 4.0 * ky * sy ** 3 * cy
    fx2 = 4.0 * kx ** 2 * (3.0 * sx ** 2 * cx ** 2 - sx ** 4)
    fy2 = 4.0 * ky ** 2 * (3.0 * sy ** 2 * cy ** 2 - sy ** 4)
    return {
        "B":   fx * fy,
        "B_x": fx1 * fy,
        "B_y": fx * fy1,
        "B_xx": fx2 * fy,
        "B_xy": fx1 * fy1,
        "B_yy": fx * fy2,
    }


# ---------------------------------------------------------------------------
# 2. TEMPORAL MODULATION  g(t) = 1 − cos(Ω t)   (compatible zero ICs)
#
#    g(0) = 0, ġ(0) = 0  → zero initial displacement and velocity.
# ---------------------------------------------------------------------------

def temporal(amplitude: float, omega: float, mode: str, t: float, order: int = 0) -> float:
    """Return the temporal factor and its time derivatives.

    Parameters
    ----------
    mode : 'ramp'   → (1 − cos Ωt), zero IC for q and q̇
           'cos'    → cos Ωt
           'sin'    → sin Ωt
    order : 0 = value, 1 = first derivative, 2 = second derivative.
    """
    phase = float(omega) * float(t)
    amp = float(amplitude)
    mode = str(mode)
    if mode == "ramp":
        vals = (amp * (1.0 - math.cos(phase)),
                amp * omega * math.sin(phase),
                amp * omega ** 2 * math.cos(phase))
    elif mode == "cos":
        vals = (amp * math.cos(phase),
                -amp * omega * math.sin(phase),
                -amp * omega ** 2 * math.cos(phase))
    elif mode == "sin":
        vals = (amp * math.sin(phase),
                amp * omega * math.cos(phase),
                -amp * omega ** 2 * math.sin(phase))
    else:
        raise ValueError(f"unsupported temporal mode {mode!r}")
    if order not in (0, 1, 2):
        raise ValueError("temporal derivative order must be 0, 1, or 2")
    return vals[order]


# ---------------------------------------------------------------------------
# 3. STRUCTURAL TARGET  q* = (u₀*, v₀*, w*, θ_x*, θ_y*)
# ---------------------------------------------------------------------------

#  Default component specification.  Every component uses a separate
#  frequency so that the five governing equations are exercised
#  independently.  All use mode='ramp' for zero initial conditions.
DEFAULT_COMPONENTS = {
    "u":       {"amplitude": 1.0e-4, "omega": 2.3,  "mode": "ramp"},
    "v":       {"amplitude": 1.0e-4, "omega": 1.9,  "mode": "ramp"},
    "w":       {"amplitude": 1.0e-3, "omega": 2.3,  "mode": "ramp"},
    "theta_x": {"amplitude": 1.0e-3, "omega": 2.7,  "mode": "ramp"},
    "theta_y": {"amplitude":-1.0e-3, "omega": 2.3,  "mode": "ramp"},
}


@dataclass(frozen=True)
class StructuralTarget:
    """Analytical structural state target q* = (u₀, v₀, w, θ_x, θ_y).

    Function spaces: each component lies in H¹(Ω) ∩ C^∞(Ω) for all t.
    The spatial envelope B ∈ C^∞(Ω̄) guarantees the required regularity.

    Boundary conditions satisfied (by construction):
      q*(x, 0, t) = 0  for all x (clamped root, y = 0)
      N·n = 0  on free edges  (B vanishes at x=0, Lx, y=Ly)
      M·n = 0  on free edges
      (Q + ∂P/∂x)·n = 0 on free edges
    """
    Lx: float
    Ly: float
    components: dict

    def q(self, x: float, y: float, t: float) -> np.ndarray:
        """State vector q* at (x, y, t); shape (5,)."""
        B = envelope(x, y, self.Lx, self.Ly)["B"]
        return np.array([
            B * temporal(**self.components[name], t=t, order=0)
            for name in ("u", "v", "w", "theta_x", "theta_y")
        ])

    def q_t(self, x: float, y: float, t: float) -> np.ndarray:
        """First time derivative q̇* at (x, y, t); shape (5,)."""
        B = envelope(x, y, self.Lx, self.Ly)["B"]
        return np.array([
            B * temporal(**self.components[name], t=t, order=1)
            for name in ("u", "v", "w", "theta_x", "theta_y")
        ])

    def q_tt(self, x: float, y: float, t: float) -> np.ndarray:
        """Second time derivative q̈* at (x, y, t); shape (5,)."""
        B = envelope(x, y, self.Lx, self.Ly)["B"]
        return np.array([
            B * temporal(**self.components[name], t=t, order=2)
            for name in ("u", "v", "w", "theta_x", "theta_y")
        ])

    def all_derivatives(self, x: float, y: float, t: float
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (q*, q̇*, q̈*) at (x, y, t)."""
        return self.q(x, y, t), self.q_t(x, y, t), self.q_tt(x, y, t)


# ---------------------------------------------------------------------------
# 4. INDEPENDENT STRUCTURAL SOURCE (SymPy derivation)
#    S* = M q̈* + C q̇* + K(q*)
#    derived independently from the strong Reissner–Mindlin equations.
# ---------------------------------------------------------------------------

def _make_sympy_symbols():
    if sp is None:
        raise ImportError(
            "SymPy is required for symbolic plate MMS derivation. "
            "Please run inside an environment with sympy installed (e.g. fenics-env)."
        )
    return sp.symbols("x y t", real=True)


def build_plate_mms_sympy(config: dict, nonlinear: bool = True,
                           include_quadratic_inplane: bool = True,
                           components: dict = None) -> "PlateMMS":
    """Derive manufactured source symbolically from the strong Reissner–Mindlin equations.

    This function does NOT call any FEniCS/DOLFIN operator.  It evaluates
    the governing differential equations symbolically on the prescribed target
    and returns the resulting load vector.

    Strong equilibrium equations (per unit area of mid-surface):
      ρh ü₀ + η_m ρh u̇₀ - div(P_mem) = f_u
      ρh ẅ  + η_m ρh ẇ  - div(Q + P_w) = f_w
      ρh³/12 θ̈ + η_m ρh³/12 θ̇ - div(M) + Q = f_θ

    where the stiffness damping acts on the LINEAR reference stiffness applied
    to the velocity field (Rayleigh K₀ ẋ, not the tangent stiffness K(q)ẋ).

    Parameters
    ----------
    config  : solid_params.yaml as a dict.
    nonlinear : include Green–Lagrange nonlinear membrane terms.
    include_quadratic_inplane : include quadratic in-plane strain correction
                                (matches production include_quadratic_inplane_strain).
    components : component specification dict; defaults to DEFAULT_COMPONENTS.
    """
    if components is None:
        components = DEFAULT_COMPONENTS

    x, y, t = _make_sympy_symbols()
    Lx = sp.Float(config["root_chord"])
    Ly = sp.Float(config["span"])
    E  = sp.Float(config["E"])
    nu = sp.Float(config["nu"])
    rho = sp.Float(config["rho_s"])
    h  = Lx * sp.Float(config["thickness_ratio"])
    eta_m = sp.Float(config["eta_m"])
    eta_k = sp.Float(config["eta_k"])
    kappa = sp.Float(config["kappa_shear"])

    # --- Target fields
    bump = sp.sin(sp.pi * x / Lx) ** 4 * sp.sin(sp.pi * y / Ly) ** 4

    def sympy_field(name):
        c = components[name]
        A, om, mode = sp.Float(c["amplitude"]), sp.Float(c["omega"]), c["mode"]
        phase = om * t
        if mode == "ramp":
            g = 1 - sp.cos(phase)
        elif mode == "cos":
            g = sp.cos(phase)
        elif mode == "sin":
            g = sp.sin(phase)
        else:
            raise ValueError(f"unsupported mode {mode!r}")
        return A * bump * g

    u = sp.Matrix([sympy_field("u"), sympy_field("v")])
    w = sympy_field("w")
    theta = sp.Matrix([sympy_field("theta_x"), sympy_field("theta_y")])

    # --- Kinematics
    grad_u = u.jacobian([x, y])
    grad_w = sp.Matrix([sp.diff(w, x), sp.diff(w, y)])
    # Linear membrane strain
    eps_lin = (grad_u + grad_u.T) / 2
    eps = eps_lin
    if nonlinear:
        eps = eps + grad_w * grad_w.T / 2
        if include_quadratic_inplane:
            eps = eps + grad_u.T * grad_u / 2

    # Constitutive
    Cmem = E * h / (1 - nu ** 2)
    N = Cmem * ((1 - nu) * eps + nu * sp.trace(eps) * sp.eye(2))

    D = E * h ** 3 / (12 * (1 - nu ** 2))
    grad_theta = theta.jacobian([x, y])
    curvature = (grad_theta + grad_theta.T) / 2
    M = D * ((1 - nu) * curvature + nu * sp.trace(curvature) * sp.eye(2))

    G_shear = E / (2 * (1 + nu))
    shear = grad_w - theta
    Q = kappa * G_shear * h * shear

    # --- Internal force divergences
    if nonlinear:
        # First Piola–Kirchhoff equivalent: P = N + (∇u)N
        Pmem = N + grad_u * N
        P_w = N * grad_w
    else:
        Pmem = N
        P_w = sp.zeros(2, 1)

    def _div_rows(T):
        return sp.Matrix([sp.diff(T[i, 0], x) + sp.diff(T[i, 1], y) for i in range(2)])

    internal_u = -_div_rows(Pmem)
    internal_w = -sp.diff(P_w[0] + Q[0], x) - sp.diff(P_w[1] + Q[1], y)
    internal_theta = -_div_rows(M) - Q

    # --- Rayleigh stiffness damping  (uses LINEAR reference stiffness on q̇)
    u_t = u.diff(t)
    w_t = sp.diff(w, t)
    theta_t = theta.diff(t)

    grad_ut = u_t.jacobian([x, y])
    eps_lin_dot = (grad_ut + grad_ut.T) / 2
    N_dot = Cmem * ((1 - nu) * eps_lin_dot + nu * sp.trace(eps_lin_dot) * sp.eye(2))

    grad_wt = sp.Matrix([sp.diff(w_t, x), sp.diff(w_t, y)])
    Q_dot = kappa * G_shear * h * (grad_wt - theta_t)

    grad_tt = theta_t.jacobian([x, y])
    kap_dot = (grad_tt + grad_tt.T) / 2
    M_dot = D * ((1 - nu) * kap_dot + nu * sp.trace(kap_dot) * sp.eye(2))

    damp_u_stiff  = -_div_rows(N_dot)
    damp_w_stiff  = -sp.diff(Q_dot[0], x) - sp.diff(Q_dot[1], y)
    damp_theta_stiff = -_div_rows(M_dot) - Q_dot

    # --- Inertia
    mass = rho * h
    rotary_mass = rho * h ** 3 / 12

    # --- Manufactured load: S* = ρh q̈* + (damping) + (internal force div)
    f_u = (mass * u.diff(t, 2)
           + eta_m * mass * u_t
           + eta_k * damp_u_stiff
           + internal_u)
    f_w = (mass * sp.diff(w, t, 2)
           + eta_m * mass * w_t
           + eta_k * damp_w_stiff
           + internal_w)
    f_theta = (rotary_mass * theta.diff(t, 2)
               + eta_m * rotary_mass * theta_t
               + eta_k * damp_theta_stiff
               + internal_theta)

    # Factor common terms
    f_u = f_u.applyfunc(sp.factor_terms)
    f_w = sp.factor_terms(f_w)
    f_theta = f_theta.applyfunc(sp.factor_terms)

    # --- Boundary checks (all must be zero for free-edge + clamped conditions)
    boundary_checks = {
        "u_at_root":         [sp.simplify(v.subs(y, 0)) for v in u],
        "w_at_root":          sp.simplify(w.subs(y, 0)),
        "theta_at_root":     [sp.simplify(v.subs(y, 0)) for v in theta],
        "membrane_x0":       [sp.simplify(Pmem[i, 0].subs(x, 0)) for i in range(2)],
        "membrane_xL":       [sp.simplify(Pmem[i, 0].subs(x, Lx)) for i in range(2)],
        "membrane_yL":       [sp.simplify(Pmem[i, 1].subs(y, Ly)) for i in range(2)],
        "moment_x0":         [sp.simplify(M[i, 0].subs(x, 0)) for i in range(2)],
        "moment_xL":         [sp.simplify(M[i, 0].subs(x, Lx)) for i in range(2)],
        "moment_yL":         [sp.simplify(M[i, 1].subs(y, Ly)) for i in range(2)],
        "shear_x0":          [sp.simplify((P_w[i] + Q[i]).subs(x, 0)) for i in range(2)],
        "shear_xL":          [sp.simplify((P_w[i] + Q[i]).subs(x, Lx)) for i in range(2)],
        "shear_yL":          [sp.simplify((P_w[i] + Q[i]).subs(y, Ly)) for i in range(2)],
    }

    return PlateMMS(
        x=x, y=y, t=t,
        fields=(u, w, theta),
        loads=(f_u, f_w, f_theta),
        kinematics={
            "membrane_strain": eps,
            "curvature": curvature,
            "shear_strain": shear,
            "membrane_resultant": N,
            "bending_moment": M,
            "shear_resultant": Q,
        },
        boundary_checks=boundary_checks,
        target_derivatives={
            "q":    (u, w, theta),
            "q_t":  (u.diff(t), sp.diff(w, t), theta.diff(t)),
            "q_tt": (u.diff(t, 2), sp.diff(w, t, 2), theta.diff(t, 2)),
        },
        load_components={
            "inertia": (mass * u.diff(t, 2), mass * sp.diff(w, t, 2),
                        rotary_mass * theta.diff(t, 2)),
            "mass_damping": (eta_m * mass * u_t, eta_m * mass * w_t,
                             eta_m * rotary_mass * theta_t),
            "reference_stiffness_damping": (
                eta_k * damp_u_stiff, eta_k * damp_w_stiff, eta_k * damp_theta_stiff),
            "membrane": (internal_u, sp.Integer(0), sp.zeros(2, 1)),
            "transverse": (sp.zeros(2, 1), internal_w, sp.Integer(0)),
            "bending": (sp.zeros(2, 1), sp.Integer(0), -_div_rows(M)),
            "shear": (sp.zeros(2, 1), -sp.diff(Q[0], x) - sp.diff(Q[1], y), -Q),
        },
    )


@dataclass
class PlateMMS:
    """Container for all symbolic plate MMS quantities."""
    x: Any
    y: Any
    t: Any
    fields: tuple          # (u, w, theta) SymPy expressions
    loads: tuple           # (f_u, f_w, f_theta)  — independent strong-form loads
    kinematics: dict
    boundary_checks: dict
    target_derivatives: dict
    load_components: dict

    @property
    def velocities(self):
        return self.target_derivatives["q_t"]

    @property
    def accelerations(self):
        return self.target_derivatives["q_tt"]


# ---------------------------------------------------------------------------
# 5. GEOMETRY TARGET  (structural → aerodynamic coordinate map)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GeometryTarget:
    """Maps structural mid-surface displacements to aerodynamic surface points.

    Implements the linearized interface map used by the production geometry
    transfer: the bound-vortex station displacement includes the in-plane
    translation plus a linearized rotation about the bound-vortex chord
    station x_bound.

    Mathematical form:
      X*(x, y, t) = (x + u₀*, y + v₀*, w* + θ_x*(x − x_bound))
    """
    structural: StructuralTarget
    sweep: float
    x_bound: float

    def point(self, x: float, y: float, t: float) -> np.ndarray:
        """Deformed surface point in global coordinates."""
        q = self.structural.q(x, y, t)
        return np.array([x + q[0], y + q[1], q[2] + q[3] * (x - self.x_bound)])

    def velocity(self, x: float, y: float, t: float) -> np.ndarray:
        """Material velocity of the surface point."""
        q_t = self.structural.q_t(x, y, t)
        return np.array([q_t[0], q_t[1], q_t[2] + q_t[3] * (x - self.x_bound)])


# ---------------------------------------------------------------------------
# 6. INTERFACE TRACTION TARGET  t*(η_s, η_c, t)
#
#    Independent of CRM/RBF transfer.  Used to construct exact panel forces
#    and exact FEM nodal loads for transfer MMS.
# ---------------------------------------------------------------------------

def interface_traction(eta_s: float, eta_c: float, t: float,
                        T_xyz: tuple = (1.0e-2, 5.0e-3, 2.0e-2),
                        omega_t: float = 2.3) -> np.ndarray:
    """Smooth analytical interface traction in normalized panel coordinates.

    Parameters
    ----------
    eta_s : normalized spanwise coordinate ∈ [0, 1].
    eta_c : normalized chordwise coordinate ∈ [0, 1].
    t     : physical time (s).
    T_xyz : traction amplitudes (N/m²) in (x, y, z).
    omega_t : temporal frequency (rad/s).

    Returns
    -------
    t* : ndarray shape (3,) in N/m².

    Notes
    -----
    This traction is used only in the standalone transfer MMS check.
    For the coupled MMS, the structural source uses the Kutta–Joukowski
    panel force derived from BoundCirculationTarget.
    """
    Tx, Ty, Tz = T_xyz
    s = math.sin(omega_t * t)
    return np.array([
        Tx * math.sin(2.0 * math.pi * eta_s) * math.cos(math.pi * eta_c) * s,
        Ty * math.cos(2.0 * math.pi * eta_s) * math.sin(math.pi * eta_c) * s,
        Tz * math.sin(math.pi * eta_s) * math.sin(math.pi * eta_c) * s,
    ])


def exact_panel_force(eta_s_lo: float, eta_s_hi: float,
                       t: float,
                       T_xyz: tuple = (1.0e-2, 5.0e-3, 2.0e-2),
                       omega_t: float = 2.3,
                       n_quad: int = 32) -> np.ndarray:
    """Gauss–Legendre integral of t* over one spanwise panel.

    Parameters
    ----------
    eta_s_lo, eta_s_hi : panel spanwise extent in normalized coordinates.
    n_quad : number of quadrature points (each direction).

    Returns
    -------
    F* : ndarray shape (3,) in N (traction × panel area).
    """
    nodes, weights = np.polynomial.legendre.leggauss(n_quad)
    # Map [−1, 1] → [eta_s_lo, eta_s_hi] and [0, 1] chordwise
    s_mid = 0.5 * (eta_s_lo + eta_s_hi)
    s_half = 0.5 * (eta_s_hi - eta_s_lo)
    c_mid, c_half = 0.5, 0.5

    force = np.zeros(3)
    for ws, xs in zip(weights, nodes):
        eta_s = s_mid + s_half * xs
        for wc, xc in zip(weights, nodes):
            eta_c = c_mid + c_half * xc
            t_star = interface_traction(eta_s, eta_c, t, T_xyz, omega_t)
            force += ws * wc * t_star * s_half * c_half
    return force


# ---------------------------------------------------------------------------
# 7. PARTICLE TARGET  (discrete rVPM MMS; not continuum N–S)
#
#    SCOPE: This is a particle-level discrete-model MMS.
#    It does NOT claim to verify the continuum Navier–Stokes equations.
#    It verifies that the production RK3 integrator correctly advances
#    particle positions, circulations, and core widths subject to an
#    independently derived manufactured source.
# ---------------------------------------------------------------------------

@dataclass
class ParticlePopulation:
    """Manufactured particle target population.

    Trajectory parametrization:
      X_p*(t) = X0_p + a_p sin(ω_x,p t) + b_p (1 − cos(ω_x,p t))
      Γ_p*(t) = Γ0_p + c_p sin(ω_Γ,p t) + d_p (1 − cos(ω_Γ,p t))
      σ_p*(t) = σ0_p + e_p (1 − cos(ω_σ,p t))

    All trajectories are C^∞([0,T]).  Initial conditions:
      X_p*(0) = X0_p, Ẋ_p*(0) = ω_x,p a_p
      Γ_p*(0) = Γ0_p, Γ̇_p*(0) = ω_Γ,p c_p
      σ_p*(0) = σ0_p, σ̇_p*(0) = 0

    Spatial refinement: scale the number of particles N with the refinement
    level.  Maintain constant core-overlap ratio λ_σ = σ₀/h_p where h_p is
    the initial inter-particle spacing.

    Parameters
    ----------
    n_particles : number of target particles.
    Ly, Lx     : domain dimensions for initial layout.
    sigma0_ref  : reference core width at level 1 (m).
    lambda_sigma: core-overlap ratio (dimensionless).  σ₀ = λ_σ * h_p.
    """

    ids: np.ndarray           # (N,) int unique IDs
    x0: np.ndarray            # (N, 3) initial positions
    gamma0: np.ndarray        # (N, 3) initial circulations
    sigma0: np.ndarray        # (N,) initial core widths
    a: np.ndarray             # (N, 3) position amplitude 1
    b: np.ndarray             # (N, 3) position amplitude 2
    c: np.ndarray             # (N, 3) circulation amplitude 1
    d: np.ndarray             # (N, 3) circulation amplitude 2
    e: np.ndarray             # (N,)   core modulation amplitude
    omega_x: np.ndarray       # (N,)   position frequency
    omega_gamma: np.ndarray   # (N,)   circulation frequency
    omega_sigma: np.ndarray   # (N,)   core frequency
    lambda_sigma: float = 2.0 # core-overlap ratio (dimensionless)

    def state(self, t: float):
        """Return (X*, Γ*, σ*) at time t."""
        t = float(t)
        tx = self.omega_x[:, None] * t
        tg = self.omega_gamma[:, None] * t
        ts = self.omega_sigma * t
        X = self.x0 + self.a * np.sin(tx) + self.b * (1.0 - np.cos(tx))
        Gamma = self.gamma0 + self.c * np.sin(tg) + self.d * (1.0 - np.cos(tg))
        sigma = self.sigma0 + self.e * (1.0 - np.cos(ts))
        return X, Gamma, sigma

    def derivatives(self, t: float):
        """Return (Ẋ*, Γ̇*, σ̇*) at time t."""
        t = float(t)
        tx = self.omega_x[:, None] * t
        tg = self.omega_gamma[:, None] * t
        ts = self.omega_sigma * t
        X_t = self.omega_x[:, None] * (self.a * np.cos(tx) + self.b * np.sin(tx))
        G_t = self.omega_gamma[:, None] * (self.c * np.cos(tg) + self.d * np.sin(tg))
        s_t = self.e * self.omega_sigma * np.sin(ts)
        return X_t, G_t, s_t

    @property
    def n(self) -> int:
        return len(self.ids)

    def to_dict(self) -> dict:
        return {k: np.asarray(getattr(self, k)).tolist() for k in
                ("ids", "x0", "gamma0", "sigma0", "a", "b", "c", "d", "e",
                 "omega_x", "omega_gamma", "omega_sigma")}

    @classmethod
    def from_dict(cls, data: dict, lambda_sigma: float = 2.0) -> "ParticlePopulation":
        arrays = {k: np.asarray(data[k], dtype=(int if k == "ids" else float))
                  for k in ("ids", "x0", "gamma0", "sigma0", "a", "b", "c", "d", "e",
                             "omega_x", "omega_gamma", "omega_sigma")}
        return cls(**arrays, lambda_sigma=lambda_sigma)

    @staticmethod
    def build(n_particles: int, Lx: float, Ly: float, sigma0: float,
              lambda_sigma: float = 2.0) -> "ParticlePopulation":
        """Construct a deterministic population of n_particles.

        Particles are arranged in a regular grid downstream of the wing.
        Position frequency = 2.0 rad/s, circulation frequency = 1.7 rad/s,
        core frequency = 1.3 rad/s — all incommensurable to prevent standing modes.
        """
        rng = np.random.default_rng(seed=42)
        n = int(n_particles)
        # Downstream wake positions: x > Lx, spread across span
        x0 = np.column_stack([
            np.full(n, 1.5 * Lx) + rng.uniform(-0.1 * Lx, 0.1 * Lx, n),
            np.linspace(0.1 * Ly, 0.9 * Ly, n),
            rng.uniform(-0.05 * Lx, 0.05 * Lx, n),
        ])
        gamma0 = np.tile([0.0, 0.0, 1.0e-4], (n, 1)) * rng.uniform(0.8, 1.2, (n, 3))
        sigma0_arr = np.full(n, sigma0)
        # Position modulation: physically small relative to Lx
        a = rng.uniform(-0.01 * Lx, 0.01 * Lx, (n, 3))
        b = rng.uniform(-0.005 * Lx, 0.005 * Lx, (n, 3))
        # Circulation modulation: physically small
        c = np.zeros((n, 3))
        d = np.tile([0.0, 0.0, 1.0e-5], (n, 1))
        # Core modulation: 10% of σ₀
        e = np.full(n, 0.1 * sigma0)
        return ParticlePopulation(
            ids=np.arange(1, n + 1, dtype=int),
            x0=x0, gamma0=gamma0, sigma0=sigma0_arr,
            a=a, b=b, c=c, d=d, e=e,
            omega_x=np.full(n, 2.0),
            omega_gamma=np.full(n, 1.7),
            omega_sigma=np.full(n, 1.3),
            lambda_sigma=lambda_sigma,
        )


# ---------------------------------------------------------------------------
# 8. PARTICLE SOURCE DEFINITIONS  b_p* = Ẏ_p* − G_p^independent(Y_p*, t)
#
#    Position source: b_x,p = Ẋ_p* − (U_kernel(X_p*, Γ_all, σ_all) + U_inf)
#    Circulation source: b_Γ,p = Γ̇_p* − S_p^independent(X_p*, Γ_p*, σ_p*)
#    Core source: b_σ,p = σ̇_p* − C_σ(σ_p*, ν, t)  [= 0 for inviscid]
#
#    INDEPENDENCE REQUIREMENT: circulation/stretching RHS is derived from
#    the rVPM equations directly, NOT by calling vpm.update_particle_states.
# ---------------------------------------------------------------------------

def rvpm_kernel_g(r_sigma: float) -> float:
    """Regularization function g(r/σ) for the Gaussian-erf kernel.

    For the FLOWVPM gaussianerf kernel:
      g(ρ) = erf(ρ/√2) − √(2/π) · ρ · exp(−ρ²/2)
    This matches FLOWVPM.g_gauserf exactly.

    Returns g(r/σ) used in the Biot-Savart-like velocity reconstruction:
      u_p(x) = Σ_q (1/4π) g(|x−x_q|/σ_q) / |x−x_q|³ × (Γ_q × (x−x_q))
    """
    rho = float(r_sigma)
    sqrt2 = math.sqrt(2.0)
    sqrt2_pi = math.sqrt(2.0 / math.pi)
    return math.erf(rho / sqrt2) - sqrt2_pi * rho * math.exp(-0.5 * rho ** 2)


def biot_savart_induced_velocity(x_target: np.ndarray,
                                  X: np.ndarray,
                                  Gamma: np.ndarray,
                                  sigma: np.ndarray) -> np.ndarray:
    """Direct summation O(N²) Biot-Savart reconstruction.

    Implements the standard regularized Biot-Savart for the Gaussian-erf kernel:
      u(x) = (1/4π) Σ_p g(r_p/σ_p) / r_p³ × (Γ_p × (x − x_p))

    This is derived INDEPENDENTLY of FLOWVPM's UJ operators for use in
    the manufactured source computation.  Used only for small-N (≤ 100)
    source verification cases.  Production runs use FLOWVPM's FMM.

    Parameters
    ----------
    x_target : (3,) evaluation point.
    X, Gamma, sigma : (N, 3), (N, 3), (N,) particle arrays.
    """
    u = np.zeros(3)
    inv4pi = 1.0 / (4.0 * math.pi)
    for i in range(len(sigma)):
        dx = x_target - X[i]
        r2 = float(np.dot(dx, dx))
        if r2 < 1.0e-28:
            continue
        r = math.sqrt(r2)
        gfac = rvpm_kernel_g(r / float(sigma[i]))
        cross = np.cross(Gamma[i], dx)
        u += inv4pi * gfac / (r2 * r) * cross
    return u


def independent_rvpm_stretching_rhs(X: np.ndarray,
                                     Gamma: np.ndarray,
                                     sigma: np.ndarray,
                                     f_rvpm: float = 1.0 / 2.0,
                                     g_rvpm: float = 5.0 / 4.0,
                                     return_core_rate: bool = False):
    """Independent implementation of the rVPM vortex stretching RHS.

    Computes dΓ_p/dt = S_p − 3 Z_p Γ_p  (reformulated VPM formulation)
    and stretching-induced core evolution rate dσ_p/dt = −σ_p Z_p
    where:
      S_p = (∇u)^T Γ_p  (transposed stretching, production default)
            or (∇u) Γ_p  (classic stretching)
      Z_p = [(f+g)/(1+3f)] S_p·Γ_p/|Γ_p|²

    This function is derived directly from the rVPM formulation in
    Alvarez & Ning (2022), NOT by calling vpm.update_particle_states.

    Parameters
    ----------
    X, Gamma, sigma : particle state at the current stage time.
    f_rvpm, g_rvpm  : rVPM formulation constants (default: transposed rVPM).
    return_core_rate: if True, returns tuple (dGamma_dt, dsigma_dt).

    Returns
    -------
    dGamma_dt : (N, 3) independent stretching RHS.
    (optional) dsigma_dt : (N,) stretching-induced core evolution RHS.
    """
    N = len(sigma)
    dGamma_dt = np.zeros((N, 3))
    dsigma_dt = np.zeros(N)
    h_fd = 1.0e-5  # finite-difference step for Jacobian
    # Compute u at all particle positions
    U = np.array([biot_savart_induced_velocity(X[p], X, Gamma, sigma) for p in range(N)])
    # Velocity Jacobian at each particle by FD
    for p in range(N):
        J = np.zeros((3, 3))
        for k in range(3):
            dxk = np.zeros(3)
            dxk[k] = h_fd
            up = biot_savart_induced_velocity(X[p] + dxk, X, Gamma, sigma)
            um = biot_savart_induced_velocity(X[p] - dxk, X, Gamma, sigma)
            J[:, k] = (up - um) / (2.0 * h_fd)
        # Transposed stretching: S = J^T Γ_p
        S = J.T @ Gamma[p]
        G2 = float(np.dot(Gamma[p], Gamma[p]))
        Z = 0.0
        if G2 > 0.0:
            Z = (f_rvpm + g_rvpm) / (1.0 + 3.0 * f_rvpm) * np.dot(S, Gamma[p]) / G2
        # rVPM: dΓ/dt = S − 3 Z Γ  (no SFS term for inviscid reference config)
        dGamma_dt[p] = S - 3.0 * Z * Gamma[p]
        # rVPM core-evolution: dσ/dt = −σ Z
        dsigma_dt[p] = -sigma[p] * Z
    if return_core_rate:
        return dGamma_dt, dsigma_dt
    return dGamma_dt


# ---------------------------------------------------------------------------
# 9. ERROR NORM DEFINITIONS
#
#    All norms are defined here mathematically to ensure consistent usage
#    across mms_driver.py and mms_rvpm.jl.
# ---------------------------------------------------------------------------

class ErrorNorms:
    """Mathematical definitions of all error norms used in the MMS.

    Structural (FEM quadrature, relative L²):
      E_L2(q) = (∫_Ω ‖q_h − q*‖² dΩ / ∫_Ω ‖q*‖² dΩ)^(1/2)
      Space-time: integrate over [0, T] with trapezoidal weights.

    Particle (discrete, per-component):
      e_x  = (Σ_p ‖X_p^h − X_p*‖² / Σ_p ‖X_p*‖²)^(1/2)
      e_Γ  = (Σ_p ‖Γ_p^h − Γ_p*‖² / Σ_p ‖Γ_p*‖²)^(1/2)
      e_σ  = (Σ_p |σ_p^h − σ_p*|² / Σ_p |σ_p*|²)^(1/2)

    Reconstructed field (probe-based):
      e_u  = ‖u_h(x_probe) − u*(x_probe)‖₂ / ‖u*(x_probe)‖₂
      e_ω  = ‖ω_h(x_probe) − ω*(x_probe)‖₂ / ‖ω*(x_probe)‖₂

    Interface transfer:
      e_F  = ‖F_h − F*‖₂ / ‖F*‖₂    (resultant force)
      e_M  = ‖M_h − M*‖₂ / ‖M*‖₂    (resultant moment)
      e_W  = |W_fluid − W_struct| / |W_struct|   (virtual-work defect)

    Observed spatial order:
      p_l = log(E_l / E_{l+1}) / log(h_l / h_{l+1})
    """

    @staticmethod
    def relative_l2(numerical: np.ndarray, reference: np.ndarray,
                    weights: np.ndarray = None) -> float:
        """Weighted relative L² error.

        Parameters
        ----------
        numerical, reference : arrays of equal shape.
        weights : optional quadrature weights; uniform if None.
        """
        num = np.asarray(numerical, dtype=float).ravel()
        ref = np.asarray(reference, dtype=float).ravel()
        w = np.ones(len(num)) if weights is None else np.asarray(weights, dtype=float).ravel()
        err_sq = float(np.dot(w, (num - ref) ** 2))
        ref_sq = float(np.dot(w, ref ** 2))
        if ref_sq < 1.0e-30:
            return math.sqrt(err_sq)  # absolute norm if reference is zero
        return math.sqrt(err_sq / ref_sq)

    @staticmethod
    def observed_order(coarse_error: float, fine_error: float,
                       refinement_ratio_or_h1: float, h2: Optional[float] = None) -> float:
        """Pairwise convergence order estimate.

        Supports either:
          observed_order(e_coarse, e_fine, ratio)
          observed_order(e_coarse, e_fine, h_coarse, h_fine)

        Returns NaN if inputs are non-positive, non-finite, or ratio <= 1.
        """
        if h2 is not None:
            if h2 <= 0.0 or refinement_ratio_or_h1 <= 0.0:
                return float("nan")
            refinement_ratio = float(refinement_ratio_or_h1) / float(h2)
        else:
            refinement_ratio = float(refinement_ratio_or_h1)

        if (not math.isfinite(coarse_error) or not math.isfinite(fine_error)
                or coarse_error <= 0.0 or fine_error <= 0.0
                or refinement_ratio <= 1.0 or fine_error >= coarse_error):
            return float("nan")
        return math.log(coarse_error / fine_error) / math.log(refinement_ratio)

    @staticmethod
    def virtual_work_defect(W_fluid: float, W_struct: float) -> float:
        """Work conservation defect: |W_fluid − W_struct| / |W_struct|."""
        denom = abs(float(W_struct))
        if denom < 1.0e-20:
            return abs(float(W_fluid) - float(W_struct))
        return abs(float(W_fluid) - float(W_struct)) / denom


# ---------------------------------------------------------------------------
# 10. BOUND-CIRCULATION TARGET  Γ_bound*(t)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BoundCirculationTarget:
    """Analytical spanwise bound-circulation target at VLM stations.

    Γ_bound_i*(t) = Γ0_i + a_i sin(ω t) + b_i (1 − cos(ω t))

    This is NOT forced into the production VLM system (which solves a
    no-penetration problem).  It is used only to prescribe the target
    Kutta–Joukowski panel forces.
    """
    gamma0: np.ndarray
    a: np.ndarray
    b: np.ndarray
    omega: float

    def state(self, t: float) -> np.ndarray:
        phase = float(self.omega) * float(t)
        return (np.asarray(self.gamma0)
                + np.asarray(self.a) * math.sin(phase)
                + np.asarray(self.b) * (1.0 - math.cos(phase)))

    def to_dict(self) -> dict:
        return {"gamma0": np.asarray(self.gamma0).tolist(),
                "a": np.asarray(self.a).tolist(),
                "b": np.asarray(self.b).tolist(),
                "omega": float(self.omega)}


# ---------------------------------------------------------------------------
# 11. PARAMETER LOADING HELPERS
# ---------------------------------------------------------------------------

def load_production_config(config_dir: str | Path) -> dict:
    """Load all three production YAML files from config_dir."""
    config_dir = Path(config_dir)
    configs = {}
    for name in ("solid_params", "fluid_params", "coupling_params"):
        path = config_dir / f"{name}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"Production config not found: {path}")
        with open(path, encoding="utf-8") as fh:
            configs[name] = yaml.safe_load(fh)
    return configs


def load_mms_parameters(mms_yaml: str | Path) -> dict:
    """Load mms_parameters.yaml."""
    path = Path(mms_yaml)
    if not path.exists():
        raise FileNotFoundError(f"MMS parameter file not found: {path}")
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# 12. MANIFEST SERIALIZATION
# ---------------------------------------------------------------------------

import json


def build_mms_manifest(solid: dict, fluid: dict, coupling: dict,
                        mms_params: dict, refinement_level: int = 1) -> dict:
    """Build the JSON manifest shared between Python and Julia processes.

    Parameters
    ----------
    solid, fluid, coupling : production config dicts.
    mms_params : mms_parameters.yaml as a dict.
    refinement_level : spatial refinement factor (1 = coarsest).
    """
    Lx = float(solid["root_chord"])
    Ly = float(solid["span"])
    amp = mms_params["structural"]["amplitudes"]
    frq = mms_params["structural"]["frequencies"]
    components = {
        "u":       {"amplitude": amp["u"],       "omega": frq["u"],       "mode": "ramp"},
        "v":       {"amplitude": amp["v"],       "omega": frq["v"],       "mode": "ramp"},
        "w":       {"amplitude": amp["w"],       "omega": frq["w"],       "mode": "ramp"},
        "theta_x": {"amplitude": amp["theta_x"], "omega": frq["theta_x"], "mode": "ramp"},
        "theta_y": {"amplitude": amp["theta_y"], "omega": frq["theta_y"], "mode": "ramp"},
    }
    structural = StructuralTarget(Lx, Ly, components)

    n_span = int(coupling.get("comm_nspan", fluid.get("n_span", 80))) * refinement_level
    y_stations = (np.arange(n_span) + 0.5) / n_span * Ly

    x_bound = float(mms_params.get("x_bound_fraction", 0.25)) * Lx
    geometry = GeometryTarget(structural, float(solid.get("leading_edge_sweep", 0.0)), x_bound)

    # Particle population scaled with refinement level
    p_params = mms_params["particles"]
    n_particles = int(p_params["n_base"]) * refinement_level
    lambda_sigma = float(p_params["lambda_sigma"])
    h_p = Ly / n_particles  # nominal inter-particle spacing
    sigma0 = lambda_sigma * h_p
    particles = ParticlePopulation.build(n_particles, Lx, Ly, sigma0, lambda_sigma)

    omega_bc = float(frq.get("bound_circulation", frq["w"]))
    bound_circ = BoundCirculationTarget(
        gamma0=np.full(n_span, float(mms_params["bound_circulation"]["gamma0"])),
        a=np.full(n_span, float(mms_params["bound_circulation"]["a"])),
        b=np.full(n_span, float(mms_params["bound_circulation"]["b"])),
        omega=omega_bc,
    )

    freestream = {
        "vinf": float(fluid.get("vinf", 15.0)),
        "aoa_deg": float(fluid.get("aoa_deg", 0.0)),
        "rho": float(fluid.get("rho", 1.225)),
        "nu": float(fluid.get("nu", 1.5e-5)),
        "oscillation": fluid.get("freestream_oscillation", [0.0, 0.0, 0.0]),
        "frequency_hz": float(fluid.get("freestream_frequency_hz", 0.0)),
        "gradient": fluid.get("freestream_gradient", [0.0] * 9),
        "wavenumber": fluid.get("freestream_wavenumber", [0.0] * 3),
        "phase_deg": float(fluid.get("freestream_phase_deg", 0.0)),
        "reference": fluid.get("freestream_reference", [0.0] * 3),
    }

    return {
        "scope": (
            "Particle-level discrete-model MMS; not continuum N-S. "
            "Standalone plate MMS uses SymPy-derived independent strong-form source. "
            "Coupled structural source: independent weak-form body-force assembly."
        ),
        "refinement_level": refinement_level,
        "studies": {
            "particle_temporal": mms_params["particles"]["temporal_study"],
            "particle_population": mms_params["particles"]["population_study"],
            "interface_transfer": mms_params["interface_traction"],
        },
        "lambda_sigma": lambda_sigma,
        "Lx": Lx, "Ly": Ly,
        "n_span": n_span,
        "y_stations": y_stations.tolist(),
        "chord": Lx,
        "tip_chord": float(solid.get("tip_chord", Lx)),
        "sweep": float(solid.get("leading_edge_sweep", 0.0)),
        "x_bound": x_bound,
        "freestream": freestream,
        "moment_origin_m": np.asarray(
            mms_params.get("coupled", {}).get("moment_origin_m", [0.0, 0.0, 0.0]), dtype=float
        ).tolist(),
        "structural": {
            "components": components,
            "function_spaces": {
                "displacement": "H1(Omega)^2 × H1(Omega)",
                "rotation": "H1(Omega)^2",
                "velocity": "H1(Omega)^5",
                "acceleration": "L2(Omega)^5",
                "manufactured_load": "L2(Omega)^5",
                "note": "Target fields are C^inf; required function-space regularity is H1 for displacements/rotations.",
            },
        },
        "particles": particles.to_dict(),
        "bound_circulation": bound_circ.to_dict(),
        "bound_circulation_status": (
            "manufactured RHS is injected into the isolated target-geometry production VLM solve; "
            "the coupled numerical wing circulation is recorded separately"
        ),
        "particle_formulas": {
            "position": "X0 + a*sin(omega_x*t) + b*(1-cos(omega_x*t))",
            "circulation": "Gamma0 + c*sin(omega_gamma*t) + d*(1-cos(omega_gamma*t))",
            "core": "sigma0 + e*(1-cos(omega_sigma*t))",
            "stretching_rhs": "independent rVPM: S - 3*Z*Gamma (not vpm.update_particle_states)",
            "position_rhs": "independent Biot-Savart: Biot_Savart_sum(X*, Gamma*, sigma*) + U_inf",
        },
        "error_norms": {
            "structural": {
                "e_u0": "relative FE spatial L2(Omega_s) error for u0",
                "e_v0": "relative FE spatial L2(Omega_s) error for v0",
                "e_w": "relative FE spatial L2(Omega_s) error for w",
                "e_theta_x": "relative FE spatial L2(Omega_s) error for theta_x",
                "e_theta_y": "relative FE spatial L2(Omega_s) error for theta_y",
                "time_kind": "final time",
            },
            "particle": {
                "e_pos": "relative Frobenius norm of matched position arrays",
                "e_circ": "relative Frobenius norm of matched circulation arrays",
                "e_core": "relative Euclidean norm of matched core arrays",
            },
            "reconstruction": {
                "e_velocity": "relative weighted error at fixed physical probes",
                "e_vorticity": "relative weighted error at fixed physical probes",
            },
            "interface": {
                "e_force": "normalized resultant-force transfer residual",
                "e_moment": "normalized resultant-moment transfer residual about configured origin",
                "e_work": "relative virtual-work transfer defect",
            },
        },
    }


_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent


def write_manifest(manifest: dict, path: str | Path) -> Path:
    """Serialize manifest to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# 13. COUPLED MMS TARGET CONTAINER & INTERACTION HOOKS
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CoupledMMSTarget:
    """Complete coupled target container loaded from manifest JSON."""
    structural: StructuralTarget
    geometry: GeometryTarget
    particles: ParticlePopulation
    bound_circulation: BoundCirculationTarget
    n_span: int
    y_stations: np.ndarray
    chord: float
    tip_chord: float
    freestream: dict
    moment_origin_m: np.ndarray
    T: float = 0.0

    @property
    def panel_edges(self) -> np.ndarray:
        return np.linspace(0.0, float(self.structural.Ly), int(self.n_span) + 1)

    def panel_force(self, t: float) -> np.ndarray:
        """Kutta–Joukowski target force at each bound station at time t."""
        eta = (np.arange(self.n_span, dtype=float) + 0.5) / self.n_span
        gamma = self.bound_circulation.state(t)
        rho = float(self.freestream["rho"])
        speed = float(self.freestream["vinf"])
        aoa = math.radians(float(self.freestream.get("aoa_deg", 0.0)))
        gradient = np.asarray(self.freestream.get("gradient", np.zeros(9)), dtype=float).reshape(3, 3).T
        amplitudes = np.asarray(self.freestream.get("oscillation", np.zeros(3)), dtype=float)
        wave = np.asarray(self.freestream.get("wavenumber", np.zeros(3)), dtype=float)
        reference = np.asarray(self.freestream.get("reference", np.zeros(3)), dtype=float)
        frequency = float(self.freestream.get("frequency_hz", 0.0))
        phase0 = math.radians(float(self.freestream.get("phase_deg", 0.0)))

        force = np.zeros((self.n_span, 3), dtype=float)
        dt = float(t)
        xb = float(self.geometry.x_bound if self.geometry.x_bound is not None else 0.25 * self.structural.Lx)
        for i, eta_i in enumerate(eta):
            y = float(self.y_stations[i])
            x = xb
            surface_velocity = self.geometry.velocity(x, y, dt)
            point = self.geometry.point(x, y, dt)
            xp = point - reference
            base = speed * np.asarray((math.cos(aoa), 0.0, math.sin(aoa)))
            gust_phase = 2.0 * math.pi * frequency * dt + phase0
            gust = amplitudes * np.sin(gust_phase + np.asarray((wave[1] * xp[1], wave[2] * xp[2], wave[0] * xp[0])))
            local_velocity = base + gradient @ xp + gust - surface_velocity
            chord_i = self.tip_chord * (1.0 - eta_i) + self.chord * eta_i
            circulation_line = np.asarray((0.0, chord_i, 0.0))
            force[i] = rho * gamma[i] * np.cross(local_velocity, circulation_line)
        return force

    @classmethod
    def from_manifest(cls, data: dict) -> "CoupledMMSTarget":
        """Reconstruct CoupledMMSTarget from manifest JSON dictionary."""
        Lx = float(data["Lx"])
        Ly = float(data["Ly"])
        structural = StructuralTarget(Lx, Ly, data["structural"]["components"])
        geometry = GeometryTarget(structural, float(data["sweep"]), float(data["x_bound"]))
        p_pop = ParticlePopulation.from_dict(data["particles"], lambda_sigma=float(data.get("lambda_sigma", 2.0)))
        bc_data = data["bound_circulation"]
        bound_circ = BoundCirculationTarget(
            gamma0=np.asarray(bc_data["gamma0"], dtype=float),
            a=np.asarray(bc_data["a"], dtype=float),
            b=np.asarray(bc_data["b"], dtype=float),
            omega=float(bc_data["omega"]),
        )
        return cls(
            structural=structural,
            geometry=geometry,
            particles=p_pop,
            bound_circulation=bound_circ,
            n_span=int(data["n_span"]),
            y_stations=np.asarray(data["y_stations"], dtype=float),
            chord=float(data["chord"]),
            tip_chord=float(data.get("tip_chord", data["chord"])),
            freestream=data["freestream"],
            moment_origin_m=np.asarray(data.get("moment_origin_m", [0.0, 0.0, 0.0]), dtype=float),
            T=float(data.get("T", 0.0)),
        )


def _scalar_expression_code(target: CoupledMMSTarget, name: str, t: float, order: int = 0) -> str:
    spec = target.structural.components[name]
    amp, omega, mode = float(spec["amplitude"]), float(spec["omega"]), spec["mode"]
    phase = f"({omega:.17g}*targ_t)"
    if mode == "cos":
        factors = (f"cos({phase})", f"(-{omega:.17g})*sin({phase})", f"(-{omega**2:.17g})*cos({phase})")
    elif mode == "sin":
        factors = (f"sin({phase})", f"{omega:.17g}*cos({phase})", f"(-{omega**2:.17g})*sin({phase})")
    elif mode == "ramp":
        factors = (f"(1-cos({phase}))", f"{omega:.17g}*sin({phase})", f"{omega**2:.17g}*cos({phase})")
    else:
        raise ValueError(f"unsupported mode {mode!r}")
    Lx, Ly = float(target.structural.Lx), float(target.structural.Ly)
    B = f"pow(sin(pi*x[0]/{Lx:.17g}),4)*pow(sin(pi*x[1]/{Ly:.17g}),4)"
    return f"{amp:.17g}*({B})*({factors[order]})"


def _target_function(model, target: CoupledMMSTarget, t: float, order: int, spaces: list):
    """Interpolate analytical target fields into the production mixed FE space."""
    from dolfin import Expression, interpolate, Function, FunctionAssigner
    Vu = spaces[0][0] if isinstance(spaces[0], tuple) else spaces[0]
    Vw = spaces[1][0] if isinstance(spaces[1], tuple) else spaces[1]
    Vtheta = spaces[2][0] if isinstance(spaces[2], tuple) else spaces[2]

    uexpr = Expression((_scalar_expression_code(target, "u", t, order),
                        _scalar_expression_code(target, "v", t, order)), degree=8, targ_t=float(t))
    wexpr = Expression(_scalar_expression_code(target, "w", t, order), degree=8, targ_t=float(t))
    thetexpr = Expression((_scalar_expression_code(target, "theta_x", t, order),
                           _scalar_expression_code(target, "theta_y", t, order)), degree=8, targ_t=float(t))

    u = interpolate(uexpr, Vu)
    w = interpolate(wexpr, Vw)
    theta = interpolate(thetexpr, Vtheta)

    out = Function(model.V)
    FunctionAssigner(model.V, [Vu, Vw, Vtheta]).assign(out, [u, w, theta])
    return out


class ReferenceForceFunctional:
    """Computes virtual work of aerodynamic target forces on the plate surface."""

    def __init__(self, model, target: CoupledMMSTarget, eta_cp: float = 0.75):
        self.model = model
        self.target = target
        fs = target.freestream
        self.pressure_scale = 0.01 * 0.5 * float(fs["rho"]) * float(fs["vinf"]) ** 2
        self.omega = 2.3
        self.panel_forces = None
        self.panel_edges = np.linspace(0.0, float(target.structural.Ly), int(target.n_span) + 1)
        self.eta_cp = float(eta_cp)
        if not 0.0 < self.eta_cp < 1.0:
            raise ValueError("eta_cp must lie strictly between the leading and trailing edges")

    def set_panel_forces(self, forces: np.ndarray):
        forces = np.asarray(forces, dtype=float).reshape(-1, 3)
        self.panel_forces = forces.copy()

    def work(self, t: float, dq):
        from dolfin import as_vector, dot, UserExpression
        if self.panel_forces is None:
            pressure = self.pressure_scale * math.sin(self.omega * float(t))
            traction = as_vector((0.0, 0.0, pressure))
        else:
            class PanelPressure(UserExpression):
                def __init__(self, parent, **kwargs):
                    self.parent = parent
                    super().__init__(**kwargs)

                def eval(self, values, x):
                    j = min(max(int(np.searchsorted(self.parent.panel_edges, x[1], side="right") - 1), 0),
                            len(self.parent.panel_forces) - 1)
                    y0, y1 = self.parent.panel_edges[j:j+2]
                    dy = y1 - y0
                    eta_span = min(max(float(x[1]) / self.parent.target.structural.Ly, 0.0), 1.0)
                    chord = (self.parent.target.chord * (1.0 - eta_span)
                             + self.parent.target.tip_chord * eta_span)
                    # A normalized beta traction has unit integral and mean
                    # eta_cp, so each panel preserves its resultant and its
                    # chordwise first moment at the production load point.
                    concentration = 8.0
                    alpha = self.parent.eta_cp * concentration
                    beta = (1.0 - self.parent.eta_cp) * concentration
                    x_le = float(self.parent.target.geometry.sweep) * eta_span
                    xi = min(max((float(x[0]) - x_le) / max(chord, 1e-15), 1e-12), 1.0-1e-12)
                    beta_norm = math.gamma(alpha) * math.gamma(beta) / math.gamma(alpha + beta)
                    shape = xi**(alpha-1.0) * (1.0-xi)**(beta-1.0) / beta_norm
                    density = self.parent.panel_forces[j] * shape / max(chord * dy, 1e-15)
                    values[0], values[1], values[2] = density

                def value_shape(self):
                    return (3,)

            traction = PanelPressure(self, degree=0)
        return dot(self.model.displacement_3d(dq), traction) * self.model.dx

    def vector(self, t: float):
        from dolfin import assemble
        return assemble(self.work(t, self.model.q_test))


def add_coupled_mms_source(model, target: CoupledMMSTarget, t: float,
                           reference_force: ReferenceForceFunctional, spaces: list):
    """Assembles the independent manufactured structural source for the coupled FSI solve.

    f_s^* = M(q_s^*) + C(q_dot_s^*) + K(q_s^*) - F_s^*

    INDEPENDENCE GUARANTEE:
    This source is assembled directly from the independent continuous strong-form
    Reissner–Mindlin differential operator integrated against test functions,
    minus the independent analytical interface force reference_force.
    It does NOT call model.mass_form(), model.damping_form(), or model.internal_work().
    """
    from dolfin import Expression, assemble, inner, split, dx
    import re
    global _CACHED_MMS_FORMS

    # Lazy initialization and caching of symbolic load expressions on the model
    if not hasattr(model, "_mms_force_expressions"):
        cfg = {
            "root_chord": target.structural.Lx,
            "span": target.structural.Ly,
            "E": model.E,
            "nu": model.nu,
            "rho_s": model.rho,
            "thickness_ratio": float(model.domain.h) / float(target.structural.Lx),
            "eta_m": model.eta_m,
            "eta_k": model.eta_k,
            "kappa_shear": model.kappa_shear,
        }
        nonlinear = getattr(model, "geometric_nonlinearity", True)
        mms_sym = build_plate_mms_sympy(cfg, nonlinear=nonlinear, components=target.structural.components)

        def _clean_code(expr):
            code = sp.ccode(expr)
            code = re.sub(r"\bx\b", "x[0]", code)
            return re.sub(r"\by\b", "x[1]", code)

        fu_code = tuple(_clean_code(mms_sym.loads[0][i]) for i in range(2))
        fw_code = _clean_code(mms_sym.loads[1])
        fth_code = tuple(_clean_code(mms_sym.loads[2][i]) for i in range(2))

        model._mms_fu = Expression(fu_code, degree=8, t=float(t))
        model._mms_fw = Expression(fw_code, degree=8, t=float(t))
        model._mms_fth = Expression(fth_code, degree=8, t=float(t))

        test_u, test_w, test_theta = split(model.q_test)
        source_measure = dx(domain=model.domain.mesh, metadata={"quadrature_degree": 24})
        model._mms_source_form = (
            inner(model._mms_fu, test_u)
            + model._mms_fw * test_w
            + inner(model._mms_fth, test_theta)
        ) * source_measure
        model._mms_force_expressions = True

    # Update continuous evaluation time
    model._mms_fu.t = float(t)
    model._mms_fw.t = float(t)
    model._mms_fth.t = float(t)

    source = assemble(model._mms_source_form)
    source.axpy(-1.0, reference_force.vector(float(t)))
    for bc in model.bcs:
        bc.apply(source)
    return source


def build_default_target(config: str | Path | dict, T: float, n_span_override: Optional[int] = None) -> CoupledMMSTarget:
    """Helper for backwards compatibility: builds default target from config."""
    if isinstance(config, (str, Path)):
        c_dir = Path(config)
        def _r(name):
            with open(c_dir / name, encoding="utf-8") as fh:
                return yaml.safe_load(fh)
        solid = _r("solid_params.yaml")
        fluid = _r("fluid_params.yaml")
        coupling = _r("coupling_params.yaml")
    else:
        solid, fluid, coupling = config["solid"], config["fluid"], config["coupling"]

    mms_yaml = _REPO_ROOT / "verification/mms/mms_parameters.yaml"
    with open(mms_yaml, encoding="utf-8") as fh:
        mms_params = yaml.safe_load(fh)

    manifest = build_mms_manifest(solid, fluid, coupling, mms_params, refinement_level=1)
    if n_span_override is not None:
        manifest["n_span"] = int(n_span_override)
        manifest["y_stations"] = ((np.arange(manifest["n_span"]) + 0.5) / manifest["n_span"] * float(solid["span"])).tolist()
    manifest["T"] = float(T)
    return CoupledMMSTarget.from_manifest(manifest)
