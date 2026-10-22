"""Symbolic Reissner-Mindlin manufactured fields and strong-form loads.

The equations below are independently assembled from the constitutive laws in
solid/utils/reisner_mindlin_model.py; they do not call the FEniCS residual.

The smooth spatial envelope is B(x,y)=sin(pi*x/Lx)^4*sin(pi*y/Ly)^4.
Fields are u_x=1e-4 B cos(2.3t), u_y=1e-4 B sin(2.3t),
w=1e-3 B sin(2.3t), theta_x=1e-3 B cos(2.3t), and
theta_y=-1e-3 B sin(2.3t), in SI units. Symbolic body loads are obtained by
substituting these fields into the plate's strong equilibrium equations.
"""

from dataclasses import dataclass
from contextlib import nullcontext

import sympy as sp


@dataclass
class PlateMMS:
    x: sp.Symbol
    y: sp.Symbol
    t: sp.Symbol
    fields: tuple
    loads: tuple
    load_components: dict
    kinematics: dict
    boundary_checks: dict


def _div_rows(tensor, x, y):
    return sp.Matrix([sp.diff(tensor[i, 0], x) + sp.diff(tensor[i, 1], y) for i in range(2)])


def make_plate_mms(config, nonlinear, include_quadratic_inplane=True, operation_factory=None):
    """Build analytic fields and strong loads, optionally timing symbolic stages."""
    timed = operation_factory or (lambda _stage, **_kwargs: nullcontext())
    x, y, t = sp.symbols("x y t", real=True)
    Lx = sp.Float(config["root_chord"])
    Ly = sp.Float(config["span"])
    E = sp.Float(config["E"])
    nu = sp.Float(config["nu"])
    rho = sp.Float(config["rho_s"])
    h = Lx * sp.Float(config["thickness_ratio"])
    eta_m = sp.Float(config["eta_m"])
    eta_k = sp.Float(config["eta_k"])
    kappa = sp.Float(config["kappa_shear"])
    omega = sp.Float("2.3")

    # Fourth-order zeros make the exact field and required derivatives vanish
    # at the clamped y=0 root and satisfy the natural free-edge conditions.
    with timed("sympy_manufactured_fields"):
        bump = sp.sin(sp.pi * x / Lx) ** 4 * sp.sin(sp.pi * y / Ly) ** 4
        u = sp.Matrix([
            sp.Float("1e-4") * bump * sp.cos(omega * t),
            sp.Float("1e-4") * bump * sp.sin(omega * t),
        ])
        w = sp.Float("1e-3") * bump * sp.sin(omega * t)
        theta = sp.Matrix([
            sp.Float("1e-3") * bump * sp.cos(omega * t),
            -sp.Float("1e-3") * bump * sp.sin(omega * t),
        ])

    with timed("sympy_membrane_kinematics_and_stress"):
        grad_u = u.jacobian((x, y))
        grad_w = sp.Matrix([sp.diff(w, x), sp.diff(w, y)])
        eps = (grad_u + grad_u.T) / 2
        if nonlinear:
            eps += grad_w * grad_w.T / 2
            if include_quadratic_inplane:
                eps += grad_u.T * grad_u / 2

        Cmem = E * h / (1 - nu**2)
        N = Cmem * ((1 - nu) * eps + nu * sp.trace(eps) * sp.eye(2))
        D = E * h**3 / (12 * (1 - nu**2))
        grad_theta = theta.jacobian((x, y))
        curvature = (grad_theta + grad_theta.T) / 2
        M = D * ((1 - nu) * curvature + nu * sp.trace(curvature) * sp.eye(2))
        G = E / (2 * (1 + nu))
        shear = grad_w - theta
        Q = kappa * G * h * shear

    with timed("sympy_strong_internal_divergence"):
        if nonlinear:
            Pmem = N + grad_u * N
            P_w = N * grad_w
        else:
            Pmem = N
            P_w = sp.zeros(2, 1)
        internal_u = -_div_rows(Pmem, x, y)
        internal_w = -sp.diff(P_w[0] + Q[0], x) - sp.diff(P_w[1] + Q[1], y)
        internal_theta = -_div_rows(M, x, y) - Q

    # The implemented eta_m and eta_k are mass- and initial-stiffness
    # proportional damping. Apply the linear reference stiffness to q_t.
    with timed("sympy_reference_stiffness_damping"):
        udot, wdot, tdot = u.diff(t), sp.diff(w, t), theta.diff(t)
        grad_udot = udot.jacobian((x, y))
        eps_dot = (grad_udot + grad_udot.T) / 2
        N_dot = Cmem * ((1 - nu) * eps_dot + nu * sp.trace(eps_dot) * sp.eye(2))
        grad_wdot = sp.Matrix([sp.diff(wdot, x), sp.diff(wdot, y)])
        Q_dot = kappa * G * h * (grad_wdot - tdot)
        grad_tdot = tdot.jacobian((x, y))
        kap_dot = (grad_tdot + grad_tdot.T) / 2
        M_dot = D * ((1 - nu) * kap_dot + nu * sp.trace(kap_dot) * sp.eye(2))
        damp_u_stiff = -_div_rows(N_dot, x, y)
        damp_w_stiff = -sp.diff(Q_dot[0], x) - sp.diff(Q_dot[1], y)
        damp_theta_stiff = -_div_rows(M_dot, x, y) - Q_dot

    with timed("sympy_manufactured_load_components"):
        mass = rho * h
        rotary_mass = rho * h**3 / 12
        zero_vec = sp.zeros(2, 1)
        div_pw = sp.diff(P_w[0], x) + sp.diff(P_w[1], y)
        div_q = sp.diff(Q[0], x) + sp.diff(Q[1], y)
        div_m = _div_rows(M, x, y)
        load_components = {
            "inertia": (mass * u.diff(t, 2), mass * sp.diff(w, t, 2),
                        rotary_mass * theta.diff(t, 2)),
            "mass_damping": (eta_m * mass * udot, eta_m * mass * wdot,
                             eta_m * rotary_mass * tdot),
            "reference_stiffness_damping": (
                eta_k * damp_u_stiff, eta_k * damp_w_stiff, eta_k * damp_theta_stiff
            ),
            "membrane": (internal_u, -div_pw, zero_vec),
            "bending": (zero_vec, sp.Integer(0), -div_m),
            "shear": (zero_vec, -div_q, -Q),
        }
        with timed("sympy_factor_manufactured_load_components"):
            load_components = {
                name: tuple(
                    sp.Matrix(component).applyfunc(sp.factor_terms)
                    if isinstance(component, sp.MatrixBase)
                    else sp.factor_terms(component)
                    for component in components
                )
                for name, components in load_components.items()
            }
        f_u = sum((part[0] for part in load_components.values()), zero_vec)
        f_w = sum((part[1] for part in load_components.values()), sp.Integer(0))
        f_theta = sum((part[2] for part in load_components.values()), zero_vec)
        f_u = f_u.applyfunc(sp.factor_terms)
        f_w = sp.factor_terms(f_w)
        f_theta = f_theta.applyfunc(sp.factor_terms)

    # Boundary traction checks for root/clamped and free edges.
    with timed("sympy_boundary_substitution_and_simplification"):
        checks = {
        "u_at_root": [sp.simplify(v.subs(y, 0)) for v in u],
        "w_at_root": sp.simplify(w.subs(y, 0)),
        "theta_at_root": [sp.simplify(v.subs(y, 0)) for v in theta],
        "membrane_x_edges": [sp.simplify(Pmem[i, 0].subs(x, e)) for e in (0, Lx) for i in range(2)],
        "membrane_y_edges": [sp.simplify(Pmem[i, 1].subs(y, e)) for e in (0, Ly) for i in range(2)],
        "moment_x_edges": [sp.simplify(M[i, 0].subs(x, e)) for e in (0, Lx) for i in range(2)],
        "moment_y_edges": [sp.simplify(M[i, 1].subs(y, e)) for e in (0, Ly) for i in range(2)],
        "shear_x_edges": [sp.simplify((P_w[i] + Q[i]).subs(x, e)) for e in (0, Lx) for i in range(2)],
        "shear_y_edges": [sp.simplify((P_w[i] + Q[i]).subs(y, e)) for e in (0, Ly) for i in range(2)],
        }
    return PlateMMS(
        x, y, t, (u, w, theta), (f_u, f_w, f_theta), load_components,
        {"membrane_strain": eps, "curvature": curvature, "shear_strain": shear,
         "membrane_resultant": N, "bending_moment": M, "shear_resultant": Q},
        checks,
    )
