#!/usr/bin/env python3
"""Solve one Reissner-Mindlin plate MMS case with the production FEniCS model.

Run with the repository's legacy FEniCS environment. The manufactured source
is derived symbolically in plate.py, not from the assembled FEM residual.
"""

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np
import sympy as sp
import yaml

ROOT = Path(__file__).resolve().parents[2]
from plate import make_plate_mms

try:
    from dolfin import (
        Constant, Expression, Function, FunctionAssigner, TestFunction,
        VectorFunctionSpace, assemble, derivative, dx, interpolate, inner,
        parameters, split, dot, solve,
    )
except ImportError as exc:
    raise SystemExit(f"FEniCS plate MMS requires legacy dolfin: {exc}") from exc

sys.path.insert(0, str(ROOT / "solid"))
from utils import ReissnerMindlinModel, SolidDomain


def expression_code(expr):
    code = sp.ccode(expr)
    code = re.sub(r"\bx\b", "x[0]", code)
    return re.sub(r"\by\b", "x[1]", code)


def scalar_expr(expr, time=0.0):
    return Expression(expression_code(expr), degree=8, t=float(time))


def vector_expr(expr, time=0.0):
    return Expression(tuple(expression_code(expr[i]) for i in range(len(expr))), degree=8, t=float(time))


def tensor_vector_expr(expr, time):
    values = tuple(expression_code(expr[i, j]) for i in range(2) for j in range(2))
    return Expression(values, degree=8, t=float(time))


def make_mixed_function(model, exact_fields, time, spaces, assigner):
    u, w, theta = exact_fields
    pieces = [
        interpolate(vector_expr(u, time), spaces[0]),
        interpolate(scalar_expr(w, time), spaces[1]),
        interpolate(vector_expr(theta, time), spaces[2]),
    ]
    result = Function(model.V)
    assigner.assign(result, pieces)
    return result


def run_case(config_dir, nx, ny, dt, duration, nonlinear, out_csv):
    with open(Path(config_dir) / "solid_params.yaml") as stream:
        cfg = yaml.safe_load(stream)
    mms = make_plate_mms(cfg, nonlinear, cfg.get("include_quadratic_inplane_strain", True))
    domain = SolidDomain(
        cfg["span"], cfg["root_chord"], cfg["tip_chord"], cfg["thickness_ratio"],
        cfg.get("leading_edge_sweep", 0.0), nx, ny, min(ny, 40), "midpoint",
    )
    model = ReissnerMindlinModel(
        domain, cfg["E"], cfg["nu"], cfg["rho_s"], cfg["eta_m"], cfg["eta_k"],
        cfg["kappa_shear"], Constant(dt), cfg.get("alpha_m", 0.1), cfg.get("alpha_f", 0.2),
        geometric_nonlinearity=nonlinear,
        include_quadratic_inplane_strain=cfg.get("include_quadratic_inplane_strain", True),
    )
    spaces = [model.V.sub(i).collapse() for i in range(3)]
    assigner = FunctionAssigner(model.V, spaces)
    exact_u, exact_w, exact_theta = mms.fields
    exact_fields = (exact_u, exact_w, exact_theta)
    f_u, f_w, f_theta = mms.loads
    q_test = split(model.q_test)
    source = (
        dot(vector_expr(f_u), q_test[0])
        + scalar_expr(f_w) * q_test[1]
        + dot(vector_expr(f_theta), q_test[2])
    ) * dx
    model.jacobian = derivative(model.residual, model.q, model.dq_trial)

    model.q_old.assign(make_mixed_function(model, exact_fields, 0, spaces, assigner))
    model.q.assign(model.q_old)
    model.v_old.assign(make_mixed_function(model, tuple(v.diff(mms.t) for v in exact_fields), 0, spaces, assigner))
    model.a_old.assign(make_mixed_function(model, tuple(v.diff(mms.t, 2) for v in exact_fields), 0, spaces, assigner))

    nsteps = int(round(duration / dt))
    for step in range(1, nsteps + 1):
        # The load is evaluated at t_(n+1-alpha_f), matching the residual's
        # internal-force time level. Acceleration/velocity use the integrator's
        # actual discrete update from q_n, v_n, and a_n.
        t_force = step * dt - float(cfg.get("alpha_f", 0.2)) * dt
        for expr in source.coefficients():
            if hasattr(expr, "t"):
                expr.t = t_force
        force_vector = assemble(source)
        converged = False
        initial_norm = None
        for iteration in range(30):
            A = assemble(model.jacobian)
            rhs = assemble(model.residual)
            rhs *= -1.0
            rhs.axpy(1.0, force_vector)
            for bc in model.bcs:
                bc.apply(A, rhs)
            norm = rhs.norm("l2")
            initial_norm = norm if initial_norm is None else initial_norm
            if norm < 1.0e-8 or norm / max(initial_norm, 1e-30) < 1.0e-8:
                converged = True
                break
            solve(A, model.dq_newton.vector(), rhs, "mumps")
            model.q.vector().axpy(1.0, model.dq_newton.vector())
            model.q.vector().apply("insert")
        if not converged:
            raise RuntimeError(f"MMS Newton solve failed at step {step}: residual={norm:.4e}")
        model.update_fields()

    q_num = split(model.q_old)
    final_time = nsteps * dt
    u_ref = interpolate(vector_expr(exact_u, final_time), spaces[0])
    w_ref = interpolate(scalar_expr(exact_w, final_time), spaces[1])
    theta_ref = interpolate(vector_expr(exact_theta, final_time), spaces[2])
    q_ref = (u_ref, w_ref, theta_ref)
    errors, refs = [], []
    for numerical, reference in zip(q_num, q_ref):
        errors.append(float(np.sqrt(assemble(inner(numerical-reference, numerical-reference) * dx))))
        refs.append(float(np.sqrt(assemble(inner(reference, reference) * dx))))
    names = ("inplane_u_l2", "transverse_w_l2", "rotation_theta_l2")
    row = {"formulation": "nonlinear" if nonlinear else "linear", "nx": nx, "ny": ny,
           "dt_s": dt, "duration_s": final_time, "steps": nsteps,
           "newton_iterations_last_step": iteration + 1, "newton_residual_last_step": norm}
    for name, error, ref in zip(names, errors, refs):
        row[name] = error
        row[name.replace("_l2", "_relative_l2")] = error / ref if ref > 1e-15 else error
    for comp, label in enumerate(("u_x", "u_y")):
        err = float(np.sqrt(assemble((q_num[0][comp] - q_ref[0][comp])**2 * dx)))
        ref = float(np.sqrt(assemble(q_ref[0][comp]**2 * dx)))
        row[f"{label}_relative_l2"] = err / ref if ref > 1e-15 else err
    for comp, label in enumerate(("theta_x", "theta_y")):
        err = float(np.sqrt(assemble((q_num[2][comp] - q_ref[2][comp])**2 * dx)))
        ref = float(np.sqrt(assemble(q_ref[2][comp]**2 * dx)))
        row[f"{label}_relative_l2"] = err / ref if ref > 1e-15 else err
    exact_velocity = tuple(v.diff(mms.t) for v in exact_fields)
    exact_acceleration = tuple(v.diff(mms.t, 2) for v in exact_fields)
    v_ref = make_mixed_function(model, exact_velocity, final_time, spaces, assigner)
    a_ref = make_mixed_function(model, exact_acceleration, final_time, spaces, assigner)
    for name, numerical, reference in (("velocity", split(model.v_old), split(v_ref)),
                                       ("acceleration", split(model.a_old), split(a_ref))):
        error_sq = sum(assemble(inner(numerical[i]-reference[i], numerical[i]-reference[i]) * dx) for i in range(3))
        ref_sq = sum(assemble(inner(reference[i], reference[i]) * dx) for i in range(3))
        row[f"{name}_relative_l2"] = float(np.sqrt(error_sq / ref_sq)) if ref_sq > 1e-30 else float(np.sqrt(error_sq))

    from dolfin import as_vector
    V4 = VectorFunctionSpace(domain.mesh, "DG", 0, dim=4)
    V2 = VectorFunctionSpace(domain.mesh, "DG", 0, dim=2)
    u_num, w_num, theta_num = q_num
    eps_num = model.nonlinear_membrane_strain(u_num, w_num) if nonlinear else model.membrane_strain(u_num)
    kap_num = model.curvature(theta_num)
    shear_num = model.shear_strain(theta_num, w_num)
    N_num = model.nonlinear_membrane_stress(u_num, w_num) if nonlinear else model.membrane_stress(u_num)
    M_num = model.bending_moment(theta_num)
    Q_num = model.kappa_shear * (model.E / (2.0 * (1.0 + model.nu))) * model.h * shear_num
    num_kin = {
        "membrane_strain": as_vector((eps_num[0, 0], eps_num[0, 1], eps_num[1, 0], eps_num[1, 1])),
        "curvature": as_vector((kap_num[0, 0], kap_num[0, 1], kap_num[1, 0], kap_num[1, 1])),
        "shear_strain": shear_num,
        "membrane_resultant": as_vector((N_num[0, 0], N_num[0, 1], N_num[1, 0], N_num[1, 1])),
        "bending_moment": as_vector((M_num[0, 0], M_num[0, 1], M_num[1, 0], M_num[1, 1])),
        "shear_resultant": Q_num,
    }
    for name, num_expr in num_kin.items():
        ref_expr = mms.kinematics[name]
        if isinstance(ref_expr, sp.MatrixBase) and ref_expr.shape == (2, 2):
            reference = interpolate(tensor_vector_expr(ref_expr, final_time), V4)
            numeric = interpolate(num_expr, V4)
        else:
            reference = interpolate(vector_expr(ref_expr, final_time), V2)
            numeric = interpolate(num_expr, V2)
        err = float(np.sqrt(assemble(inner(numeric-reference, numeric-reference) * dx)))
        ref = float(np.sqrt(assemble(inner(reference, reference) * dx)))
        row[f"{name}_relative_l2"] = err / ref if ref > 1e-30 else err

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    mesh_xy = domain.mesh.coordinates()
    triangulation = mtri.Triangulation(mesh_xy[:, 0], mesh_xy[:, 1], domain.mesh.cells())
    w_plot_num = model.q_old.split(deepcopy=True)[1]
    numerical_values = w_plot_num.compute_vertex_values(domain.mesh)
    exact_values = w_ref.compute_vertex_values(domain.mesh)
    error_values = numerical_values - exact_values
    field_dir = Path(out_csv).parent
    field_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), constrained_layout=True)
    limit = max(float(np.max(np.abs(exact_values))), float(np.max(np.abs(numerical_values))), 1e-16) * 1e3
    for ax, values, title in zip(axes[:2], (exact_values, numerical_values), ("Exact w", "FEniCS w")):
        contour = ax.tricontourf(triangulation, values * 1e3, levels=24, cmap="viridis", vmin=-limit, vmax=limit)
        ax.set_title(title)
        ax.set_xlabel("Chordwise x [m]")
        ax.set_ylabel("Spanwise y [m]")
        fig.colorbar(contour, ax=ax, label="w [mm]")
    err_limit = max(float(np.max(np.abs(error_values))) * 1e3, 1e-14)
    contour = axes[2].tricontourf(triangulation, error_values * 1e3, levels=24, cmap="coolwarm", vmin=-err_limit, vmax=err_limit)
    axes[2].set_title("FEniCS - exact")
    axes[2].set_xlabel("Chordwise x [m]")
    axes[2].set_ylabel("Spanwise y [m]")
    fig.colorbar(contour, ax=axes[2], label="w error [mm]")
    fig.savefig(field_dir / f"plate_{row['formulation']}_{nx}x{ny}_field_error.png", dpi=170)
    plt.close(fig)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    is_new = not out_csv.exists()
    with open(out_csv, "a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if is_new:
            writer.writeheader()
        writer.writerow(row)
    print(row)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", default=str(ROOT / "config"))
    parser.add_argument("--nx", type=int, default=12)
    parser.add_argument("--ny", type=int, default=32)
    parser.add_argument("--dt", type=float, default=0.0025)
    parser.add_argument("--duration", type=float, default=0.05)
    parser.add_argument("--nonlinear", action="store_true")
    parser.add_argument("--output", default=str(Path(__file__).parent / "results" / "plate_fenics.csv"))
    args = parser.parse_args()
    parameters["form_compiler"]["optimize"] = True
    run_case(args.config_dir, args.nx, args.ny, args.dt, args.duration, args.nonlinear, args.output)


if __name__ == "__main__":
    main()
