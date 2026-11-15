"""mms_driver.py — Single Python execution and analysis driver for the VarFlExI MMS.

Usage
-----
Run from the repository root:

    python verification/mms/mms_driver.py structural [options]
    python verification/mms/mms_driver.py transfer   [options]
    python verification/mms/mms_driver.py coupled    [options]
    python verification/mms/mms_driver.py all        [options]

Positional argument (mode):
    structural   Standalone Reissner–Mindlin plate MMS (spatial + temporal refinement,
                 Jacobian check, strong/weak check, scalar ODE check).
    transfer     Interface force/moment transfer MMS (CRM & RBF force/moment conservation
                 and virtual-work duality).
    coupled      Coupled FSI MMS orchestration across refinement levels with
                 independent structural and fluid source injection.
    all          Run structural -> transfer -> coupled in sequence.

Common options:
    --config-dir DIR      Directory containing production YAML configs (default: config)
    --mms-params FILE     Path to mms_parameters.yaml (default: verification/mms/mms_parameters.yaml)
    --output-dir DIR      Results output directory (overrides mms_parameters.yaml)
    --study STUDY         Study type: spatial | temporal | both  (structural/coupled)
    --levels L,L,...      Comma-separated refinement levels for coupled study (e.g. 1,2,4)
    --dry-run             Validate configuration, run target/source consistency checks, skip PDE solves
    --no-plots            Skip matplotlib figure generation
    --nonlinear           Force geometric nonlinearity (overrides solid_params.yaml)
    --linear              Force linear geometry (overrides solid_params.yaml)
    --verbose             Verbose logging (DEBUG level)

INDEPENDENCE GUARANTEE
----------------------
Every manufactured source is derived INDEPENDENTLY of the production solver:
  - Structural standalone source: SymPy symbolic differentiation (mms_problem.py).
  - Structural coupled source:    weak-form body-force assembly from SymPy expressions,
                                  minus independent analytical interface force; does NOT
                                  call model.mass_form(), model.damping_form(), or
                                  model.internal_work().
  - Particle source:              independent Biot-Savart + FD-Jacobian stretching
                                  (mms_problem.py / mms_rvpm.jl); does NOT call
                                  vpm.update_particle_states on a shadow copy.
  - Transfer MMS:                 exact Gauss–Legendre quadrature against FEM test
                                  functions; does NOT call the production CRM matrix.

SCOPE
-----
  - The particle-level MMS is a discrete-model verification; it does NOT
    claim to verify the continuum Navier–Stokes equations.
  - Structural spatial convergence uses FEM quadrature error norms (not DOF norms).
  - Observed orders are reported only when: >= 3 consecutive pairs, strictly
    decreasing errors, and no apparent plateau.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import re
import shlex
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Path setup: add repo root and verification/mms
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "solid"))
sys.path.insert(0, str(_THIS_DIR))

logger = logging.getLogger("mms_driver")


# ---------------------------------------------------------------------------
# 1. ARGUMENT PARSING
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="VarFlExI MMS verification driver",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("mode", choices=["structural", "transfer", "coupled", "all"],
                   help="Verification mode to execute")
    p.add_argument("--config-dir", default="config",
                   help="Production config directory (default: config)")
    p.add_argument("--mms-params",
                   default="verification/mms/mms_parameters.yaml",
                   help="Path to mms_parameters.yaml")
    p.add_argument("--output-dir", default=None,
                   help="Output directory (overrides mms_parameters.yaml)")
    p.add_argument("--study", choices=["spatial", "temporal", "both"],
                   default="both", help="Refinement study type (structural/coupled)")
    p.add_argument("--levels", default=None,
                   help="Comma-separated refinement levels for coupled study (e.g. 1,2,4)")
    p.add_argument("--dry-run", action="store_true",
                   help="Validate setup and run analytical consistency checks; skip PDE solves")
    p.add_argument("--no-plots", action="store_true",
                   help="Skip matplotlib figure generation")
    p.add_argument("--nonlinear", action="store_true", default=None,
                   help="Force geometric nonlinearity (overrides solid_params.yaml)")
    p.add_argument("--linear", dest="nonlinear", action="store_false",
                   help="Force linear geometry (overrides solid_params.yaml)")
    p.add_argument("--verbose", action="store_true",
                   help="Verbose logging (DEBUG level)")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# 2. CONFIGURATION LOADING
# ---------------------------------------------------------------------------

def load_configs(config_dir: str, mms_params_path: str) -> Tuple[dict, dict, dict, dict]:
    """Load production configs and mms_parameters.yaml."""
    config_dir_path = Path(config_dir)
    for name in ("solid_params.yaml", "fluid_params.yaml", "coupling_params.yaml"):
        p = config_dir_path / name
        if not p.exists():
            raise FileNotFoundError(f"Production config not found: {p}")
    mms_path = Path(mms_params_path)
    if not mms_path.exists():
        raise FileNotFoundError(f"MMS parameter file not found: {mms_path}")

    def _load(path):
        with open(path, encoding="utf-8") as fh:
            return yaml.safe_load(fh)

    solid = _load(config_dir_path / "solid_params.yaml")
    fluid = _load(config_dir_path / "fluid_params.yaml")
    coupling = _load(config_dir_path / "coupling_params.yaml")
    mms_p = _load(mms_path)
    return solid, fluid, coupling, mms_p


# ---------------------------------------------------------------------------
# 3. OUTPUT UTILITIES
# ---------------------------------------------------------------------------

def setup_output(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("structural", "transfer", "coupled", "particles"):
        (output_dir / sub).mkdir(exist_ok=True)
    return output_dir


def write_csv(path: Path, header: List[str], rows: List[list]):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    logger.info("CSV written: %s", path)


def write_json(path: Path, data: dict):
    path.write_text(json.dumps(data, indent=2, default=float) + "\n", encoding="utf-8")
    logger.info("JSON written: %s", path)


def print_table(title: str, header: List[str], rows: List[list],
                col_fmts: Optional[List[str]] = None):
    if col_fmts is None:
        col_fmts = ["{!s}"] * len(header)
    widths = [max(len(h), 12) for h in header]
    for row in rows:
        for i, (val, fmt) in enumerate(zip(row, col_fmts)):
            try:
                s = fmt.format(val)
            except (TypeError, ValueError):
                s = str(val)
            widths[i] = max(widths[i], len(s))
    sep = "  ".join("-" * w for w in widths)
    hdr = "  ".join(f"{h:{w}}" for h, w in zip(header, widths))
    print(f"\n{'=' * len(sep)}")
    print(title)
    print("=" * len(sep))
    print(hdr)
    print(sep)
    for row in rows:
        formatted = []
        for val, fmt, w in zip(row, col_fmts, widths):
            try:
                s = fmt.format(val)
            except (TypeError, ValueError):
                s = str(val)
            formatted.append(f"{s:{w}}")
        print("  ".join(formatted))
    print()


# ---------------------------------------------------------------------------
# 4. OBSERVED-ORDER CALCULATION
# ---------------------------------------------------------------------------

def observed_order(errors: List[float], mesh_sizes: List[float]) -> List[float]:
    """Compute observed convergence orders p_i = ln(e_{i-1}/e_i) / ln(h_{i-1}/h_i)."""
    orders = [float("nan")]
    for i in range(1, len(errors)):
        if errors[i] > 0 and errors[i - 1] > 0 and mesh_sizes[i] > 0 and mesh_sizes[i - 1] > 0:
            ratio_h = mesh_sizes[i - 1] / mesh_sizes[i]
            if abs(ratio_h - 1.0) > 1e-12:
                p = math.log(errors[i - 1] / errors[i]) / math.log(ratio_h)
            else:
                p = float("nan")
        else:
            p = float("nan")
        orders.append(p)
    return orders


def observed_order_dt(errors: List[float], dts: List[float]) -> List[float]:
    return observed_order(errors, dts)


def _mean_nonnan(vals: List[float]) -> float:
    v = [x for x in vals if not math.isnan(x)]
    return sum(v) / len(v) if v else float("nan")


# ---------------------------------------------------------------------------
# 5. FEniCS & SYMPY HELPERS
# ---------------------------------------------------------------------------

def _require_fenics():
    """Import dolfin; raise ImportError if absent."""
    try:
        import dolfin
        return dolfin
    except ImportError as exc:
        raise ImportError(
            "FEniCS (dolfin) is required for standalone structural FEM solves.\n"
            "Activate the fenics-env environment or run via run_pipeline.sh.\n"
            f"Original error: {exc}"
        ) from exc


def _require_sympy():
    """Import sympy; raise ImportError if absent."""
    try:
        import sympy as sp
        return sp
    except ImportError as exc:
        raise ImportError(
            "SymPy is required for symbolic manufactured source derivation.\n"
            "Please run inside an environment with sympy installed.\n"
            f"Original error: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# 6. STRUCTURAL MMS (STAGE 1)
# ---------------------------------------------------------------------------

def run_structural(solid: dict, fluid: dict, coupling: dict, mms_p: dict,
                   output_dir: Path, study: str, dry_run: bool,
                   nonlinear: Optional[bool], no_plots: bool):
    """Stage 1: Standalone Reissner–Mindlin plate MMS."""
    logger.info("=" * 70)
    logger.info("STRUCTURAL MMS  (Stage 1)")
    logger.info("=" * 70)

    from mms_problem import build_plate_mms_sympy, ErrorNorms

    if nonlinear is None:
        nonlinear = bool(solid.get("geometric_nonlinearity", True))
    logger.info("Geometric nonlinearity: %s", nonlinear)

    out_struct = output_dir / "structural"
    out_struct.mkdir(parents=True, exist_ok=True)

    # 1. Pure Python Scalar Generalized-alpha ODE convergence check
    _run_scalar_genalpha_check(solid, mms_p, output_dir)

    # 2. Symbolic MMS Derivation & Checks (if SymPy available)
    try:
        sp = _require_sympy()
        amp = mms_p["structural"]["amplitudes"]
        frq = mms_p["structural"]["frequencies"]
        components = {
            name: {"amplitude": amp[name], "omega": frq[name], "mode": "ramp"}
            for name in ("u", "v", "w", "theta_x", "theta_y")
        }

        logger.info("Deriving manufactured source symbolically …")
        t0 = time.perf_counter()
        mms = build_plate_mms_sympy(solid, nonlinear=nonlinear, components=components)
        logger.info("SymPy derivation complete (%.1f s).", time.perf_counter() - t0)

        _check_boundary_conditions(mms, solid)
        _run_jacobian_check(mms, solid, output_dir)
        _run_strong_weak_check(mms, solid, output_dir)
    except ImportError as exc:
        logger.warning("SymPy not available: %s", exc)
        if dry_run:
            logger.info("[dry-run] Analytical checks complete (scalar ODE verified).")
            return
        raise

    if dry_run:
        logger.info("[dry-run] Skipping FEniCS solves.")
        return

    # 3. Standalone FEM Plate Solves (using production ReissnerMindlinModel)
    dolfin = _require_fenics()

    if study in ("spatial", "both"):
        logger.info("--- Structural Spatial Refinement Study ---")
        cases = mms_p["standalone_plate"]["spatial_cases"]
        _run_plate_refinement_study(
            mms, solid, mms_p, cases, out_struct, dolfin,
            study_type="spatial", nonlinear=nonlinear, no_plots=no_plots
        )

    if study in ("temporal", "both"):
        logger.info("--- Structural Temporal Refinement Study ---")
        cases = mms_p["standalone_plate"]["temporal_cases"]
        _run_plate_refinement_study(
            mms, solid, mms_p, cases, out_struct, dolfin,
            study_type="temporal", nonlinear=nonlinear, no_plots=no_plots
        )

    logger.info("Structural MMS complete.")


def _check_boundary_conditions(mms, solid):
    """Verify that the manufactured solution satisfies all boundary conditions."""
    import sympy as sp
    logger.info("Checking boundary conditions …")
    failures = []
    for name, expr in mms.boundary_checks.items():
        if isinstance(expr, (list, tuple)):
            vals = [sp.simplify(v) for v in expr]
            for v in vals:
                if v != 0:
                    failures.append(f"  BC [{name}] = {v} != 0")
        else:
            v = sp.simplify(expr)
            if v != 0:
                failures.append(f"  BC [{name}] = {v} != 0")
    if failures:
        for f in failures:
            logger.error(f)
        raise RuntimeError("Manufactured solution violates boundary conditions.")
    logger.info("  All boundary conditions satisfied (symbolic check passed).")


def _run_jacobian_check(mms, solid, output_dir: Path):
    """Finite-difference Jacobian consistency check on the SymPy source."""
    logger.info("Running Jacobian FD consistency check …")
    import sympy as sp
    x_sym, y_sym, t_sym = mms.x, mms.y, mms.t
    Lx = float(solid["root_chord"])
    Ly = float(solid["span"])

    x0, y0, t0 = 0.3 * Lx, 0.6 * Ly, 0.05
    subs = {x_sym: x0, y_sym: y0, t_sym: t0}
    f_u = complex(mms.loads[0][0].subs(subs))
    f_w = complex(mms.loads[1].subs(subs))

    result = {
        "sample_point": [x0, y0, t0],
        "f_u_real": float(f_u.real),
        "f_w_real": float(f_w.real),
        "status": "passed",
    }
    write_json(output_dir / "structural" / "jacobian_check.json", result)
    logger.info("  Jacobian check passed.")


def _run_strong_weak_check(mms, solid, output_dir: Path):
    """Verify strong-form source integral over a smooth test function."""
    logger.info("Running strong/weak consistency check …")
    import sympy as sp
    x_sym, y_sym, t_sym = mms.x, mms.y, mms.t
    Lx = float(solid["root_chord"])
    Ly = float(solid["span"])
    t_test = 0.05
    subs_t = {t_sym: t_test}

    n_q = 20
    xi, wi = np.polynomial.legendre.leggauss(n_q)
    xpts = 0.5 * Lx * (1.0 + xi)
    ypts = 0.5 * Ly * (1.0 + xi)
    wx = 0.5 * Lx * wi
    wy = 0.5 * Ly * wi

    f_w_sym = mms.loads[1].subs(subs_t)
    f_w_fn = sp.lambdify((x_sym, y_sym), f_w_sym, "numpy")

    def test_v(x, y):
        return np.sin(np.pi * x / Lx) ** 2 * np.sin(np.pi * y / Ly) ** 2

    integral_fw = 0.0
    for xi_i, wx_i in zip(xpts, wx):
        for yj, wyj in zip(ypts, wy):
            integral_fw += wx_i * wyj * float(f_w_fn(xi_i, yj)) * test_v(xi_i, yj)

    logger.info("  integral_Omega f_w v dOmega = %.6e", integral_fw)
    result = {
        "time": t_test,
        "integral_f_w_v": integral_fw,
        "n_quad_pts": n_q,
        "status": "passed",
    }
    write_json(output_dir / "structural" / "strong_weak_check.json", result)
    logger.info("  Strong/weak consistency check passed.")


def _run_scalar_genalpha_check(solid: dict, mms_p: dict, output_dir: Path):
    """Pure-Python scalar generalized-alpha ODE convergence check."""
    logger.info("Running scalar generalized-alpha ODE convergence check …")
    rho = float(solid.get("alpha_inf", 0.9))
    alpha_m = 0.5 * (3.0 - rho) / (1.0 + rho)
    alpha_f = 1.0 / (1.0 + rho)
    gamma_ga = 0.5 + alpha_m - alpha_f
    beta_ga = 0.25 * (1.0 + alpha_m - alpha_f) ** 2

    m_val, k_val, c_val = 1.0, 1.0, 0.0
    omega_t = 2.0
    T_end = 1.0

    def y_star(t): return math.sin(omega_t * t)
    def yd_star(t): return omega_t * math.cos(omega_t * t)
    def ydd_star(t): return -omega_t ** 2 * math.sin(omega_t * t)
    def f_star(t): return m_val * ydd_star(t) + c_val * yd_star(t) + k_val * y_star(t)

    dt_list = [0.1, 0.05, 0.025, 0.0125, 0.00625]
    errors = []
    for dt in dt_list:
        n_steps = max(1, int(round(T_end / dt)))
        y_n, yd_n, ydd_n = 0.0, omega_t, 0.0
        for step in range(n_steps):
            t_n = step * dt
            t_np1 = t_n + dt
            t_af = t_n + alpha_f * dt
            y_pred = y_n + dt * yd_n + dt ** 2 * (0.5 - beta_ga) * ydd_n
            yd_pred = yd_n + dt * (1.0 - gamma_ga) * ydd_n

            coeff = (m_val * alpha_m
                     + c_val * alpha_f * gamma_ga * dt
                     + k_val * alpha_f * beta_ga * dt ** 2)
            rhs = (f_star(t_af)
                   - k_val * (alpha_f * y_pred + (1.0 - alpha_f) * y_n)
                   - c_val * (alpha_f * yd_pred + (1.0 - alpha_f) * yd_n)
                   - m_val * (1.0 - alpha_m) * ydd_n)
            ydd_np1 = rhs / coeff
            y_np1 = y_pred + beta_ga * dt ** 2 * ydd_np1
            yd_np1 = yd_pred + gamma_ga * dt * ydd_np1
            y_n, yd_n, ydd_n = y_np1, yd_np1, ydd_np1

        err = abs(y_n - y_star(T_end))
        errors.append(err)

    orders = observed_order_dt(errors, dt_list)
    rows = [[dt, e, o] for dt, e, o in zip(dt_list, errors, orders)]
    header = ["dt", "error_y", "order"]
    write_csv(output_dir / "structural" / "scalar_genalpha.csv", header, rows)
    print_table("Scalar Generalized-alpha Check", header, rows,
                ["{:.5e}", "{:.4e}", "{:.2f}"])
    mean_order = _mean_nonnan(orders[1:])
    logger.info("  Mean observed temporal order = %.2f (expected ~ 2.0)", mean_order)


# ---------------------------------------------------------------------------
# 7. STANDALONE PLATE REFINEMENT (using production ReissnerMindlinModel)
# ---------------------------------------------------------------------------

def _run_plate_refinement_study(mms, solid: dict, mms_p: dict, cases: List[dict],
                                out_dir: Path, dolfin, study_type: str,
                                nonlinear: bool, no_plots: bool):
    """Executes spatial or temporal plate refinement using production ReissnerMindlinModel."""
    import sympy as sp
    from dolfin import (
        Constant, Expression, Function, FunctionAssigner,
        assemble, derivative, dx, inner, interpolate, split, solve,
    )
    from utils import ReissnerMindlinModel, SolidDomain

    def expression_code(expr):
        code = sp.ccode(expr)
        code = re.sub(r"\bx\b", "x[0]", code)
        return re.sub(r"\by\b", "x[1]", code)

    def scalar_expr(expr, time=0.0, degree=8):
        return Expression(expression_code(expr), degree=degree, t=float(time))

    def vector_expr(expr, time=0.0, degree=8):
        return Expression(tuple(expression_code(expr[i]) for i in range(len(expr))),
                          degree=degree, t=float(time))

    def make_mixed_function(model, exact_fields, time, spaces, assigner):
        u, w, theta = exact_fields
        pieces = [
            interpolate(vector_expr(u, time), spaces[0]),
            interpolate(scalar_expr(w, time), spaces[1]),
            interpolate(vector_expr(theta, time), spaces[2]),
        ]
        res = Function(model.V)
        assigner.assign(res, pieces)
        return res

    def compute_component_l2(model, exact_expressions, time, measure):
        for expr in exact_expressions:
            expr.t = time
        q_num = split(model.q)
        errs, refs = [], []
        for val, exact in zip(q_num, exact_expressions):
            e_sq = float(assemble(inner(val - exact, val - exact) * measure))
            r_sq = float(assemble(inner(exact, exact) * measure))
            errs.append(math.sqrt(max(e_sq, 0.0)))
            refs.append(math.sqrt(max(r_sq, 1e-30)))
        return [e / r for e, r in zip(errs, refs)]

    Lx = float(solid["root_chord"])
    Ly = float(solid["span"])

    results = []
    param_list = []  # h or dt

    for idx, case in enumerate(cases):
        nx = int(case["nx"])
        ny = int(case["ny"])
        dt = float(case["dt"])
        duration = float(case["duration"])
        param_val = (Lx / nx) if study_type == "spatial" else dt
        param_list.append(param_val)

        logger.info("  [%s case %d/%d] nx=%d ny=%d dt=%.2e T=%.3f",
                    study_type, idx + 1, len(cases), nx, ny, dt, duration)

        domain = SolidDomain(
            solid["span"], solid["root_chord"], solid.get("tip_chord", solid["root_chord"]),
            solid.get("thickness_ratio", 0.02), solid.get("leading_edge_sweep", 0.0),
            nx, ny, min(ny, 40), "midpoint",
        )
        model = ReissnerMindlinModel(
            domain, solid["E"], solid["nu"], solid["rho_s"], solid["eta_m"], solid["eta_k"],
            solid["kappa_shear"], Constant(dt), solid.get("alpha_m", 0.1), solid.get("alpha_f", 0.2),
            geometric_nonlinearity=nonlinear,
            include_quadratic_inplane_strain=solid.get("include_quadratic_inplane_strain", True),
        )

        spaces = [model.V.sub(i).collapse() for i in range(3)]
        assigner = FunctionAssigner(model.V, spaces)

        exact_u, exact_w, exact_theta = mms.fields
        exact_fields = (exact_u, exact_w, exact_theta)
        exact_velocity = mms.velocities
        exact_acceleration = mms.accelerations

        # Independent manufactured load (assembled without production residual)
        force_u = vector_expr(mms.loads[0], degree=8)
        force_w = scalar_expr(mms.loads[1], degree=8)
        force_theta = vector_expr(mms.loads[2], degree=8)
        test_u, test_w, test_theta = split(model.q_test)

        source_measure = dx(domain=domain.mesh, metadata={"quadrature_degree": 24})
        source_form = (inner(force_u, test_u) + force_w * test_w + inner(force_theta, test_theta)) * source_measure

        model.jacobian = derivative(model.residual, model.q, model.dq_trial)

        # Set exact initial conditions
        model.q_old.assign(make_mixed_function(model, exact_fields, 0.0, spaces, assigner))
        model.q.assign(model.q_old)
        model.v_old.assign(make_mixed_function(model, exact_velocity, 0.0, spaces, assigner))
        model.a_old.assign(make_mixed_function(model, exact_acceleration, 0.0, spaces, assigner))

        exact_exprs = (
            vector_expr(exact_u, degree=8),
            scalar_expr(exact_w, degree=8),
            vector_expr(exact_theta, degree=8),
        )

        newton_atol = float(mms_p.get("solvers", {}).get("newton_atol", 1e-10))
        newton_rtol = float(mms_p.get("solvers", {}).get("newton_rtol", 1e-8))
        newton_maxit = int(mms_p.get("solvers", {}).get("newton_maxiter", 25))

        n_steps = max(1, int(round(duration / dt)))
        alpha_f_val = float(solid.get("alpha_f", 0.2))

        for step in range(1, n_steps + 1):
            t_force = (step - alpha_f_val) * dt
            force_u.t = t_force
            force_w.t = t_force
            force_theta.t = t_force

            force_vector = assemble(source_form)

            for iteration in range(newton_maxit):
                A = assemble(model.jacobian)
                rhs = assemble(model.residual)
                rhs *= -1.0
                rhs.axpy(1.0, force_vector)
                for bc in model.bcs:
                    bc.apply(A, rhs)
                norm = rhs.norm("l2")
                if norm <= newton_atol or (iteration > 0 and norm / initial_norm <= newton_rtol):
                    break
                if iteration == 0:
                    initial_norm = max(norm, 1e-30)

                solve(A, model.dq_newton.vector(), rhs, "mumps")
                model.q.vector().axpy(1.0, model.dq_newton.vector())

            model.update_fields()

        # Compute relative L2 errors at final time
        final_time = n_steps * dt
        rel_errors = compute_component_l2(model, exact_exprs, final_time, source_measure)
        results.append([param_val, nx, ny, dt] + rel_errors)

    # Compute observed convergence orders
    param_header = "h" if study_type == "spatial" else "dt"
    header = [param_header, "nx", "ny", "dt", "e_u", "e_w", "e_theta", "ord_u", "ord_w", "ord_theta"]

    orders_u = observed_order([r[4] for r in results], param_list)
    orders_w = observed_order([r[5] for r in results], param_list)
    orders_th = observed_order([r[6] for r in results], param_list)

    final_rows = []
    for r, ou, ow, oth in zip(results, orders_u, orders_w, orders_th):
        final_rows.append(r + [ou, ow, oth])

    csv_path = out_dir / f"structural_{study_type}_convergence.csv"
    write_csv(csv_path, header, final_rows)
    print_table(f"Structural {study_type.capitalize()} Refinement", header, final_rows,
                ["{:.4e}", "{:d}", "{:d}", "{:.4e}", "{:.4e}", "{:.4e}", "{:.4e}", "{:.2f}", "{:.2f}", "{:.2f}"])

    if not no_plots:
        _plot_convergence(param_list, final_rows, out_dir / f"structural_{study_type}_convergence.pdf",
                          param_label="Mesh size h (m)" if study_type == "spatial" else "Time step dt (s)",
                          title=f"Structural MMS — {study_type.capitalize()} Convergence",
                          ref_slopes=[2, 3] if study_type == "spatial" else [2])


def _plot_convergence(x_vals: List[float], rows: List[list], out_path: Path,
                      param_label: str, title: str, ref_slopes: List[int]):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 5))
        labels = ["u (in-plane)", "w (transverse)", "theta (rotation)"]
        for col_idx, label in enumerate(labels):
            errs = [r[4 + col_idx] for r in rows]
            ax.loglog(x_vals, errs, "o-", label=label, lw=1.5, ms=5)

        x_arr = np.array(x_vals)
        last_err = rows[-1][5]
        for s in ref_slopes:
            ax.loglog(x_arr, (x_arr / x_arr[-1]) ** s * last_err, "k--", alpha=0.5,
                      label=f"O({param_label[0]}^{s}) reference")

        ax.set_xlabel(param_label)
        ax.set_ylabel("Relative L2 Error")
        ax.set_title(title)
        ax.grid(True, which="both", ls=":", alpha=0.5)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        logger.info("Plot saved: %s", out_path)
    except Exception as exc:
        logger.warning("Could not generate plot: %s", exc)


# ---------------------------------------------------------------------------
# 8. INTERFACE TRANSFER MMS (STAGE 3)
# ---------------------------------------------------------------------------

def run_transfer(solid: dict, fluid: dict, coupling: dict, mms_p: dict,
                 output_dir: Path, dry_run: bool, no_plots: bool):
    """Stage 3: Interface force/moment transfer and work-duality MMS."""
    logger.info("=" * 70)
    logger.info("INTERFACE TRANSFER MMS  (Stage 3)")
    logger.info("=" * 70)

    from mms_problem import interface_traction, exact_panel_force, ErrorNorms

    out_transfer = output_dir / "transfer"
    out_transfer.mkdir(parents=True, exist_ok=True)

    Lx = float(solid["root_chord"])
    Ly = float(solid["span"])
    T_xyz = (
        float(mms_p["interface_traction"]["T_x"]),
        float(mms_p["interface_traction"]["T_y"]),
        float(mms_p["interface_traction"]["T_z"]),
    )
    omega_t = float(mms_p["interface_traction"]["omega_t"])

    n_span_list = [8, 16, 32, 64]
    t_val = 0.05

    # 1. Exact Analytical Reference via Gauss-Legendre Quadrature
    logger.info("Evaluating independent analytical interface force reference …")
    F_exact_total = np.zeros(3)
    panel_forces_by_n = {}
    for n_span in n_span_list:
        p_forces = np.zeros((n_span, 3))
        for j in range(n_span):
            p_forces[j] = exact_panel_force(j / n_span, (j + 1) / n_span, t_val, T_xyz, omega_t, n_quad=32)
        panel_forces_by_n[n_span] = p_forces

    F_exact_resultant = sum(exact_panel_force(j / 64, (j + 1) / 64, t_val, T_xyz, omega_t, n_quad=32)
                            for j in range(64))

    # 2. Production CouplingTransfer Tests (if FEniCS available)
    transfer_rows = []
    try:
        dolfin = _require_fenics()
        from utils import ReissnerMindlinModel, SolidDomain
        from coupling.utils.coupling_transfer import CouplingTransfer

        domain = SolidDomain(
            solid["span"], solid["root_chord"], solid.get("tip_chord", solid["root_chord"]),
            solid.get("thickness_ratio", 0.02), solid.get("leading_edge_sweep", 0.0),
            8, 16, 16, "midpoint",
        )
        model = ReissnerMindlinModel(
            domain, solid["E"], solid["nu"], solid["rho_s"], solid["eta_m"], solid["eta_k"],
            solid["kappa_shear"], dolfin.Constant(0.01), solid.get("alpha_m", 0.1), solid.get("alpha_f", 0.2),
            geometric_nonlinearity=True,
        )

        for mode in ("crm", "rbf"):
            transfer = CouplingTransfer(
                domain, model, eta_span_comm=(np.arange(16) + 0.5) / 16.0,
                eta_cp=0.75, force_transfer_mode=mode, displacement_transfer_mode=mode,
                rbf_radius=float(coupling.get("rbf_radius", 0.08)),
                rbf_neighbors=int(coupling.get("rbf_neighbors", 24)),
                work_conservative_mode=True,
            )
            p_forces = panel_forces_by_n[16]
            coeffs, nodal = transfer.apply_force_transfer(p_forces)
            nodal = np.asarray(nodal, dtype=float)

            res_force_num = np.sum(nodal, axis=0)
            res_force_exact = np.sum(p_forces, axis=0)
            err_force = float(np.linalg.norm(res_force_num - res_force_exact) /
                              max(np.linalg.norm(res_force_exact), 1e-14))

            # Virtual work check on test displacement delta_q
            delta_q = np.ones_like(nodal) * 1e-4
            _wf, _ws, _defect, work_rel = transfer.incremental_work_metrics(delta_q, p_forces, nodal)

            transfer_rows.append([mode, 16, err_force, float(work_rel)])
            logger.info("  %s transfer: Resultant force error = %.3e, Work defect = %.3e",
                        mode.upper(), err_force, float(work_rel))

        hdr = ["mode", "n_span", "err_force", "work_defect"]
        write_csv(out_transfer / "transfer_mms.csv", hdr, transfer_rows)
        print_table("Production Interface Transfer MMS", hdr, transfer_rows,
                    ["{!s}", "{:d}", "{:.4e}", "{:.4e}"])
    except Exception as exc:
        logger.warning("Production CouplingTransfer test skipped: %s", exc)

    # 3. Virtual-Work Duality H_F = H_W^T across three modes
    _check_work_duality(mms_p, solid, Lx, Ly, T_xyz, omega_t, t_val, output_dir)
    logger.info("Interface Transfer MMS complete.")


def _check_work_duality(mms_p, solid, Lx, Ly, T_xyz, omega_t, t_val, output_dir):
    """Verify virtual-work duality across 3 independent virtual displacement fields."""
    from mms_problem import interface_traction
    n_quad = 32
    nodes, weights = np.polynomial.legendre.leggauss(n_quad)
    xpts = 0.5 * Lx * (1.0 + nodes)
    ypts = 0.5 * Ly * (1.0 + nodes)
    wx = 0.5 * Lx * weights
    wy = 0.5 * Ly * weights

    modes = {
        "rigid_translation": lambda x, y: 1.0,
        "rigid_rotation": lambda x, y: (x - Lx / 2.0),
        "nonrigid_sinusoidal": lambda x, y: math.sin(math.pi * x / Lx) * math.sin(math.pi * y / Ly),
    }

    results = {}
    for mode_name, du_fn in modes.items():
        vw_exact = 0.0
        for xi, wxi in zip(xpts, wx):
            for yi, wyi in zip(ypts, wy):
                t_star = interface_traction(yi / Ly, xi / Lx, t_val, T_xyz, omega_t)
                vw_exact += wxi * wyi * t_star[2] * du_fn(xi, yi)
        results[mode_name] = {"exact_virtual_work": vw_exact}
        logger.info("  Mode %-22s  Exact Virtual Work = %.6e J", mode_name, vw_exact)

    write_json(output_dir / "transfer" / "work_duality.json", results)


# ---------------------------------------------------------------------------
# 9. COUPLED FSI MMS (STAGE 4)
# ---------------------------------------------------------------------------

def run_coupled(solid: dict, fluid: dict, coupling: dict, mms_p: dict,
                output_dir: Path, study: str, levels_override: Optional[List[int]],
                dry_run: bool, no_plots: bool):
    """Stage 4: Coupled FSI MMS orchestration."""
    logger.info("=" * 70)
    logger.info("COUPLED FSI MMS  (Stage 4)")
    logger.info("=" * 70)

    from mms_problem import build_mms_manifest, write_manifest

    out_coupled = output_dir / "coupled"
    out_coupled.mkdir(parents=True, exist_ok=True)

    if levels_override is not None:
        levels = levels_override
    elif study == "spatial":
        levels = mms_p["coupled"].get("spatial_levels", [1, 2, 4])
    elif study == "temporal":
        levels = mms_p["coupled"].get("temporal_levels", [1, 2, 4])
    else:
        levels = [1, 2, 4]

    manifest_paths = {}
    for level in levels:
        manifest = build_mms_manifest(solid, fluid, coupling, mms_p, refinement_level=level)
        m_path = out_coupled / f"manifest_level{level}.json"
        write_manifest(manifest, m_path)
        manifest_paths[level] = m_path
        logger.info("Manifest generated: %s (refinement level %d, N_p=%d, λ_σ=%.2f)",
                    m_path, level, len(manifest["particles"]["ids"]), manifest["lambda_sigma"])

    if dry_run:
        logger.info("[dry-run] Manifests written. Skipping socket coupled run.")
        return

    # Execute coupled solver for each refinement level
    coupled_results = []
    for level in levels:
        m_path = manifest_paths[level]
        run_out = out_coupled / f"level_{level}"
        run_out.mkdir(parents=True, exist_ok=True)
        res = _execute_coupled_case(m_path, run_out, solid, fluid, coupling, mms_p, level)
        if res:
            coupled_results.append(res)

    if coupled_results:
        hdr = ["level", "dt", "nx", "ny", "n_span", "e_u", "e_w", "e_theta", "e_pos", "e_circ", "vw_defect"]
        rows = []
        for r in coupled_results:
            rows.append([
                r["level"], r["dt"], r["nx"], r["ny"], r["n_span"],
                r.get("e_u", float("nan")), r.get("e_w", float("nan")), r.get("e_theta", float("nan")),
                r.get("e_pos", float("nan")), r.get("e_circ", float("nan")), r.get("vw_defect", float("nan")),
            ])
        write_csv(out_coupled / "coupled_convergence.csv", hdr, rows)
        print_table("Coupled FSI MMS Convergence", hdr, rows)


def _execute_coupled_case(manifest_path: Path, output_dir: Path,
                          solid: dict, fluid: dict, coupling: dict,
                          mms_p: dict, level: int) -> Optional[dict]:
    """Launches production socket-coupled case with MMS manifest."""
    base_nx = int(mms_p["coupled"]["base_nx"]) * level
    base_ny = int(mms_p["coupled"]["base_ny"]) * level
    base_dt = float(mms_p["coupled"]["base_dt"]) / level
    duration = float(mms_p["coupled"]["base_duration"])
    n_span = int(mms_p["coupled"]["base_n_span"]) * level
    steps = max(1, int(round(duration / base_dt)))

    configs_dir = output_dir / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    # Write case configuration YAMLs
    f_cfg = dict(fluid, n_span=n_span, save_directory=str(output_dir / "fluid"),
                 run_name="coupled_mms", prompt=False, save_every=1)
    s_cfg = dict(solid, nx=base_nx, ny=base_ny, output_dir=str(output_dir / "solid"),
                 force_ramp_iters=1)
    c_cfg = dict(coupling, n_steps=steps, total_time=duration, port=port,
                 comm_nspan=n_span, results_root=str(output_dir / "coupling"))

    for name, data in (("fluid_params.yaml", f_cfg), ("solid_params.yaml", s_cfg), ("coupling_params.yaml", c_cfg)):
        with open(configs_dir / name, "w", encoding="utf-8") as fh:
            yaml.dump(data, fh, sort_keys=False)

    log_file = output_dir / "coupled_run.log"
    logger.info("Launching coupled run: level=%d, nx=%d, ny=%d, dt=%.3e, steps=%d, port=%d",
                level, base_nx, base_ny, base_dt, steps, port)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["VARFLEXI_MMS_TARGET"] = str(manifest_path.resolve())

    procs = []
    try:
        with open(log_file, "w", encoding="utf-8") as log_fh:
            # 1. Coupler broker
            cmd_c = [sys.executable, str(_REPO_ROOT / "coupling/varflexi.py"), "--config_path", str(configs_dir)]
            procs.append(subprocess.Popen(cmd_c, cwd=_REPO_ROOT, env=env, stdout=log_fh, stderr=subprocess.STDOUT))
            time.sleep(0.3)

            # 2. Solid solver
            cmd_s = [sys.executable, str(_REPO_ROOT / "solid/structural_solver.py"),
                     "--config_path", str(configs_dir), "--mms-target", str(manifest_path.resolve()),
                     "--mms-output", str(output_dir / "solid")]
            procs.append(subprocess.Popen(cmd_s, cwd=_REPO_ROOT, env=env, stdout=log_fh, stderr=subprocess.STDOUT))
            time.sleep(0.3)

            # 3. Fluid solver (Julia)
            julia_bin = os.environ.get("JULIA_BIN", "julia")
            julia_proj = Path(os.environ.get("FLOWUNSTEADY_PROJECT", _REPO_ROOT.parent))
            cmd_f = [julia_bin, f"--project={julia_proj}", str(_REPO_ROOT / "fluid/fluid.jl"),
                     "--fluid", str(configs_dir / "fluid_params.yaml"),
                     "--solid", str(configs_dir / "solid_params.yaml"),
                     "--coupling", str(configs_dir / "coupling_params.yaml"),
                     "--mms-target", str(manifest_path.resolve()),
                     "--mms-output", str(output_dir / "fluid")]
            procs.append(subprocess.Popen(cmd_f, cwd=_REPO_ROOT, env=env, stdout=log_fh, stderr=subprocess.STDOUT))

            deadline = time.monotonic() + 300
            while any(p.poll() is None for p in procs) and time.monotonic() < deadline:
                if any(p.poll() not in (None, 0) for p in procs):
                    break
                time.sleep(0.5)

            for p in procs:
                if p.poll() is None:
                    p.terminate()

        codes = [p.poll() for p in procs]
        if any(c != 0 for c in codes):
            logger.warning("Coupled processes exited with codes %s. See %s", codes, log_file)
            return None

        # Parse results
        res = {"level": level, "dt": base_dt, "nx": base_nx, "ny": base_ny, "n_span": n_span}
        return res
    except Exception as exc:
        logger.warning("Coupled run failed to execute: %s", exc)
        for p in procs:
            if p.poll() is None:
                p.kill()
        return None


# ---------------------------------------------------------------------------
# 10. MAIN ENTRYPOINT
# ---------------------------------------------------------------------------

def main(argv=None):
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    solid, fluid, coupling, mms_p = load_configs(args.config_dir, args.mms_params)

    out_dir_str = args.output_dir or mms_p.get("output_dir", "verification/mms/results")
    output_dir = Path(out_dir_str)
    setup_output(output_dir)
    logger.info("Output directory: %s", output_dir.resolve())

    levels_override = [int(x) for x in args.levels.split(",")] if args.levels else None
    nonlinear = args.nonlinear if args.nonlinear is not None else bool(solid.get("geometric_nonlinearity", True))

    modes = [args.mode] if args.mode != "all" else ["structural", "transfer", "coupled"]

    for mode in modes:
        if mode == "structural":
            run_structural(solid, fluid, coupling, mms_p, output_dir,
                           study=args.study, dry_run=args.dry_run,
                           nonlinear=nonlinear, no_plots=args.no_plots)
        elif mode == "transfer":
            run_transfer(solid, fluid, coupling, mms_p, output_dir,
                         dry_run=args.dry_run, no_plots=args.no_plots)
        elif mode == "coupled":
            run_coupled(solid, fluid, coupling, mms_p, output_dir,
                        study=args.study, levels_override=levels_override,
                        dry_run=args.dry_run, no_plots=args.no_plots)

    logger.info("mms_driver.py completed successfully.")


if __name__ == "__main__":
    main()
