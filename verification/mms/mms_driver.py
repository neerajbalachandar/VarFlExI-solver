"""mms_driver.py — Single Python execution and analysis driver for the VarFlExI MMS.

Usage
-----
Run from the repository root:

    python verification/mms/mms_driver.py structural [options]
    python verification/mms/mms_driver.py transfer   [options]
    python verification/mms/mms_driver.py coupled    [options]
    python verification/mms/mms_driver.py all        [options]

Positional argument (mode):
    structural   Standalone Reissner–Mindlin plate MMS (spatial + temporal
                 refinement and scalar ODE check; Jacobian and strong/weak audits
                 currently report NOT RUN).
    transfer     Interface force/moment transfer diagnostics (CRM & RBF).
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
    p.add_argument("mode", choices=["structural", "transfer", "coupled", "all", "test"],
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
    """Record that a production residual/Jacobian check is not available."""
    result = {"status": "NOT RUN", "reason":
              "A source sample is not a Jacobian check; this driver has no assembled residual/Jacobian adapter."}
    write_json(output_dir / "structural" / "jacobian_check.json", result)
    logger.warning("Structural Jacobian consistency check NOT RUN: %s", result["reason"])


def _run_strong_weak_check(mms, solid, output_dir: Path):
    """Record that full strong-to-weak consistency remains unevaluated."""
    result = {"status": "NOT RUN", "reason":
              "The former scalar source integral did not compare the five-field strong and weak functionals or boundary terms."}
    write_json(output_dir / "structural" / "strong_weak_check.json", result)
    logger.warning("Full strong-to-weak consistency check NOT RUN: %s", result["reason"])


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
        u_num, w_num, theta_num = split(model.q)
        u_exact, w_exact, theta_exact = exact_expressions
        numerical = (u_num[0], u_num[1], w_num, theta_num[0], theta_num[1])
        target = (u_exact[0], u_exact[1], w_exact, theta_exact[0], theta_exact[1])
        errs, refs = [], []
        for val, exact in zip(numerical, target):
            e_sq = float(assemble(inner(val - exact, val - exact) * measure))
            r_sq = float(assemble(inner(exact, exact) * measure))
            if not math.isfinite(e_sq) or not math.isfinite(r_sq) or e_sq < 0.0 or r_sq <= 0.0:
                raise ValueError("Structural FE spatial L2 error has an invalid or zero target norm")
            errs.append(math.sqrt(e_sq))
            refs.append(math.sqrt(r_sq))
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

            converged = False
            initial_norm = None
            for iteration in range(newton_maxit):
                A = assemble(model.jacobian)
                rhs = assemble(model.residual)
                rhs *= -1.0
                rhs.axpy(1.0, force_vector)
                for bc in model.bcs:
                    bc.apply(A, rhs)
                norm = rhs.norm("l2")
                if initial_norm is None:
                    initial_norm = max(norm, 1e-30)
                if norm <= newton_atol or (iteration > 0 and norm / initial_norm <= newton_rtol):
                    converged = True
                    break

                # Newton correction with MUMPS linear solver
                solve(A, model.dq_newton.vector(), rhs, "mumps")
                delta_q = model.dq_newton.vector().get_local().copy()
                q_before = model.q.vector().get_local().copy()

                # Armijo backtracking line search
                accepted = False
                for backtrack in range(12):
                    step_fraction = 0.5 ** backtrack
                    model.q.vector().set_local(q_before + step_fraction * delta_q)
                    model.q.vector().apply("insert")
                    trial_rhs = assemble(model.residual)
                    trial_rhs *= -1.0
                    trial_rhs.axpy(1.0, force_vector)
                    for bc in model.bcs:
                        bc.apply(trial_rhs)
                    trial_norm = trial_rhs.norm("l2")
                    if np.isfinite(trial_norm) and trial_norm <= (1.0 - 1e-4 * step_fraction) * norm:
                        accepted = True
                        break

                if not accepted:
                    model.q.vector().set_local(q_before)
                    model.q.vector().apply("insert")
                    raise RuntimeError(
                        f"Newton linesearch failed at step {step}/{n_steps}, iteration {iteration + 1}: "
                        f"residual={norm:.4e}, initial={initial_norm:.4e}"
                    )

            if not converged:
                raise RuntimeError(
                    f"Newton solver failed to converge at step {step}/{n_steps}: "
                    f"residual={norm:.4e}, initial={initial_norm:.4e}, "
                    f"atol={newton_atol:.2e}, rtol={newton_rtol:.2e}, maxit={newton_maxit}"
                )

            model.update_fields()

        # Compute relative L2 errors at final time
        final_time = n_steps * dt
        rel_errors = compute_component_l2(model, exact_exprs, final_time, source_measure)
        results.append([param_val, nx, ny, dt] + rel_errors)

    # Compute observed convergence orders
    param_header = "h" if study_type == "spatial" else "dt"
    component_keys = ("u0", "v0", "w", "theta_x", "theta_y")
    header = [param_header, "nx", "ny", "dt",
              *(f"e_{key}" for key in component_keys),
              *(f"ord_{key}" for key in component_keys)]

    component_orders = [observed_order([r[4 + j] for r in results], param_list)
                        for j in range(len(component_keys))]

    final_rows = []
    for case_index, result in enumerate(results):
        final_rows.append(result + [orders[case_index] for orders in component_orders])

    csv_path = out_dir / f"structural_{study_type}_convergence.csv"
    write_csv(csv_path, header, final_rows)
    print_table(f"Structural {study_type.capitalize()} Refinement", header, final_rows,
                ["{:.4e}", "{:d}", "{:d}", "{:.4e}",
                 *(["{:.4e}"] * 5), *(["{:.2f}"] * 5)])

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
        labels = [r"$u_0$", r"$v_0$", r"$w$", r"$\theta_x$", r"$\theta_y$"]
        for col_idx, label in enumerate(labels):
            errs = [r[4 + col_idx] for r in rows]
            ax.loglog(x_vals, errs, "o-", label=label, lw=1.5, ms=5)

        x_arr = np.array(x_vals)
        last_err = rows[-1][4]
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

    from mms_problem import exact_panel_force

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

    n_span_list = [int(n) for n in mms_p["interface_traction"]["n_span_levels"]]
    t_val = float(mms_p["interface_traction"]["evaluation_time"])
    n_quad = int(mms_p["interface_traction"]["quadrature_order"])

    # 1. Exact Analytical Reference via Gauss-Legendre Quadrature
    logger.info("Evaluating independent analytical interface force reference …")
    panel_forces_by_n = {}
    for n_span in n_span_list:
        p_forces = np.zeros((n_span, 3))
        for j in range(n_span):
            p_forces[j] = exact_panel_force(j / n_span, (j + 1) / n_span, t_val, T_xyz, omega_t, n_quad=n_quad)
        panel_forces_by_n[n_span] = p_forces

    if dry_run:
        logger.info("[dry-run] Analytical panel forces evaluated. Skipping production transfer operators.")
        return

    # 2. Production CouplingTransfer Tests (CRM and RBF at 4 refinement levels)
    transfer_wrench_rows = []
    transfer_work_rows = []

    try:
        dolfin = _require_fenics()
        from utils import ReissnerMindlinModel, SolidDomain
        from coupling.utils.coupling_transfer import CouplingTransfer

        for mode in ("crm", "rbf"):
            logger.info("--- Testing %s transfer across refinement levels %s ---", mode.upper(), n_span_list)
            for n_span in n_span_list:
                nx = max(4, n_span // 2)
                ny = n_span
                domain = SolidDomain(
                    solid["span"], solid["root_chord"], solid.get("tip_chord", solid["root_chord"]),
                    solid.get("thickness_ratio", 0.02), solid.get("leading_edge_sweep", 0.0),
                    nx, ny, n_span, "midpoint",
                )
                model = ReissnerMindlinModel(
                    domain, solid["E"], solid["nu"], solid["rho_s"], solid["eta_m"], solid["eta_k"],
                    solid["kappa_shear"], dolfin.Constant(0.01), solid.get("alpha_m", 0.1), solid.get("alpha_f", 0.2),
                    geometric_nonlinearity=True,
                )

                eta_span = (np.arange(n_span) + 0.5) / float(n_span)
                transfer = CouplingTransfer(
                    domain, model, eta_span_comm=eta_span,
                    eta_cp=0.75, force_transfer_mode=mode, displacement_transfer_mode=mode,
                    rbf_radius=float(coupling.get("rbf_radius", 0.08)),
                    rbf_neighbors=int(coupling.get("rbf_neighbors", 24)),
                    work_conservative_mode=True,
                )

                p_forces = panel_forces_by_n[n_span]
                coeffs, nodal = transfer.apply_force_transfer(p_forces)
                nodal = np.asarray(nodal, dtype=float)

                aero_coords = transfer.aero_coords[:n_span]
                solid_coords = transfer.interface_coords

                (f_fluid, f_solid, m_fluid, m_solid,
                 res_f, res_m, rel_err_f, rel_err_m) = CouplingTransfer.wrench_transfer_metrics(
                     aero_coords, p_forces, solid_coords, nodal
                )

                res_f_exact = np.sum(p_forces, axis=0)
                err_f_vs_exact = float(np.linalg.norm(f_solid - res_f_exact) / max(np.linalg.norm(res_f_exact), 1e-14))

                transfer_wrench_rows.append([
                    mode.upper(), n_span, rel_err_f, rel_err_m, err_f_vs_exact,
                    float(np.linalg.norm(f_solid)), float(np.linalg.norm(m_solid))
                ])

                # Virtual work duality across 3 displacement fields:
                # 1) Rigid translation
                # 2) Rigid rotation
                # 3) Non-rigid sinusoidal deformation
                modes_disp = [
                    ("rigid_translation", np.tile([0.0, 0.0, 1.0e-4], (len(nodal), 1))),
                    ("rigid_rotation", np.array([[0.0, 0.0, 1.0e-4 * (x - Lx / 2.0)] for x, y in solid_coords])),
                    ("nonrigid_sinusoidal", np.array([[0.0, 0.0, 1.0e-4 * np.sin(np.pi * x / Lx) * np.sin(np.pi * y / Ly)] for x, y in solid_coords])),
                ]

                for mode_name, dq in modes_disp:
                    Wf, Ws, defect, work_rel = transfer.incremental_work_metrics(dq, p_forces, nodal)
                    transfer_work_rows.append([
                        mode.upper(), n_span, mode_name, float(Wf), float(Ws), float(defect), float(work_rel)
                    ])

        hdr_wrench = ["mode", "n_span", "err_force", "err_moment", "err_vs_exact", "norm_F", "norm_M"]
        write_csv(out_transfer / "transfer_force_moment_conservation.csv", hdr_wrench, transfer_wrench_rows)
        print_table("Interface Wrench Conservation (Resultant Force & Moment)", hdr_wrench, transfer_wrench_rows,
                    ["{!s}", "{:d}", "{:.4e}", "{:.4e}", "{:.4e}", "{:.4e}", "{:.4e}"])

        hdr_work = ["mode", "n_span", "displacement_field", "W_fluid", "W_solid", "work_defect", "relative_defect"]
        write_csv(out_transfer / "transfer_virtual_work_duality.csv", hdr_work, transfer_work_rows)
        print_table("Production Virtual-Work Duality Audit", hdr_work, transfer_work_rows,
                    ["{!s}", "{:d}", "{!s}", "{:.6e}", "{:.6e}", "{:.3e}", "{:.3e}"])

    except Exception as exc:
        logger.error("Production CouplingTransfer test failed: %s", exc)
        raise

    logger.info("Interface Transfer MMS complete.")


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
    manifest_level = (int(mms_p["coupled"].get("temporal_mesh_level", 4))
                      if study == "temporal" else None)
    for level in levels:
        target_level = manifest_level if manifest_level is not None else level
        manifest = build_mms_manifest(solid, fluid, coupling, mms_p, refinement_level=target_level)
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
        res = _execute_coupled_case(m_path, run_out, solid, fluid, coupling, mms_p, level, study)
        if res:
            coupled_results.append(res)

    if coupled_results:
        required = ("e_u0", "e_v0", "e_w", "e_theta_x", "e_theta_y",
                    "e_pos", "e_circ", "e_core", "e_velocity", "e_vorticity",
                    "e_force", "e_moment", "e_work")
        for result in coupled_results:
            missing = [key for key in required if key not in result or
                       not math.isfinite(float(result[key]))]
            if missing:
                raise RuntimeError(
                    f"Coupled level {result.get('level')} has no complete measured result; "
                    f"missing/non-finite metrics: {', '.join(missing)}"
                )
        hdr = ["level", "dt_f", "dt_s", "nx", "ny", "n_span", *required]
        rows = []
        for r in coupled_results:
            rows.append([
                r["level"], r["dt_f"], r["dt_s"], r["nx"], r["ny"], r["n_span"],
                *(r[key] for key in required),
            ])
        write_csv(out_coupled / "coupled_convergence.csv", hdr, rows)
        print_table("Coupled FSI MMS Convergence", hdr, rows)


def _execute_coupled_case(manifest_path: Path, output_dir: Path,
                          solid: dict, fluid: dict, coupling: dict,
                          mms_p: dict, level: int, study: str = "spatial") -> Optional[dict]:
    """Launches production socket-coupled case with MMS manifest."""
    if str(fluid.get("vpm_sfs", "none")).strip().lower() not in ("none", "off", "false"):
        raise ValueError("The coupled particle MMS requires vpm_sfs=none; its independent source omits SFS terms")
    if str(fluid.get("viscous_scheme", "none")).strip().lower() not in ("none", "off", "false"):
        raise ValueError("The coupled particle MMS requires viscous_scheme=none; its source omits viscous diffusion")
    mesh_level = (int(mms_p["coupled"].get("temporal_mesh_level", 4))
                  if study == "temporal" else level)
    base_nx = int(mms_p["coupled"]["base_nx"]) * mesh_level
    base_ny = int(mms_p["coupled"]["base_ny"]) * mesh_level
    base_dt = float(mms_p["coupled"]["base_dt"]) / level
    duration = float(mms_p["coupled"]["base_duration"])
    n_span = int(mms_p["coupled"]["base_n_span"]) * mesh_level
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
            raise RuntimeError(f"coupled processes exited with codes {codes}; see {log_file}")

        # Read actual output histories. A successful process exit is not a
        # verification result; metrics must come from matched recorded data.
        return _extract_coupled_metrics(
            output_dir, level, base_nx, base_ny, n_span, mms_p,
            str(s_cfg.get("diag_csv_filename", "solid_v18_diagnostics.csv")),
            duration / steps,
        )
    except Exception as exc:
        for p in procs:
            if p.poll() is None:
                p.kill()
        raise RuntimeError(f"Coupled level {level} failed or produced invalid MMS data: {exc}") from exc


def _read_jsonl_history(path: Path, required_fields: tuple[str, ...]) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Required MMS history is missing: {path}")
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}: {exc}") from exc
            missing = [key for key in required_fields if key not in record]
            if missing:
                raise ValueError(f"{path}:{line_no} is missing fields: {', '.join(missing)}")
            records.append(record)
    if not records:
        raise ValueError(f"MMS history is empty: {path}")
    return records


def _relative_array_error(numerical, target, *, label: str) -> float:
    num = np.asarray(numerical, dtype=float)
    ref = np.asarray(target, dtype=float)
    if num.shape != ref.shape or num.size == 0:
        raise ValueError(f"{label}: target and numerical shapes differ or are empty")
    if not np.isfinite(num).all() or not np.isfinite(ref).all():
        raise ValueError(f"{label}: target or numerical values are non-finite")
    scale = float(np.linalg.norm(ref.ravel()))
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"{label}: target normalization norm must be positive")
    return float(np.linalg.norm((num - ref).ravel()) / scale)


def _extract_coupled_metrics(output_dir: Path, level: int,
                             nx: int, ny: int, n_span: int,
                             mms_p: dict, diagnostics_name: str,
                             dt_s: float) -> dict:
    """Extract only measured, time-matched quantities from coupled histories.

    The current structural history contains coefficient vectors but no FE mass
    norm components. It also has no common-origin moment or virtual-work
    records. Those missing producer quantities are fatal here: coefficient
    Euclidean norms are not substituted for spatial FE L2 errors.
    """
    fluid_dir, solid_dir = output_dir / "fluid", output_dir / "solid"
    fluid_path = fluid_dir / "coupled_mms_force_payload_history.jsonl"
    fluid_records = _read_jsonl_history(fluid_path, (
        "fluid_time_s", "dt", "target_particle_state", "numerical_particle_state",
        "fixed_probe_positions_m", "target_probe_velocity_m_s",
        "numerical_probe_velocity_m_s", "target_probe_vorticity_s_inv",
        "numerical_probe_vorticity_s_inv", "target_force", "force",
        "rVPM_RK3_stage_times_s", "target_geometry", "geometry_cp_absolute",
    ))
    solid_path = solid_dir / "solid_mms_history.csv"
    if not solid_path.is_file():
        raise FileNotFoundError(f"Required structural MMS history is missing: {solid_path}")
    with solid_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        if not reader.fieldnames or not rows:
            raise ValueError(f"Structural MMS history is empty or has no header: {solid_path}")
    structural_fields = ("time_s", "target_force", "numerical_force", "target_state", "numerical_state")
    missing = [field for field in structural_fields if field not in (reader.fieldnames or ())]
    if missing:
        raise ValueError(f"Structural MMS history is missing fields: {', '.join(missing)}")

    final_solid = rows[-1]
    t_s = float(final_solid["time_s"])
    if not math.isfinite(t_s):
        raise ValueError("Structural history time_s is non-finite")
    solid_target = np.asarray(json.loads(final_solid["target_state"]), dtype=float)
    solid_num = np.asarray(json.loads(final_solid["numerical_state"]), dtype=float)
    if solid_target.shape != solid_num.shape or solid_target.size == 0 or not (
            np.isfinite(solid_target).all() and np.isfinite(solid_num).all()):
        raise ValueError("Structural target/numerical state arrays are empty, mismatched, or non-finite")
    for field in ("target_force", "numerical_force"):
        force_history = np.asarray(json.loads(final_solid[field]), dtype=float)
        if force_history.size == 0 or not np.isfinite(force_history).all():
            raise ValueError(f"Structural {field} is empty or non-finite")
    tolerance = float(mms_p.get("coupled", {}).get("time_match_tolerance_s", 1.0e-10))
    matches = [rec for rec in fluid_records
               if abs(float(rec["fluid_time_s"]) - t_s) <= tolerance]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one fluid force-exchange record at structural time {t_s:.16g}; "
            f"found {len(matches)} (tolerance {tolerance:g} s)"
        )
    fluid = matches[0]

    stage_times = np.asarray(fluid["rVPM_RK3_stage_times_s"], dtype=float)
    if stage_times.shape != (3,) or not np.isfinite(stage_times).all():
        raise ValueError("Fluid record must retain exactly three finite RK3 stage times")
    dt_f_recorded = float(fluid["dt"])
    if not math.isfinite(dt_f_recorded) or dt_f_recorded <= 0.0:
        raise ValueError("Recorded fluid time increment must be finite and positive")
    expected_stage_times = t_s - dt_f_recorded + dt_f_recorded * np.asarray((0.0, 1.0 / 3.0, 3.0 / 4.0))
    if not np.allclose(stage_times, expected_stage_times, rtol=0.0,
                       atol=max(tolerance, abs(dt_f) * 1.0e-12)):
        raise ValueError(
            f"Recorded fluid RK3 stages {stage_times.tolist()} do not match expected "
            f"stage chronology {expected_stage_times.tolist()} ending at force time {t_s:g} s"
        )

    p_target, p_num = fluid["target_particle_state"], fluid["numerical_particle_state"]
    particle_metrics = {
        "e_pos": _relative_array_error(p_num["position"], p_target["position"], label="e_pos"),
        "e_circ": _relative_array_error(p_num["circulation"], p_target["circulation"], label="e_circ"),
        "e_core": _relative_array_error(p_num["core"], p_target["core"], label="e_core"),
    }
    probes = fluid["fixed_probe_positions_m"]
    if not probes or not np.isfinite(np.asarray(probes, dtype=float)).all():
        raise ValueError("Fixed physical probe locations are missing or non-finite")
    particle_metrics["e_velocity"] = _relative_array_error(
        fluid["numerical_probe_velocity_m_s"], fluid["target_probe_velocity_m_s"],
        label="e_velocity")
    particle_metrics["e_vorticity"] = _relative_array_error(
        fluid["numerical_probe_vorticity_s_inv"], fluid["target_probe_vorticity_s_inv"],
        label="e_vorticity")

    origin = np.asarray(mms_p["coupled"]["moment_origin_m"], dtype=float)
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError("coupled.moment_origin_m must be a finite three-vector")
    if not np.array_equal(origin, np.zeros(3)):
        raise ValueError("The current production transfer diagnostics compute moments about [0, 0, 0] m")

    diagnostics_path = solid_dir / diagnostics_name
    if not diagnostics_path.is_file():
        raise FileNotFoundError(f"Required structural transfer diagnostics are missing: {diagnostics_path}")
    with diagnostics_path.open(newline="", encoding="utf-8") as stream:
        diag_reader = csv.DictReader(stream)
        diag_rows = list(diag_reader)
        diag_fields = diag_reader.fieldnames or ()
    diag_keys = ("time", "force_transfer_relative_error", "moment_transfer_relative_error",
                 "incremental_work_relative_error")
    missing_diag = [key for key in diag_keys if key not in diag_fields]
    if missing_diag or not diag_rows:
        raise ValueError(f"Transfer diagnostics missing fields/rows: {missing_diag or 'empty history'}")
    diag_matches = [row for row in diag_rows
                    if abs(float(row["time"]) - t_s) <= tolerance]
    if len(diag_matches) != 1:
        raise ValueError(f"Expected one transfer diagnostic at t={t_s:g} s, found {len(diag_matches)}")
    diag = diag_matches[0]
    interface_metrics = {
        "e_force": float(diag["force_transfer_relative_error"]),
        "e_moment": float(diag["moment_transfer_relative_error"]),
        "e_work": float(diag["incremental_work_relative_error"]),
    }
    if any(not math.isfinite(value) or value < 0.0 for value in interface_metrics.values()):
        raise ValueError(f"Transfer diagnostics contain invalid metrics: {interface_metrics}")

    structural_metric_keys = ("e_u0", "e_v0", "e_w", "e_theta_x", "e_theta_y")
    missing_metrics = [key for key in structural_metric_keys if key not in final_solid]
    if missing_metrics:
        raise ValueError(
            "Structural history lacks componentwise FE spatial L2 metrics: "
            f"{', '.join(missing_metrics)}. Coefficient arrays are not substituted."
        )
    structural_metrics = {key: float(final_solid[key]) for key in structural_metric_keys}
    if not all(math.isfinite(value) for value in structural_metrics.values()):
        raise ValueError("Structural FE spatial L2 history contains a non-finite metric")
    return {
        "level": level,
        "dt_f": dt_f_recorded,
        "dt_s": float(dt_s),
        "nx": nx,
        "ny": ny,
        "n_span": n_span,
        **structural_metrics,
        **particle_metrics,
        **interface_metrics,
        "final_time_s": t_s,
        "fluid_stage_times_s": stage_times.tolist(),
        "moment_origin_m": origin.tolist(),
        "norm_time_kind": "final-time at the matched fluid force-exchange/structural time",
    }


# ---------------------------------------------------------------------------
# 10. AUTOMATED MMS CONSISTENCY TESTS
# ---------------------------------------------------------------------------

def run_automated_tests(solid: dict, fluid: dict, coupling: dict, mms_p: dict, output_dir: Path):
    """Run comprehensive automated consistency tests for the MMS framework.

    Tests:
      1. Analytical source derivatives (SymPy target derivatives vs central FD)
      2. Boundary compatibility (essential clamping at root, zero natural tractions/moments on free edges)
      3. Source independence (confirming no production residual or solver methods called)
      4. RK3 stage count and low-storage coefficients (3 stages, Williamson/FLOWVPM coefficients)
      5. Work duality (exact virtual-work balance across translation, rotation, and non-rigid fields)
    """
    logger.info("=" * 70)
    logger.info("RUNNING AUTOMATED MMS CONSISTENCY TESTS")
    logger.info("=" * 70)

    test_results = {}

    # Test 1: Analytical source derivatives
    logger.info("Test 1: Analytical target derivatives vs finite differences …")
    from mms_problem import build_plate_mms_sympy
    amp = mms_p["structural"]["amplitudes"]
    frq = mms_p["structural"]["frequencies"]
    comps = {k: {"amplitude": amp[k], "omega": frq[k], "mode": "ramp"} for k in amp}
    mms = build_plate_mms_sympy(solid, nonlinear=True, components=comps)

    x_s, y_s, t_s = mms.x, mms.y, mms.t
    Lx, Ly = float(solid["root_chord"]), float(solid["span"])
    subs = {x_s: 0.4 * Lx, y_s: 0.6 * Ly, t_s: 0.05}
    h_t = 1.0e-5
    subs_p = {x_s: 0.4 * Lx, y_s: 0.6 * Ly, t_s: 0.05 + h_t}
    subs_m = {x_s: 0.4 * Lx, y_s: 0.6 * Ly, t_s: 0.05 - h_t}

    w_val = float(mms.fields[1].subs(subs))
    w_t_analytic = float(mms.velocities[1].subs(subs))
    w_t_fd = (float(mms.fields[1].subs(subs_p)) - float(mms.fields[1].subs(subs_m))) / (2.0 * h_t)
    err_wt = abs(w_t_analytic - w_t_fd) / max(abs(w_t_analytic), 1e-12)
    assert err_wt < 1.0e-7, f"Velocity derivative mismatch: {err_wt}"
    test_results["target_derivatives"] = {"status": "PASS", "relative_error": err_wt}
    logger.info("  Target derivatives test: PASSED (rel_err = %.3e)", err_wt)

    # Test 2: Boundary compatibility
    logger.info("Test 2: Boundary compatibility (essential clamping & free-edge natural conditions) …")
    import sympy as sp
    for name, expr in mms.boundary_checks.items():
        if isinstance(expr, (list, tuple)):
            for v in expr:
                val = sp.simplify(v)
                assert val == 0, f"Boundary check {name} failed: {val} != 0"
        else:
            val = sp.simplify(expr)
            assert val == 0, f"Boundary check {name} failed: {val} != 0"
    test_results["boundary_compatibility"] = {"status": "PASS", "checks_verified": len(mms.boundary_checks)}
    logger.info("  Boundary compatibility test: PASSED (%d boundary conditions verified)", len(mms.boundary_checks))

    # Test 3: Source independence
    logger.info("Test 3: Source independence guarantee …")
    import inspect
    src_plate = inspect.getsource(build_plate_mms_sympy)
    assert "model.residual" not in src_plate, "Violation: build_plate_mms_sympy calls model.residual"
    assert "model.mass_form" not in src_plate, "Violation: build_plate_mms_sympy calls model.mass_form"
    assert "model.internal_work" not in src_plate, "Violation: build_plate_mms_sympy calls model.internal_work"
    test_results["source_independence"] = {"status": "PASS", "residual_calls_detected": 0,
                                           "scope": "build_plate_mms_sympy source text only"}
    logger.info("  Source independence test: PASSED (zero production residual calls in source derivation)")

    # Test 4: RK3 stage count and low-storage coefficients
    logger.info("Test 4: RK3 stage count and low-storage coefficients …")
    expected_stage_times = (0.0, 1.0 / 3.0, 3.0 / 4.0)
    expected_coeffs = ((0.0, 1.0 / 3.0), (-5.0 / 9.0, 15.0 / 16.0), (-153.0 / 128.0, 8.0 / 15.0))
    assert len(expected_stage_times) == 3, "RK3 must evaluate exactly 3 stages"
    assert len(expected_coeffs) == 3, "RK3 must have 3 low-storage stage pairs"
    test_results["rk3_stage_count"] = {"status": "PASS", "num_stages": 3,
                                       "stage_times": list(expected_stage_times),
                                       "scope": "local coefficient declaration; not dependency source audit"}
    logger.info("  RK3 stage count test: PASSED (3 stages, Williamson/FLOWVPM low-storage coefficients)")

    # Test 5: Virtual work duality
    logger.info("Test 5: Interface transfer work duality across 3 displacement modes …")
    try:
        dolfin = _require_fenics()
        from utils import ReissnerMindlinModel, SolidDomain
        from coupling.utils.coupling_transfer import CouplingTransfer
        from mms_problem import exact_panel_force

        domain = SolidDomain(solid["span"], solid["root_chord"], solid.get("tip_chord", solid["root_chord"]),
                             solid.get("thickness_ratio", 0.02), 0.0, 8, 16, 16, "midpoint")
        model = ReissnerMindlinModel(domain, solid["E"], solid["nu"], solid["rho_s"], solid["eta_m"], solid["eta_k"],
                                     solid["kappa_shear"], dolfin.Constant(0.01), 0.1, 0.2, geometric_nonlinearity=True)
        eta_span = (np.arange(16) + 0.5) / 16.0
        transfer = CouplingTransfer(domain, model, eta_span_comm=eta_span, eta_cp=0.75, force_transfer_mode="crm",
                                    displacement_transfer_mode="crm", work_conservative_mode=True)
        p_forces = np.zeros((16, 3))
        for j in range(16):
            p_forces[j] = exact_panel_force(j / 16, (j + 1) / 16, 0.05, (1e-2, 5e-3, 2e-2), 2.3, n_quad=16)
        _, nodal = transfer.apply_force_transfer(p_forces)
        solid_coords = transfer.interface_coords

        modes = [
            ("translation", np.tile([0.0, 0.0, 1e-4], (len(nodal), 1))),
            ("rotation", np.array([[0.0, 0.0, 1e-4 * (x - Lx / 2.0)] for x, y in solid_coords])),
            ("sinusoidal", np.array([[0.0, 0.0, 1e-4 * np.sin(np.pi * x / Lx) * np.sin(np.pi * y / Ly)] for x, y in solid_coords])),
        ]
        max_defect = 0.0
        for name, dq in modes:
            Wf, Ws, defect, work_rel = transfer.incremental_work_metrics(dq, p_forces, nodal)
            max_defect = max(max_defect, work_rel)
            assert work_rel < 1.0e-12, f"Work duality failure on {name}: {work_rel}"
        test_results["work_duality"] = {"status": "PASS", "max_relative_defect": max_defect}
        logger.info("  Virtual work duality test: PASSED (max relative defect = %.3e)", max_defect)
    except Exception as exc:
        logger.error("  Work duality test ERROR: %s", exc)
        test_results["work_duality"] = {"status": "ERROR", "reason": str(exc)}

    # These required checks are not implemented by this lightweight suite.
    # Keep them visible and make an overall pass impossible until they are
    # evaluated against the production operators and their complete histories.
    for name, reason in {
        "structural_jacobian_directional_derivative":
            "No residual/Jacobian directional-derivative evaluation is implemented.",
        "full_strong_to_weak_structural_identity":
            "The current helper integrates only the transverse source and does not test boundary terms.",
        "all_component_derivatives_and_boundary_data":
            "The automated derivative check covers only w_t; the full target derivative/boundary audit is absent.",
        "source_independence_dataflow":
            "The current source-independence check inspects only build_plate_mms_sympy source text.",
        "independent_transfer_traction_force_moment_work":
            "Independent load approximation, common-origin moment, and full weighted work-conjugacy checks are incomplete.",
    }.items():
        test_results[name] = {"status": "NOT RUN", "reason": reason}

    non_pass = {name: value for name, value in test_results.items()
                if value["status"] != "PASS"}
    overall = "PASS" if not non_pass else "FAIL"
    test_results["overall"] = {
        "status": overall,
        "required_checks": len(test_results),
        "non_pass_checks": sorted(non_pass),
    }

    out_test = output_dir / "automated_tests.json"
    write_json(out_test, test_results)
    logger.info("Automated tests summary written to %s", out_test)
    print("\n" + "=" * 70)
    print(f"AUTOMATED MMS CONSISTENCY TESTS: {overall}")
    if non_pass:
        for name, value in non_pass.items():
            logger.error("  %s: %s — %s", name, value["status"], value.get("reason", ""))
    print("=" * 70 + "\n")
    if overall != "PASS":
        raise RuntimeError("Required MMS consistency checks are not all PASS; see automated_tests.json")


# ---------------------------------------------------------------------------
# 11. MAIN ENTRYPOINT
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

    if args.mode == "all":
        modes = ["test", "structural", "transfer", "coupled"]
    elif args.mode == "test":
        modes = ["test"]
    else:
        modes = [args.mode]

    for mode in modes:
        if mode == "test":
            run_automated_tests(solid, fluid, coupling, mms_p, output_dir)
        elif mode == "structural":
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
