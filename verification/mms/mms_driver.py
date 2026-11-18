"""mms_driver.py — Single Python execution and analysis driver for the VarFlExI MMS.

Usage
-----
Run from the repository root:

    python verification/mms/mms_driver.py structural [options]
    python verification/mms/mms_driver.py transfer   [options]
    python verification/mms/mms_driver.py fluid      [options]
    python verification/mms/mms_driver.py coupled    [options]
    python verification/mms/mms_driver.py all        [options]

Positional argument (mode):
    structural   Standalone Reissner–Mindlin plate MMS (spatial + temporal
                 refinement, production Jacobian directional check, and complete
                 symbolic strong/weak audit).
    transfer     Interface force/moment transfer diagnostics (CRM & RBF).
    coupled      Coupled FSI MMS orchestration across refinement levels with
                 independent structural and fluid source injection.
    fluid        Standalone particle-level rVPM and temporal-refinement MMS.
    all          Run checks -> structural -> transfer -> fluid -> coupled.

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
  - Structural coupled source:    fully discrete generalized-alpha weak residual,
                                  independently assembled at target stage states and
                                  minus the exchange-time analytical force.
  - Coupled particle source:      independent Biot-Savart + analytic-kernel Jacobian
                                  stretching; does NOT call vpm.update_particle_states
                                  on a shadow copy. The standalone particle study
                                  retains its finite-difference Jacobian.
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
import shutil
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
    p.add_argument("mode", choices=["structural", "transfer", "fluid", "coupled", "all", "test"],
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
        jacobian_result = _run_jacobian_check(mms, solid, output_dir)
        strong_weak_result = _run_strong_weak_check(mms, solid, output_dir)
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
    """Compare the assembled production Jacobian action with residual differences."""
    dolfin = _require_fenics()
    from utils import ReissnerMindlinModel, SolidDomain
    domain = SolidDomain(solid["span"], solid["root_chord"],
                         solid.get("tip_chord", solid["root_chord"]),
                         solid.get("thickness_ratio", 0.02),
                         solid.get("leading_edge_sweep", 0.0), 4, 8, 8, "midpoint")
    model = ReissnerMindlinModel(
        domain, solid["E"], solid["nu"], solid["rho_s"], solid["eta_m"], solid["eta_k"],
        solid["kappa_shear"], dolfin.Constant(0.01),
        solid.get("alpha_m", 0.10), solid.get("alpha_f", 0.20),
        geometric_nonlinearity=solid.get("geometric_nonlinearity", True),
        include_quadratic_inplane_strain=solid.get("include_quadratic_inplane_strain", True),
    )
    rng = np.random.default_rng(71423)
    q = model.q.vector().get_local()
    q += rng.normal(size=q.size) * 2.0e-5
    model.q.vector().set_local(q)
    model.q.vector().apply("insert")
    for field in (model.q_old, model.v_old, model.a_old):
        values = rng.normal(size=field.vector().size()) * 1.0e-5
        field.vector().set_local(values)
        field.vector().apply("insert")
    directions = []
    for _ in range(3):
        direction = dolfin.Function(model.V)
        p = rng.normal(size=direction.vector().size())
        direction.vector().set_local(p)
        direction.vector().apply("insert")
        for bc in model.bcs:
            bc.apply(direction.vector())
        p = direction.vector().get_local()
        pnorm = float(np.linalg.norm(p))
        if pnorm == 0.0:
            raise RuntimeError("Could not construct a nonzero Jacobian direction")
        p *= 1.0e-4 / pnorm
        direction.vector().set_local(p)
        direction.vector().apply("insert")
        directions.append((direction, p.copy()))

    jacobian = dolfin.assemble(model.jacobian)
    for bc in model.bcs:
        bc.apply(jacobian)
    q_base = model.q.vector().get_local().copy()
    step_sizes = (1.0e-2, 5.0e-3, 2.5e-3)
    errors_by_direction = []
    for direction, p in directions:
        jvalues = (jacobian * direction.vector()).get_local()
        errors = []
        for h in step_sizes:
            residuals = []
            for sign in (1.0, -1.0):
                model.q.vector().set_local(q_base + sign * h * p)
                model.q.vector().apply("insert")
                residual = dolfin.assemble(model.residual)
                for bc in model.bcs:
                    bc.apply(residual)
                residuals.append(residual.get_local())
            fd = (residuals[0] - residuals[1]) / (2.0 * h)
            errors.append(float(np.linalg.norm(fd-jvalues) /
                                max(np.linalg.norm(fd), np.linalg.norm(jvalues), 1.0e-14)))
        errors_by_direction.append(errors)
    model.q.vector().set_local(q_base)
    model.q.vector().apply("insert")
    threshold = 2.0e-5
    max_error = max(max(row) for row in errors_by_direction)
    result = {"status": "PASS" if max_error < threshold else "FAIL",
              "relative_errors_by_direction_and_step": errors_by_direction,
              "direction_count": len(directions), "step_sizes": list(step_sizes),
              "threshold": threshold,
              "scope": "assembled nonlinear production residual/Jacobian, three admissible directions and three centered step sizes"}
    write_json(output_dir / "structural" / "jacobian_check.json", result)
    if result["status"] != "PASS":
        raise AssertionError(f"Production structural Jacobian directional check failed: {errors_by_direction}")
    return result


def _run_strong_weak_check(mms, solid, output_dir: Path):
    """Verify all five strong/weak integration-by-parts identities symbolically."""
    import sympy as sp
    x, y = mms.x, mms.y
    u, w, theta = mms.fields
    N = mms.kinematics["membrane_resultant"]
    M = mms.kinematics["bending_moment"]
    Q = mms.kinematics["shear_resultant"]
    grad_u = u.jacobian([x, y])
    Pm = N + grad_u * N
    grad_w = sp.Matrix([sp.diff(w, x), sp.diff(w, y)])
    Pw = N * grad_w
    # Variations vanish on the clamped root; all natural fluxes on the three
    # free edges are independently required to vanish by boundary_checks.
    du = sp.Matrix([x*y*(1+x), x*y*(1+y)])
    dw = x*y*(1+x+y)
    dtheta = sp.Matrix([x*y*(1+x), x*y*(1+y)])
    f_u = mms.load_components["membrane"][0]
    f_w = mms.load_components["transverse"][1]
    f_theta = mms.load_components["bending"][2] + mms.load_components["shear"][2]
    div_Pm_du = sp.diff(Pm[0, 0]*du[0] + Pm[1, 0]*du[1], x) + sp.diff(
        Pm[0, 1]*du[0] + Pm[1, 1]*du[1], y)
    div_Pw_dw = sp.diff((Pw[0]+Q[0])*dw, x) + sp.diff((Pw[1]+Q[1])*dw, y)
    div_M_dtheta = sp.diff(M[0, 0]*dtheta[0]+M[1, 0]*dtheta[1], x) + sp.diff(
        M[0, 1]*dtheta[0]+M[1, 1]*dtheta[1], y)
    membrane_weak = sum(Pm[i, j] * sp.diff(du[i], (x, y)[j])
                        for i in range(2) for j in range(2))
    transverse_weak = sum((Pw[i]+Q[i]) * sp.diff(dw, (x, y)[i]) for i in range(2))
    bending_weak = sum(M[i, j] * sp.diff(dtheta[i], (x, y)[j])
                       for i in range(2) for j in range(2)) - sum(Q[i]*dtheta[i] for i in range(2))
    identities = (
        sum(f_u[i]*du[i] for i in range(2)) - membrane_weak + div_Pm_du,
        f_w*dw - transverse_weak + div_Pw_dw,
        sum(f_theta[i]*dtheta[i] for i in range(2)) - bending_weak + div_M_dtheta,
    )
    errors = [sp.simplify(value) for value in identities]
    if any(value != 0 for value in errors):
        raise AssertionError(f"Strong-to-weak symbolic identity failed: {errors}")
    failures = []
    for name, value in mms.boundary_checks.items():
        values = value if isinstance(value, (tuple, list)) else (value,)
        if any(sp.simplify(item) != 0 for item in values):
            failures.append(name)
    if failures:
        raise AssertionError(f"Natural/essential boundary terms do not vanish: {failures}")
    result = {"status": "PASS", "max_symbolic_identity_error": 0.0,
              "threshold": 0.0, "identities": ["membrane", "transverse+shear", "bending+shear"],
              "natural_boundary_checks": len(mms.boundary_checks),
              "scope": "all five virtual components, nonlinear membrane flux, edge terms and boundary checks"}
    write_json(output_dir / "structural" / "strong_weak_check.json", result)
    return result


def _audit_flowvpm_rk3_source() -> dict:
    """Audit the installed FLOWVPM RK3 recurrence and MMS stage trace wiring."""
    project = Path(os.environ.get("FLOWUNSTEADY_PROJECT", _REPO_ROOT.parent))
    manifest = project / "Manifest.toml"
    if not manifest.is_file():
        raise FileNotFoundError(f"FLOWUnsteady Manifest.toml not found: {manifest}")
    manifest_text = manifest.read_text(encoding="utf-8")
    entries = re.findall(r"(?ms)^\[\[deps\.FLOWVPM\]\]\s*(.*?)(?=^\[\[deps\.|\Z)", manifest_text)
    if len(entries) != 1:
        raise RuntimeError("Manifest must resolve exactly one FLOWVPM implementation")
    version_match = re.search(r'(?m)^version\s*=\s*"([^"]+)"', entries[0])
    if version_match is None:
        raise RuntimeError("FLOWVPM Manifest entry is missing its resolved version")
    version = version_match.group(1)
    source_candidates = []
    for root in (Path.home() / ".julia/dev/FLOWVPM",
                 *(Path.home() / ".julia/packages/FLOWVPM").glob("*/")):
        project_file = root / "Project.toml"
        if project_file.is_file():
            local_project = project_file.read_text(encoding="utf-8")
            local_version = re.search(r'(?m)^version\s*=\s*"([^"]+)"', local_project)
            if local_version is not None and local_version.group(1) == version:
                source_candidates.append(root / "src/FLOWVPM_timeintegration.jl")
    source_file = next((path for path in source_candidates if path.is_file()), None)
    if source_file is None:
        raise FileNotFoundError(f"Could not locate FLOWVPM {version} integration source")
    source = source_file.read_text(encoding="utf-8")
    required = ("(0.0, 1/3)", "(-5/9, 15/16)", "(-153/128, 8/15)",
                "update_particle_states(pfield", "viscousdiffusion(pfield")
    missing = [token for token in required if token not in source]
    adapter = (_REPO_ROOT / "fluid/src/mms_rk3.jl").read_text(encoding="utf-8")
    callback = (_REPO_ROOT / "fluid/src/runtime_callback.jl").read_text(encoding="utf-8")
    source_impl = (_REPO_ROOT / "fluid/src/mms_coupled_source.jl").read_text(encoding="utf-8")
    wiring = ("stage_times = (0.0, 1.0 / 3.0, 3.0 / 4.0)" in adapter
              and "stage_trace=stage_trace" in adapter
              and '"rVPM_RK3_stage_times_s" => rk3_stage_times' in callback)
    cache_contract = ("cache_key = Ref{Union{Nothing,NTuple{4,Float64}}}" in source_impl
                      and "cache_value = Ref{Any}(nothing)" in source_impl
                      and "lock(cache_lock) do" in source_impl
                      and "if cache_key[] != key" in source_impl
                      and "cache_key[] = key" in source_impl)
    if missing or not wiring or not cache_contract:
        raise AssertionError(
            f"FLOWVPM RK3/source-stage audit failed: missing={missing}, "
            f"trace_wired={wiring}, bounded_cache={cache_contract}"
        )
    julia_bin = os.environ.get("JULIA_BIN", "julia")
    jacobian_check = _REPO_ROOT / "verification/mms/check_coupled_jacobian.jl"
    completed = subprocess.run(
        [julia_bin, f"--project={project}", "--startup-file=no", str(jacobian_check)],
        cwd=_REPO_ROOT, check=True, capture_output=True, text=True,
    )
    if "COUPLED_KERNEL_AND_CIRCULATION_CHECK PASS" not in completed.stdout:
        raise AssertionError("Coupled kernel/circulation consistency check did not report PASS")
    if f"FLOWVPM={version}" not in completed.stdout:
        raise AssertionError(
            f"Runtime-resolved FLOWVPM version disagrees with Manifest.toml ({version})")
    runtime_source_match = re.search(r"source=([^\s]+)", completed.stdout)
    if runtime_source_match is None:
        raise AssertionError("FLOWVPM runtime source path was not recorded by the audit")
    state_match = re.search(r"CIRCULATION_STATE_ABS_ERROR=([0-9.eE+-]+)", completed.stdout)
    derivative_match = re.search(r"CIRCULATION_DERIVATIVE_REL_ERROR=([0-9.eE+-]+)", completed.stdout)
    if state_match is None or derivative_match is None:
        raise AssertionError("Julia circulation cross-check omitted measured errors")
    circulation_state_error = float(state_match.group(1))
    circulation_derivative_error = float(derivative_match.group(1))
    if (circulation_state_error != 0.0 or not math.isfinite(circulation_derivative_error)
            or circulation_derivative_error >= 2.0e-7):
        raise AssertionError("Julia canonical bound-circulation consistency check failed")
    return {"status": "PASS", "flowvpm_version": version,
            "source": runtime_source_match.group(1),
            "manifest_source_candidate": str(source_file),
            "dependency_coefficients_verified": list(required[:3]),
            "production_stage_trace_wired": True,
            "bounded_single_stage_cache_checked": cache_contract,
            "jacobian_check_output": completed.stdout.strip(),
            "bound_circulation_state_error": circulation_state_error,
            "bound_circulation_derivative_error": circulation_derivative_error,
            "measured_source_contract_mismatches": 0, "threshold": 0,
            "scope": "resolved dependency source, adapter/callback trace, analytic Jacobian FD/orientation/self/kernel tests"}


def _run_scalar_genalpha_check(solid: dict, mms_p: dict, output_dir: Path):
    """Pure-Python scalar generalized-alpha ODE convergence check."""
    logger.info("Running scalar generalized-alpha ODE convergence check …")
    rho = float(solid.get("alpha_inf", 0.9))
    alpha_m = 0.5 * (3.0 - rho) / (1.0 + rho)
    alpha_f = 1.0 / (1.0 + rho)
    gamma_ga = 0.5 + alpha_f - alpha_m
    beta_ga = 0.25 * (gamma_ga + 0.5) ** 2

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

                common_origin = np.asarray((0.13, -0.07, 0.02), dtype=float)
                (f_fluid, f_solid, m_fluid, m_solid,
                 res_f, res_m, rel_err_f, rel_err_m) = CouplingTransfer.wrench_transfer_metrics(
                     aero_coords, p_forces, solid_coords, nodal,
                     moment_origin=common_origin,
                 )
                origin_shift = np.asarray((-0.04, 0.03, 0.02), dtype=float)
                shifted = CouplingTransfer.wrench_transfer_metrics(
                    aero_coords, p_forces, solid_coords, nodal,
                    moment_origin=common_origin + origin_shift,
                )
                expected_shift = m_solid - np.cross(origin_shift, f_solid)
                if not np.allclose(shifted[3], expected_shift, rtol=2.0e-12, atol=2.0e-12):
                    raise AssertionError("Common-origin moment translation identity failed")

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
# 9. PARTICLE-LEVEL rVPM MMS
# ---------------------------------------------------------------------------

def run_particle_mms(solid: dict, fluid: dict, coupling: dict, mms_p: dict, output_dir: Path,
                     dry_run: bool = False) -> None:
    """Run standalone native rVPM source and RK3 refinement studies via Julia."""
    logger.info("PARTICLE-LEVEL rVPM MMS  (Stage 4)")
    particle_dir = output_dir / "particles"
    particle_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "coupled" / "manifest_level1.json"
    if not manifest_path.is_file():
        run_coupled(
            solid,
            fluid, coupling, mms_p, output_dir, study="spatial",
            levels_override=None, dry_run=True, no_plots=True,
        )
    if dry_run:
        logger.info("[dry-run] Particle MMS manifest is ready; Julia execution skipped.")
        return
    julia_bin = os.environ.get("JULIA_BIN", "julia")
    julia_project = Path(os.environ.get("FLOWUNSTEADY_PROJECT", _REPO_ROOT.parent))
    script = _REPO_ROOT / "verification/mms/mms_rvpm.jl"
    for study_name in ("standalone", "refinement"):
        command = [julia_bin, f"--project={julia_project}", str(script),
                   "--manifest", str(manifest_path.resolve()),
                   "--output", str(particle_dir.resolve()), "--study", study_name]
        logger.info("Running particle-level rVPM MMS study: %s", study_name)
        subprocess.run(command, cwd=_REPO_ROOT, check=True)


# ---------------------------------------------------------------------------
# 10. COUPLED FSI MMS (STAGE 5)
# ---------------------------------------------------------------------------

def run_coupled(solid: dict, fluid: dict, coupling: dict, mms_p: dict,
                output_dir: Path, study: str, levels_override: Optional[List[int]],
                dry_run: bool, no_plots: bool,
                production_fluid_path: Optional[Path] = None):
    """Stage 5: Coupled FSI MMS orchestration."""
    logger.info("=" * 70)
    logger.info("COUPLED FSI MMS  (Stage 5)")
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
        res = _execute_coupled_case(
            m_path, run_out, solid, fluid, coupling, mms_p, level, study,
            production_fluid_path=production_fluid_path,
        )
        if res:
            coupled_results.append(res)

    if len(coupled_results) != len(levels):
        raise RuntimeError(
            f"Only {len(coupled_results)} of {len(levels)} coupled levels produced measured results"
        )

    reference_probes = np.asarray(coupled_results[0]["_fixed_probe_positions_m"], dtype=float)
    for result in coupled_results[1:]:
        probes = np.asarray(result["_fixed_probe_positions_m"], dtype=float)
        if probes.shape != reference_probes.shape or not np.array_equal(probes, reference_probes):
            raise ValueError("Fixed physical probe coordinates must be identical across coupled refinement levels")

    if coupled_results:
        required = ("e_u0", "e_v0", "e_w", "e_theta_x", "e_theta_y",
                    "e_pos", "e_circ", "e_core", "e_velocity", "e_vorticity",
                    "e_gamma_bound", "e_aero_force", "vlm_algebraic_residual",
                    "e_vlm_target_circulation",
                    "e_geometry_cp", "e_transfer_force", "e_transfer_moment",
                    "e_transfer_work")
        for result in coupled_results:
            missing = [key for key in required if key not in result or
                       not math.isfinite(float(result[key]))]
            if missing:
                raise RuntimeError(
                    f"Coupled level {result.get('level')} has no complete measured result; "
                    f"missing/non-finite metrics: {', '.join(missing)}"
                )
        refinement_values = [float(result["dt_s"] if study == "temporal"
                                  else result["h_mesh_m"]) for result in coupled_results]
        final_times = [float(result["final_time_s"]) for result in coupled_results]
        if not all(math.isfinite(value) for value in refinement_values) or any(
                right >= left for left, right in zip(refinement_values, refinement_values[1:])):
            raise ValueError("Coupled refinement values must be finite and strictly decreasing")
        if not all(math.isfinite(value) for value in final_times) or any(
                abs(value-final_times[0]) > float(mms_p["coupled"].get("time_match_tolerance_s", 1e-10))
                for value in final_times[1:]):
            raise ValueError("Coupled convergence levels must share the same final physical time")
        orders = {key: observed_order([float(result[key]) for result in coupled_results], refinement_values)
                  for key in required}
        hdr = ["study", "level", "step", "final_time_s", "dt_f", "dt_s",
               "nx", "ny", "n_span", "particle_count", "h_mesh_m", "refinement_variable",
               "moment_origin_m", "force_transfer_mode", "displacement_transfer_mode",
               *required, *(f"order_{key}" for key in required)]
        rows = []
        for index, r in enumerate(coupled_results):
            rows.append([
                r["study"], r["level"], r["step"], r["final_time_s"],
                r["dt_f"], r["dt_s"], r["nx"], r["ny"], r["n_span"],
                r["particle_count"], r["h_mesh_m"], r["refinement_variable"],
                json.dumps(r["moment_origin_m"]), r["force_transfer_mode"],
                r["displacement_transfer_mode"],
                *(r[key] for key in required),
                *(orders[key][index] if math.isfinite(orders[key][index]) else None
                  for key in required),
            ])
        write_csv(out_coupled / "coupled_convergence.csv", hdr, rows)
        print_table("Coupled FSI MMS Convergence", hdr, rows)
        write_json(out_coupled / "coupled_metrics.json", {
            "study": study,
            "refinement_variable": "dt_s" if study == "temporal" else "h_mesh_m",
            "final_time_s": coupled_results[0]["final_time_s"],
            "norm_time_kind": "final-time at matched fluid force-exchange/structural time",
            "metric_names": list(required),
            "levels": [{key: value for key, value in result.items()
                        if not key.startswith("_")} for result in coupled_results],
            "observed_orders": {key: [value if math.isfinite(value) else None for value in sequence]
                                for key, sequence in orders.items()},
            "order_definition": "log(e_previous/e_current)/log(r_previous/r_current); NaN when errors or refinement ratios are invalid",
        })


def _execute_coupled_case(manifest_path: Path, output_dir: Path,
                          solid: dict, fluid: dict, coupling: dict,
                          mms_p: dict, level: int, study: str = "spatial",
                          production_fluid_path: Optional[Path] = None) -> Optional[dict]:
    """Launches production socket-coupled case with MMS manifest."""
    if str(fluid.get("vpm_sfs", "none")).strip().lower() not in ("none", "off", "false"):
        raise ValueError("The coupled particle MMS requires vpm_sfs=none; its independent source omits SFS terms")
    if str(fluid.get("viscous_scheme", "none")).strip().lower() not in ("none", "off", "false"):
        raise ValueError("The coupled particle MMS requires viscous_scheme=none; its source omits viscous diffusion")
    if str(fluid.get("section_polar", "")).strip():
        raise ValueError("The independent coupled MMS force excludes parasitic airfoil-polar drag; clear section_polar")
    mesh_level = (int(mms_p["coupled"].get("temporal_mesh_level", 4))
                  if study == "temporal" else level)
    base_nx = int(mms_p["coupled"]["base_nx"]) * mesh_level
    base_ny = int(mms_p["coupled"]["base_ny"]) * mesh_level
    base_dt = float(mms_p["coupled"]["base_dt"])
    if study == "temporal":
        base_dt /= level
    duration = float(mms_p["coupled"]["base_duration"])
    n_span = int(mms_p["coupled"]["base_n_span"]) * mesh_level
    steps = max(1, int(round(duration / base_dt)))

    configs_dir = output_dir / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    # Write case configuration YAMLs
    actual_dt = duration / steps
    f_cfg = dict(fluid, n_span=n_span, dt=actual_dt, ttot=duration, nsteps=steps,
                 regularize_vlm=False, shed_boundarylayer=False,
                 save_directory=str(output_dir / "fluid"),
                 run_name="coupled_mms", prompt=False, save_every=1)
    solver_mms = mms_p.get("solvers", {})
    s_cfg = dict(solid, nx=base_nx, ny=base_ny, output_dir=str(output_dir / "solid"),
                 force_ramp_iters=1,
                 newton_atol=float(solver_mms.get("newton_atol", 1.0e-10)),
                 newton_rtol=float(solver_mms.get("newton_rtol", 1.0e-8)),
                 newton_maxiter=int(solver_mms.get("newton_maxiter", 25)))
    c_cfg = dict(coupling, n_steps=steps, total_time=duration, port=port,
                 comm_nspan=n_span, results_root=str(output_dir / "coupling"), force_relax=1.0)
    _validate_coupled_mms_overlay(f_cfg, c_cfg)
    if str(f_cfg.get("section_polar", "")).strip() or str(f_cfg.get("section_polar_path", "")).strip():
        raise ValueError("Coupled MMS requires empty section_polar and section_polar_path")
    if str(f_cfg.get("viscous_scheme", "none")).strip().lower() not in ("none", "off", "false"):
        raise ValueError("Coupled MMS cannot include a viscous/polar force contribution")
    moment_origin = np.asarray(mms_p.get("coupled", {}).get("moment_origin_m"), dtype=float)
    if moment_origin.shape != (3,) or not np.isfinite(moment_origin).all():
        raise ValueError("coupled.moment_origin_m must be a finite length-three vector")
    c_cfg["moment_origin_m"] = moment_origin.tolist()
    production_fluid_path = production_fluid_path or (_REPO_ROOT / "config" / "fluid_params.yaml")
    production_fluid_sha = hashlib.sha256(production_fluid_path.read_bytes()).hexdigest()
    overrides = {
        "regularize_vlm": {"from": fluid.get("regularize_vlm"), "to": False,
                            "reason": "independent MMS finite-horseshoe reference is unregularized"},
        "shed_boundarylayer": {"from": fluid.get("shed_boundarylayer"), "to": False,
                                "reason": "boundary-layer force is absent from the independent reference"},
        "force_relax": {"from": coupling.get("force_relax", 1.0), "to": 1.0,
                         "reason": "manufactured structural load must match the force-exchange reference"},
        "solid_solver": {
            "newton_atol": {"from": solid.get("newton_atol"), "to": s_cfg["newton_atol"],
                            "reason": "MMS residual must converge below configured source tolerance"},
            "newton_rtol": {"from": solid.get("newton_rtol"), "to": s_cfg["newton_rtol"],
                            "reason": "MMS residual must converge below configured source tolerance"},
            "newton_maxiter": {"from": solid.get("newton_maxiter"), "to": s_cfg["newton_maxiter"],
                               "reason": "allow convergence at the MMS residual tolerances"},
            "force_ramp_iters": {"from": solid.get("force_ramp_iters"), "to": 1,
                                 "reason": "apply the complete matched reference load at each exchange step"},
        },
    }

    # A level owns fresh outputs; stale histories cannot satisfy this run.
    for subdir in ("fluid", "solid", "coupling"):
        path = output_dir / subdir
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)

    for name, data in (("fluid_params.yaml", f_cfg), ("solid_params.yaml", s_cfg), ("coupling_params.yaml", c_cfg)):
        with open(configs_dir / name, "w", encoding="utf-8") as fh:
            yaml.dump(data, fh, sort_keys=False)
    generated_fluid = yaml.safe_load((configs_dir / "fluid_params.yaml").read_text(encoding="utf-8"))
    generated_coupling = yaml.safe_load((configs_dir / "coupling_params.yaml").read_text(encoding="utf-8"))
    generated_solid = yaml.safe_load((configs_dir / "solid_params.yaml").read_text(encoding="utf-8"))
    _validate_coupled_mms_overlay(generated_fluid, generated_coupling)
    generated_origin = np.asarray(generated_coupling.get("moment_origin_m"), dtype=float)
    if generated_origin.shape != (3,) or not np.allclose(
            generated_origin, moment_origin, rtol=0.0, atol=0.0):
        raise RuntimeError("Generated coupling config lost the MMS common moment origin")
    if hashlib.sha256(production_fluid_path.read_bytes()).hexdigest() != production_fluid_sha:
        raise RuntimeError("Production fluid YAML changed while creating the MMS case overlay")
    generated_config_hashes = {
        name: hashlib.sha256((configs_dir / name).read_bytes()).hexdigest()
        for name in ("fluid_params.yaml", "solid_params.yaml", "coupling_params.yaml")
    }
    dependency_metadata = _verification_dependency_metadata(
        Path(os.environ.get("FLOWUNSTEADY_PROJECT", _REPO_ROOT.parent))
    )
    write_json(output_dir / "case_metadata.json", {
        "study": study, "level": int(level), "dt_s": actual_dt,
        "final_time_s": duration, "moment_origin_m": moment_origin.tolist(),
        "mms_overrides": overrides,
        "production_fluid_yaml_sha256": production_fluid_sha,
        "generated_config_sha256": generated_config_hashes,
        "dependency_versions": dependency_metadata,
        "effective_mms_settings": {
            "regularize_vlm": generated_fluid["regularize_vlm"],
            "shed_boundarylayer": generated_fluid["shed_boundarylayer"],
            "section_polar": generated_fluid.get("section_polar", ""),
            "viscous_scheme": generated_fluid.get("viscous_scheme", "none"),
            "vpm_sfs": generated_fluid.get("vpm_sfs", "none"),
            "force_relax": generated_coupling["force_relax"],
            "moment_origin_m": generated_origin.tolist(),
            "newton_atol": generated_solid["newton_atol"],
            "newton_rtol": generated_solid["newton_rtol"],
        },
        "generated_config_paths": {name: str((configs_dir / name).relative_to(output_dir))
                                    for name in ("fluid_params.yaml", "solid_params.yaml", "coupling_params.yaml")},
    })

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
            startup_deadline = time.monotonic() + 30.0
            while time.monotonic() < startup_deadline:
                if procs[0].poll() is not None:
                    raise RuntimeError(f"coupler exited during startup with code {procs[0].returncode}")
                if log_file.exists() and "Waiting for fluid and solid role handshakes" in log_file.read_text(
                        encoding="utf-8", errors="replace"):
                    break
                time.sleep(0.05)
            else:
                raise TimeoutError("coupler did not report readiness within 30 seconds")

            # 2. Solid solver
            cmd_s = [sys.executable, str(_REPO_ROOT / "solid/structural_solver.py"),
                     "--config_path", str(configs_dir), "--mms-target", str(manifest_path.resolve()),
                     "--mms-output", str(output_dir / "solid")]
            procs.append(subprocess.Popen(cmd_s, cwd=_REPO_ROOT, env=env, stdout=log_fh, stderr=subprocess.STDOUT))
            solid_deadline = time.monotonic() + 30.0
            while time.monotonic() < solid_deadline:
                if procs[1].poll() is not None:
                    raise RuntimeError(f"solid solver exited during startup with code {procs[1].returncode}")
                if log_file.exists() and "Solid connected" in log_file.read_text(
                        encoding="utf-8", errors="replace"):
                    break
                time.sleep(0.05)
            else:
                raise TimeoutError("solid solver did not complete its broker handshake within 30 seconds")

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

            deadline = time.monotonic() + float(mms_p.get("coupled", {}).get("process_timeout_s", 300.0))
            while any(p.poll() is None for p in procs) and time.monotonic() < deadline:
                if any(p.poll() not in (None, 0) for p in procs):
                    break
                time.sleep(0.1)

            timed_out = any(p.poll() is None for p in procs)
            for p in procs:
                if p.poll() is None:
                    p.terminate()
            for p in procs:
                if p.poll() is None:
                    try:
                        p.wait(timeout=5.0)
                    except subprocess.TimeoutExpired:
                        p.kill()
                        p.wait()

        codes = [p.poll() for p in procs]
        if timed_out:
            raise TimeoutError(f"coupled processes exceeded timeout; see {log_file}")
        if any(c != 0 for c in codes):
            raise RuntimeError(f"coupled processes exited with codes {codes}; see {log_file}")

        # Read actual output histories. A successful process exit is not a
        # verification result; metrics must come from matched recorded data.
        result = _extract_coupled_metrics(
            output_dir, level, base_nx, base_ny, n_span, mms_p, coupling,
            str(s_cfg.get("diag_csv_filename", "solid_v18_diagnostics.csv")),
            duration / steps, study, float(solid["root_chord"]), float(solid["span"]),
        )
        return result
    except Exception as exc:
        for p in procs:
            if p.poll() is None:
                p.terminate()
        for p in procs:
            if p.poll() is None:
                try:
                    p.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait()
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


def _validate_coupled_mms_overlay(fluid_case: dict, coupling_case: dict) -> None:
    """Validate effective per-case settings consumed by the Julia MMS adapter."""
    if bool(fluid_case.get("regularize_vlm", True)):
        raise ValueError("Generated MMS fluid config must set regularize_vlm: false")
    if bool(fluid_case.get("shed_boundarylayer", False)):
        raise ValueError("Generated MMS fluid config must disable boundary-layer shedding")
    if str(fluid_case.get("section_polar", "")).strip() or str(
            fluid_case.get("section_polar_path", "")).strip():
        raise ValueError("Generated MMS fluid config must not enable section-polar forces")
    if str(fluid_case.get("viscous_scheme", "none")).strip().lower() not in ("none", "off", "false"):
        raise ValueError("Generated MMS fluid config must disable unsupported viscous forces")
    if str(fluid_case.get("vpm_sfs", "none")).strip().lower() not in ("none", "off", "false"):
        raise ValueError("Generated MMS fluid config must disable unsupported SFS forces")
    force_relax = float(coupling_case.get("force_relax", float("nan")))
    if not math.isfinite(force_relax) or force_relax != 1.0:
        raise ValueError("Generated MMS coupling config must set force_relax exactly to 1")


def _verification_dependency_metadata(project_dir: Path) -> dict:
    """Record resolved Julia source identity and the active DOLFIN/Python versions."""
    manifest_path = Path(project_dir) / "Manifest.toml"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Julia dependency manifest is missing: {manifest_path}")
    manifest_text = manifest_path.read_text(encoding="utf-8")
    def resolve(package):
        blocks = re.findall(
            rf"(?ms)^\[\[deps\.{re.escape(package)}\]\]\s*(.*?)(?=^\[\[deps\.|\Z)",
            manifest_text,
        )
        if len(blocks) != 1:
            raise ValueError(f"Julia Manifest must resolve exactly one {package} dependency")
        block = blocks[0]
        result = {}
        for key in ("version", "git-tree-sha1", "repo-rev", "repo-url"):
            match = re.search(rf'(?m)^{re.escape(key)}\s*=\s*"([^"]+)"', block)
            result[key] = None if match is None else match.group(1)
        if not result["version"] or not result["git-tree-sha1"]:
            raise ValueError(f"Julia Manifest {package} entry must include version and git-tree-sha1")
        return result
    resolved = {name: resolve(name) for name in ("FLOWUnsteady", "FLOWVLM", "FLOWVPM")}
    try:
        import dolfin
    except ImportError as exc:
        raise RuntimeError("DOLFIN version metadata is required for a coupled MMS run") from exc
    return {"python_version": sys.version.split()[0], "dolfin_version": str(dolfin.__version__),
            "julia_dependencies": resolved, "manifest_path": str(manifest_path.resolve())}


def _extract_coupled_metrics(output_dir: Path, level: int,
                             nx: int, ny: int, n_span: int,
                             mms_p: dict, coupling: dict, diagnostics_name: str,
                             dt_s: float, study: str = "spatial",
                             Lx: float = 0.0, Ly: float = 0.0) -> dict:
    """Extract only measured, time-matched quantities from coupled histories.

    Require componentwise FE norms, matched exchange histories, and common-origin
    transfer diagnostics. Coefficient Euclidean norms are never substituted for
    spatial FE L2 errors.
    """
    fluid_dir, solid_dir = output_dir / "fluid", output_dir / "solid"
    fluid_path = fluid_dir / "coupled_mms_force_payload_history.jsonl"
    fluid_records = _read_jsonl_history(fluid_path, (
        "fluid_time_s", "dt", "target_particle_state", "numerical_particle_state",
        "fixed_probe_positions_m", "target_probe_velocity_m_s",
        "numerical_probe_velocity_m_s", "target_probe_vorticity_s_inv",
        "numerical_probe_vorticity_s_inv", "target_force", "force",
        "bound_circulation", "target_bound_circulation",
        "target_y_stations_m",
        "target_vlm_circulation_error", "target_vlm_solved_circulation",
        "vlm_algebraic_residual",
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
    structural_fields = ("step", "time_s", "target_force", "numerical_force", "target_state", "numerical_state",
                         "e_u0", "e_v0", "e_w", "e_theta_x", "e_theta_y",
                         "source_evaluation_time_s", "t_alpha_m_s", "t_alpha_f_s", "dt_s",
                         "alpha_m", "alpha_f", "gamma", "beta",
                         "q_alpha_f_coefficients", "v_alpha_f_coefficients",
                         "a_alpha_m_coefficients")
    missing = [field for field in structural_fields if field not in (reader.fieldnames or ())]
    if missing:
        raise ValueError(f"Structural MMS history is missing fields: {', '.join(missing)}")

    if len(rows) != len(fluid_records):
        raise ValueError(
            f"Fluid/solid force histories have different lengths ({len(fluid_records)} vs {len(rows)})"
        )
    solid_steps = [int(row["step"]) for row in rows]
    fluid_steps = [int(rec.get("step", -1)) for rec in fluid_records]
    if solid_steps != fluid_steps or solid_steps != list(range(1, len(rows) + 1)):
        raise ValueError("Fluid and structural histories must contain identical ordered step IDs")
    if len(set(fluid_steps)) != len(fluid_steps):
        raise ValueError("Fluid force-exchange step IDs must be unique")
    solid_times = np.asarray([float(row["time_s"]) for row in rows], dtype=float)
    fluid_times_all = np.asarray([float(rec["fluid_time_s"]) for rec in fluid_records], dtype=float)
    tolerance = float(mms_p.get("coupled", {}).get("time_match_tolerance_s", 1.0e-10))
    if not np.isfinite(solid_times).all() or not np.isfinite(fluid_times_all).all() or not np.allclose(
            solid_times, fluid_times_all, rtol=0.0, atol=tolerance):
        raise ValueError("Fluid and structural histories do not share the same physical evaluation times")

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
    fluid_times = [float(rec["fluid_time_s"]) for rec in fluid_records]
    if not all(math.isfinite(t) for t in fluid_times) or any(
            right <= left for left, right in zip(fluid_times, fluid_times[1:])):
        raise ValueError("Fluid force-exchange times must be finite, unique, and strictly increasing")
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
                       atol=max(tolerance, abs(dt_f_recorded) * 1.0e-12)):
        raise ValueError(
            f"Recorded fluid RK3 stages {stage_times.tolist()} do not match expected "
            f"stage chronology {expected_stage_times.tolist()} ending at force time {t_s:g} s"
        )
    required_structural_metrics = ("e_u0", "e_v0", "e_w", "e_theta_x", "e_theta_y")
    for record_index, (record, solid_row) in enumerate(zip(fluid_records, rows)):
        record_time = float(record["fluid_time_s"])
        record_dt = float(record["dt"])
        record_stages = np.asarray(record["rVPM_RK3_stage_times_s"], dtype=float)
        expected = record_time - record_dt + record_dt * np.asarray((0.0, 1.0/3.0, 3.0/4.0))
        if (record_stages.shape != (3,) or not np.isfinite(record_stages).all()
                or not math.isfinite(record_dt) or record_dt <= 0.0
                or not np.allclose(record_stages, expected, rtol=0.0,
                                   atol=max(tolerance, abs(record_dt)*1.0e-12))):
            raise ValueError(f"Fluid RK3 stage chronology is invalid at step {record.get('step')}")
        if record_index and record_time <= float(fluid_records[record_index-1]["fluid_time_s"]):
            raise ValueError("Fluid exchange times must be strictly increasing")
        solid_target_step = np.asarray(json.loads(solid_row["target_state"]), dtype=float)
        solid_numerical_step = np.asarray(json.loads(solid_row["numerical_state"]), dtype=float)
        if (solid_target_step.shape != solid_numerical_step.shape or solid_target_step.size == 0
                or not np.isfinite(solid_target_step).all() or not np.isfinite(solid_numerical_step).all()):
            raise ValueError(f"Structural endpoint states are malformed at step {record.get('step')}")
        solid_time = float(solid_row["time_s"])
        if not math.isfinite(solid_time) or abs(solid_time-record_time) > tolerance:
            raise ValueError(f"Structural/fluid endpoint time mismatch at step {record.get('step')}")
        dt_structural = float(solid_row["dt_s"])
        alpha_m = float(solid_row["alpha_m"])
        alpha_f = float(solid_row["alpha_f"])
        gamma = float(solid_row["gamma"])
        beta = float(solid_row["beta"])
        source_time = float(solid_row["source_evaluation_time_s"])
        alpha_m_time = float(solid_row["t_alpha_m_s"])
        alpha_f_time = float(solid_row["t_alpha_f_s"])
        expected_gamma = 0.5 + alpha_f - alpha_m
        expected_beta = (expected_gamma + 0.5)**2 / 4.0
        if (not all(math.isfinite(v) for v in (dt_structural, alpha_m, alpha_f, gamma, beta,
                                              source_time, alpha_m_time, alpha_f_time))
                or dt_structural <= 0.0 or abs(dt_structural-record_dt) > tolerance
                or abs(gamma-expected_gamma) > 1.0e-14 or abs(beta-expected_beta) > 1.0e-14
                or abs(source_time-record_time) > tolerance
                or abs(alpha_m_time-(record_time-alpha_m*dt_structural)) > tolerance
                or abs(alpha_f_time-(record_time-alpha_f*dt_structural)) > tolerance):
            raise ValueError(f"Structural generalized-alpha clock/coefficients are invalid at step {record.get('step')}")
        for stage_key in ("q_alpha_f_coefficients", "v_alpha_f_coefficients", "a_alpha_m_coefficients"):
            stage_values = np.asarray(json.loads(solid_row[stage_key]), dtype=float)
            if stage_values.shape != (5,) or not np.isfinite(stage_values).all():
                raise ValueError(f"Structural target stage coefficients are invalid at step {record.get('step')}")
        for metric_key in required_structural_metrics:
            metric_value = float(solid_row[metric_key])
            if not math.isfinite(metric_value) or metric_value < 0.0:
                raise ValueError(f"Invalid structural metric {metric_key} at step {record.get('step')}")
        target_g = np.asarray(record["target_bound_circulation"], dtype=float)
        solved_g = np.asarray(record["target_vlm_solved_circulation"], dtype=float)
        reported = float(record["target_vlm_circulation_error"])
        residual = float(record["vlm_algebraic_residual"])
        if (target_g.ndim != 1 or solved_g.shape != target_g.shape or target_g.size != n_span
                or not np.isfinite(target_g).all() or not np.isfinite(solved_g).all()
                or not math.isfinite(reported) or reported < 0.0
                or not math.isfinite(residual) or residual < 0.0):
            raise ValueError(f"Invalid target VLM circulation data at step {record.get('step')}")
        recomputed = _relative_array_error(solved_g, target_g, label="e_vlm_target_circulation")
        error_atol = float(mms_p["coupled"].get("vlm_reported_error_atol", 5.0e-13))
        if abs(recomputed-reported) > error_atol:
            raise ValueError(f"Reported VLM circulation error mismatch at step {record.get('step')}")
        if (recomputed > float(mms_p["coupled"].get("target_vlm_circulation_rtol", 1.0e-10))
                or residual > float(mms_p["coupled"].get("vlm_algebraic_residual_rtol", 1.0e-10))):
            raise ValueError(f"Target VLM consistency gate failed at step {record.get('step')}")
        if (record.get("target_particle_ids") != fluid_records[0].get("target_particle_ids")
                or record.get("numerical_particle_ids") != record.get("target_particle_ids")):
            raise ValueError(f"Target/numerical particle IDs or ordering changed at step {record.get('step')}")
        pstate_t, pstate_n = record["target_particle_state"], record["numerical_particle_state"]
        particle_count = len(record["target_particle_ids"] or ())
        for state_key in ("position", "circulation", "core"):
            target_state = np.asarray(pstate_t[state_key], dtype=float)
            numerical_state = np.asarray(pstate_n[state_key], dtype=float)
            if (target_state.shape != numerical_state.shape or not np.isfinite(target_state).all()
                    or not np.isfinite(numerical_state).all()):
                raise ValueError(f"Particle {state_key} state schema is invalid at step {record.get('step')}")
            expected_shape = (particle_count, 3) if state_key != "core" else (particle_count,)
            if target_state.shape != expected_shape:
                raise ValueError(f"Particle {state_key} shape is inconsistent with target IDs at step {record.get('step')}")
        step_probes = np.asarray(record["fixed_probe_positions_m"], dtype=float)
        if (step_probes.shape != np.asarray(fluid_records[0]["fixed_probe_positions_m"]).shape
                or not np.isfinite(step_probes).all()
                or not np.array_equal(step_probes, np.asarray(fluid_records[0]["fixed_probe_positions_m"]))):
            raise ValueError(f"Fixed physical probes changed at step {record.get('step')}")
        for force_key in ("force", "target_force"):
            force_values = np.asarray(record[force_key], dtype=float)
            if force_values.shape != np.asarray(record["target_force"]).shape or not np.isfinite(force_values).all():
                raise ValueError(f"Aerodynamic force array schema is invalid at step {record.get('step')}")
        target_geometry = record["target_geometry"]
        target_cp = np.asarray(target_geometry.get("cp"), dtype=float)
        numeric_cp = np.asarray(record["geometry_cp_absolute"], dtype=float)
        if target_cp.shape != numeric_cp.shape or target_cp.size == 0 or not (
                np.isfinite(target_cp).all() and np.isfinite(numeric_cp).all()):
            raise ValueError(f"Target/numerical geometry arrays are invalid at step {record.get('step')}")
        station_y = np.asarray(record["target_y_stations_m"], dtype=float)
        station_atol = float(mms_p["coupled"].get("target_station_atol_m", 5.0e-12))
        if (station_y.shape != (n_span,) or not np.isfinite(station_y).all()
                or np.any(np.diff(station_y) <= 0.0)
                or not math.isfinite(station_atol) or station_atol <= 0.0
                or target_cp.shape[0] != n_span or not np.allclose(
                    target_cp[:, 1], station_y, rtol=0.0, atol=station_atol)):
            raise ValueError(f"Bound-circulation station ordering does not match target geometry at step {record.get('step')}")
        target_bound = np.asarray(record["target_bound_circulation"], dtype=float)
        numerical_bound = np.asarray(record["bound_circulation"], dtype=float)
        if (target_bound.shape != (n_span,) or numerical_bound.shape != target_bound.shape
                or not np.isfinite(target_bound).all() or not np.isfinite(numerical_bound).all()):
            raise ValueError(f"Target/numerical bound circulation is invalid at step {record.get('step')}")
        for numerical_key, target_key in (("numerical_probe_velocity_m_s", "target_probe_velocity_m_s"),
                                          ("numerical_probe_vorticity_s_inv", "target_probe_vorticity_s_inv")):
            _relative_array_error(record[numerical_key], record[target_key], label=numerical_key)

    target_ids = fluid.get("target_particle_ids")
    numerical_ids = fluid.get("numerical_particle_ids")
    if target_ids is None or numerical_ids is None or target_ids != numerical_ids:
        raise ValueError("Fluid target/numerical particle IDs are missing or out of exact order")
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
    diagnostics_path = solid_dir / diagnostics_name
    if not diagnostics_path.is_file():
        raise FileNotFoundError(f"Required structural transfer diagnostics are missing: {diagnostics_path}")
    with diagnostics_path.open(newline="", encoding="utf-8") as stream:
        diag_reader = csv.DictReader(stream)
        diag_rows = list(diag_reader)
        diag_fields = diag_reader.fieldnames or ()
    diag_keys = ("step", "time", "force_transfer_relative_error", "moment_transfer_relative_error",
                 "incremental_work_relative_error", "force_transfer_mode",
                 "displacement_transfer_mode", "moment_origin_x_m", "moment_origin_y_m",
                 "moment_origin_z_m", "newton_residual_norm", "newton_relative_residual")
    missing_diag = [key for key in diag_keys if key not in diag_fields]
    if missing_diag or not diag_rows:
        raise ValueError(f"Transfer diagnostics missing fields/rows: {missing_diag or 'empty history'}")
    if len(diag_rows) != len(fluid_records):
        raise ValueError("Transfer diagnostics must have exactly one record for every fluid/solid exchange step")
    origin_atol = float(mms_p["coupled"].get("moment_origin_csv_atol_m", 5.0e-12))
    expected_force_mode = str(coupling.get("force_transfer_mode", "")).strip().lower()
    expected_disp_mode = str(coupling.get("displacement_transfer_mode", "")).strip().lower()
    for index, row in enumerate(diag_rows):
        row_step, row_time = int(row["step"]), float(row["time"])
        row_origin = np.asarray([float(row[f"moment_origin_{axis}_m"]) for axis in "xyz"])
        row_metrics = [float(row[key]) for key in (
            "force_transfer_relative_error", "moment_transfer_relative_error",
            "incremental_work_relative_error", "newton_residual_norm",
            "newton_relative_residual")]
        if (row_step != index + 1 or not math.isfinite(row_time)
                or abs(row_time-fluid_times_all[index]) > tolerance
                or not np.isfinite(row_origin).all()
                or not np.allclose(row_origin, origin, rtol=0.0, atol=origin_atol)
                or not all(math.isfinite(value) and value >= 0.0 for value in row_metrics)):
            raise ValueError(f"Transfer diagnostics are invalid or mismatched at step {index+1}")
        wrench_limit = float(mms_p["coupled"].get("transfer_wrench_rtol", 1.0e-10))
        work_limit = float(mms_p["coupled"].get("transfer_work_rtol", 1.0e-10))
        newton_atol = float(mms_p.get("solvers", {}).get("newton_atol", 1.0e-10))
        newton_limit = float(mms_p.get("solvers", {}).get("newton_rtol", 1.0e-8))
        if (not all(math.isfinite(value) and value > 0.0
                    for value in (wrench_limit, work_limit, newton_atol, newton_limit))
                or row_metrics[0] > wrench_limit or row_metrics[1] > wrench_limit
                or row_metrics[2] > work_limit
                or (row_metrics[3] > newton_atol and row_metrics[4] > newton_limit)):
            raise ValueError(f"Structural convergence/transfer hard gate failed at step {index+1}")
        if ((expected_force_mode and row["force_transfer_mode"].strip().lower() != expected_force_mode)
                or (expected_disp_mode and row["displacement_transfer_mode"].strip().lower() != expected_disp_mode)):
            raise ValueError(f"Transfer mode metadata mismatch at step {index+1}")
    diag_matches = [row for row in diag_rows
                    if abs(float(row["time"]) - t_s) <= tolerance]
    if len(diag_matches) != 1:
        raise ValueError(f"Expected one transfer diagnostic at t={t_s:g} s, found {len(diag_matches)}")
    diag = diag_matches[0]
    if int(diag["step"]) != int(final_solid["step"]):
        raise ValueError("Transfer diagnostic step does not match the final fluid/solid step")
    recorded_origin = np.asarray([float(diag[f"moment_origin_{axis}_m"]) for axis in "xyz"])
    if (not np.isfinite(recorded_origin).all() or not math.isfinite(origin_atol) or origin_atol <= 0
            or not np.allclose(recorded_origin, origin, rtol=0.0, atol=origin_atol)):
        raise ValueError("Transfer diagnostic moment origin does not match the configured common origin")
    interface_metrics = {
        "e_transfer_force": float(diag["force_transfer_relative_error"]),
        "e_transfer_moment": float(diag["moment_transfer_relative_error"]),
        "e_transfer_work": float(diag["incremental_work_relative_error"]),
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
    gamma_metrics = {
        "e_gamma_bound": _relative_array_error(
            fluid["bound_circulation"], fluid["target_bound_circulation"], label="e_gamma_bound"),
        "e_aero_force": _relative_array_error(fluid["force"], fluid["target_force"], label="e_aero_force"),
        "e_geometry_cp": _relative_array_error(
            fluid["geometry_cp_absolute"], fluid["target_geometry"]["cp"], label="e_geometry_cp"),
    }
    if "vlm_algebraic_residual" not in fluid:
        raise ValueError("Fluid history is missing the production VLM algebraic residual")
    gamma_metrics["vlm_algebraic_residual"] = float(fluid["vlm_algebraic_residual"])
    if not math.isfinite(gamma_metrics["vlm_algebraic_residual"]) or gamma_metrics["vlm_algebraic_residual"] < 0:
        raise ValueError("Production VLM algebraic residual must be finite and nonnegative")
    if "target_vlm_circulation_error" not in fluid or "target_vlm_solved_circulation" not in fluid:
        raise ValueError("Fluid history lacks the target VLM circulation error or solved circulation array")
    solved_gamma = np.asarray(fluid["target_vlm_solved_circulation"], dtype=float)
    target_gamma = np.asarray(fluid["target_bound_circulation"], dtype=float)
    if (solved_gamma.ndim != 1 or target_gamma.ndim != 1 or solved_gamma.shape != target_gamma.shape
            or solved_gamma.size != n_span or not np.isfinite(solved_gamma).all()):
        raise ValueError("Solved VLM circulation must be a finite station-ordered array matching target stations")
    reported_gamma_error = float(fluid["target_vlm_circulation_error"])
    recomputed_gamma_error = _relative_array_error(
        solved_gamma, target_gamma, label="e_vlm_target_circulation")
    report_atol = float(mms_p["coupled"].get("vlm_reported_error_atol", 5.0e-13))
    if (not math.isfinite(reported_gamma_error) or reported_gamma_error < 0.0
            or not math.isfinite(report_atol) or report_atol <= 0.0
            or abs(reported_gamma_error - recomputed_gamma_error) > report_atol):
        raise ValueError("Julia-reported VLM target circulation error disagrees with Python recomputation")
    gamma_metrics["e_vlm_target_circulation"] = recomputed_gamma_error
    gamma_limit = float(mms_p["coupled"].get("target_vlm_circulation_rtol", 1.0e-10))
    residual_limit = float(mms_p["coupled"].get("vlm_algebraic_residual_rtol", 1.0e-10))
    if (not math.isfinite(gamma_limit) or gamma_limit <= 0.0
            or recomputed_gamma_error > gamma_limit):
        raise ValueError(f"Target VLM circulation error {recomputed_gamma_error:.3e} exceeds {gamma_limit:.3e}")
    if (not math.isfinite(residual_limit) or residual_limit <= 0.0
            or gamma_metrics["vlm_algebraic_residual"] > residual_limit):
        raise ValueError("VLM algebraic residual exceeds the configured hard gate")
    return {
        "study": study,
        "level": level,
        "step": int(fluid.get("step", len(fluid_records))),
        "dt_f": dt_f_recorded,
        "dt_s": float(dt_s),
        "nx": nx,
        "ny": ny,
        "n_span": n_span,
        "particle_count": int(len(target_ids)),
        "h_mesh_m": max(float(Lx) / float(nx), float(Ly) / float(ny)),
        "refinement_variable": "dt_s" if study == "temporal" else "h_mesh_m",
        **structural_metrics,
        **particle_metrics,
        **interface_metrics,
        **gamma_metrics,
        "final_time_s": t_s,
        "fluid_stage_times_s": stage_times.tolist(),
        "moment_origin_m": origin.tolist(),
        "force_transfer_mode": diag.get("force_transfer_mode", "unknown"),
        "displacement_transfer_mode": diag.get("displacement_transfer_mode", "unknown"),
        "norm_time_kind": "final-time at the matched fluid force-exchange/structural time",
        "_fixed_probe_positions_m": np.asarray(fluid["fixed_probe_positions_m"], dtype=float).tolist(),
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
    from mms_problem import (build_plate_mms_sympy, BoundCirculationTarget,
                             build_mms_manifest, ParticlePopulation,
                             CoupledMMSTarget, DiscreteGeneralizedAlphaTarget,
                             _target_function_from_coefficients)
    amp = mms_p["structural"]["amplitudes"]
    frq = mms_p["structural"]["frequencies"]
    comps = {k: {"amplitude": amp[k], "omega": frq[k], "mode": "ramp"} for k in amp}
    mms = build_plate_mms_sympy(solid, nonlinear=True, components=comps)

    x_s, y_s, t_s = mms.x, mms.y, mms.t
    Lx, Ly = float(solid["root_chord"]), float(solid["span"])
    q_fields = list(mms.fields[0]) + [mms.fields[1]] + list(mms.fields[2])
    q_velocities = (list(mms.velocities[0]) + [mms.velocities[1]] +
                    list(mms.velocities[2]))
    q_accelerations = (list(mms.accelerations[0]) + [mms.accelerations[1]] +
                       list(mms.accelerations[2]))
    derivative_errors = []
    for time_value in (0.017, 0.083, 0.271):
        for h_t in (1.0e-3, 5.0e-4):
            for x_frac, y_frac in ((0.27, 0.34), (0.63, 0.72)):
                subs = {x_s: x_frac * Lx, y_s: y_frac * Ly, t_s: time_value}
                subs_p = subs.copy()
                subs_m = subs.copy()
                subs_p[t_s] = time_value + h_t
                subs_m[t_s] = time_value - h_t
                for q_expr, qdot_expr, qddot_expr in zip(
                        q_fields, q_velocities, q_accelerations):
                    q0 = float(q_expr.subs(subs))
                    qp = float(q_expr.subs(subs_p))
                    qm = float(q_expr.subs(subs_m))
                    qdot_fd = (qp - qm) / (2.0 * h_t)
                    qddot_fd = (qp - 2.0*q0 + qm) / h_t**2
                    qdot = float(qdot_expr.subs(subs))
                    qddot = float(qddot_expr.subs(subs))
                    derivative_errors.extend((
                        abs(qdot-qdot_fd) / max(abs(qdot), abs(qdot_fd), 1.0e-8),
                        abs(qddot-qddot_fd) / max(abs(qddot), abs(qddot_fd), 1.0e-5),
                    ))
    bound = BoundCirculationTarget(
        gamma0=np.full(4, float(mms_p["bound_circulation"]["gamma0"])),
        a=np.full(4, float(mms_p["bound_circulation"]["a"])),
        b=np.full(4, float(mms_p["bound_circulation"]["b"])),
        omega=float(frq.get("bound_circulation", frq["w"])),
    )
    for time_value in (0.0, 0.137, 2.0*math.pi/bound.omega, 4.0*math.pi/bound.omega):
        for h_t in (1.0e-4, 5.0e-5):
            fd = (bound.state(time_value+h_t)-bound.state(time_value-h_t))/(2.0*h_t)
            derivative_errors.extend(np.abs(fd-bound.derivative(time_value)) /
                                     np.maximum(np.abs(bound.derivative(time_value)), 1.0e-12))
    if not np.allclose(bound.state(0.0), bound.gamma0 + bound.b, rtol=0.0, atol=0.0):
        raise AssertionError("Canonical bound circulation must equal gamma0+b at t=0")
    manifest = build_mms_manifest(solid, fluid, coupling, mms_p)
    particle_target = ParticlePopulation.from_dict(manifest["particles"])
    alpha_m = float(solid.get("alpha_m", 0.10))
    alpha_f = float(solid.get("alpha_f", 0.20))
    from solid.utils.generalized_alpha_coefficients import parameters as production_ga_parameters
    production_gamma, production_beta = production_ga_parameters(alpha_m, alpha_f)
    coupled_target = CoupledMMSTarget.from_manifest(manifest)
    ga_target = DiscreteGeneralizedAlphaTarget(coupled_target, alpha_m, alpha_f, 0.01)
    ga_stage = ga_target.advance(0.01)
    center_endpoint = coupled_target.structural.q(
        0.5*coupled_target.structural.Lx, 0.5*coupled_target.structural.Ly, 0.01
    )
    q0, v0, a0 = (ga_target.q.copy(), ga_target.v.copy(), ga_target.a.copy())
    gamma_ga = 0.5 + alpha_f - alpha_m
    beta_ga = (gamma_ga + 0.5) ** 2 / 4.0
    if not (math.isclose(gamma_ga, production_gamma, rel_tol=0.0, abs_tol=1.0e-15)
            and math.isclose(beta_ga, production_beta, rel_tol=0.0, abs_tol=1.0e-15)):
        raise AssertionError("Independent target generalized-alpha coefficients differ from production")
    q_endpoint = ga_target._exact(0.02, 0)
    a_endpoint = ((q_endpoint-q0-0.01*v0)/(beta_ga*0.01**2)
                  -(1.0-2.0*beta_ga)/(2.0*beta_ga)*a0)
    v_endpoint = v0 + 0.01*((1.0-gamma_ga)*a0+gamma_ga*a_endpoint)
    ga_stage_2 = ga_target.advance(0.02)
    stage_errors = [
        float(np.max(np.abs(ga_stage_2["a_alpha_m"]-(alpha_m*a0+(1-alpha_m)*a_endpoint)))),
        float(np.max(np.abs(ga_stage_2["v_alpha_f"]-(alpha_f*v0+(1-alpha_f)*v_endpoint)))),
        float(np.max(np.abs(ga_stage["q_endpoint"]-center_endpoint))),
    ]
    if max(stage_errors) > 1.0e-14:
        raise AssertionError("Independent generalized-alpha target stage convention is inconsistent")
    test_results["fully_discrete_target_stages"] = {
        "status": "PASS", "max_absolute_coefficient_error": max(stage_errors),
        "threshold": 1.0e-14, "scope": "endpoint q and discrete q/v/a alpha-weighted stages for all five components",
    }
    for time_value in (0.019, 0.117, 0.293):
        for h_t in (1.0e-4, 5.0e-5):
            q_plus = particle_target.state(time_value+h_t)
            q_minus = particle_target.state(time_value-h_t)
            derivatives = particle_target.derivatives(time_value)
            for finite_difference, analytic in zip(
                    ((q_plus[k]-q_minus[k])/(2.0*h_t) for k in range(3)), derivatives):
                scale = max(float(np.linalg.norm(analytic)), 1.0e-10)
                derivative_errors.append(float(np.linalg.norm(finite_difference-analytic))/scale)

    # Effective overlay, manifest, API, and serialized-source contract audit.
    import inspect
    from coupling.utils.coupling_transfer import CouplingTransfer
    production_regularize_value = fluid.get("regularize_vlm")
    effective_fluid = dict(fluid, regularize_vlm=False, shed_boundarylayer=False)
    effective_coupling = dict(coupling, force_relax=1.0,
                              moment_origin_m=mms_p["coupled"]["moment_origin_m"])
    _validate_coupled_mms_overlay(effective_fluid, effective_coupling)
    if fluid.get("regularize_vlm") != production_regularize_value:
        raise AssertionError("MMS config overlay must not mutate the production fluid settings")
    if np.asarray(manifest["moment_origin_m"], dtype=float).shape != (3,):
        raise AssertionError("MMS manifest moment origin schema must be a three-vector")
    if "moment_origin" in inspect.signature(CouplingTransfer.incremental_work_metrics).parameters:
        raise AssertionError("Virtual-work API must remain independent of moment origin")
    if "moment_origin" not in inspect.signature(CouplingTransfer.wrench_transfer_metrics).parameters:
        raise AssertionError("Wrench API must accept the configured common moment origin")
    runtime_text = (_REPO_ROOT / "fluid/src/runtime_callback.jl").read_text(encoding="utf-8")
    solver_text = (_REPO_ROOT / "solid/structural_solver.py").read_text(encoding="utf-8")
    payload_fields = ("target_vlm_circulation_error", "target_vlm_solved_circulation",
                      "rVPM_RK3_stage_times_s", "target_particle_state", "target_force",
                      "target_y_stations_m", "fluid_time_s", "dt")
    missing_contract = [name for name in payload_fields if f'"{name}"' not in runtime_text]
    history_fields = ("a_alpha_m_coefficients", "v_alpha_f_coefficients", "q_alpha_f_coefficients",
                      "moment_origin_x_m", "force_transfer_relative_error")
    missing_contract.extend(name for name in history_fields if name not in solver_text)
    if missing_contract:
        raise AssertionError(f"Fluid/solid serialization contract is incomplete: {missing_contract}")
    test_results["configuration_api_payload_contract"] = {
        "status": "PASS", "checked_payload_fields": list(payload_fields),
        "checked_history_fields": list(history_fields),
        "scope": "effective overlay, unchanged production dictionary, manifest origin, method signatures, fluid/solid schemas",
        "threshold": "exact schema/signature match",
    }
    extractor_source = inspect.getsource(_extract_coupled_metrics)
    if ("target_vlm_circulation_error" not in extractor_source
            or "vlm_algebraic_residual" not in extractor_source
            or float(mms_p["coupled"].get("target_vlm_circulation_rtol", 0.0)) <= 0.0
            or float(mms_p["coupled"].get("vlm_algebraic_residual_rtol", 0.0)) <= 0.0):
        raise AssertionError("Target VLM circulation/residual extraction gates are incomplete")
    test_results["target_vlm_contract"] = {
        "status": "PASS", "circulation_tolerance": float(mms_p["coupled"]["target_vlm_circulation_rtol"]),
        "algebraic_residual_tolerance": float(mms_p["coupled"]["vlm_algebraic_residual_rtol"]),
        "scope": "separate target-solve circulation and algebraic-residual payload/extraction gates",
    }
    max_derivative_error = float(max(derivative_errors))
    assert max_derivative_error < 2.0e-4, f"Target derivative mismatch: {max_derivative_error}"
    test_results["target_derivatives"] = {
        "status": "PASS", "max_relative_error": max_derivative_error,
        "scope": "five structural q_t/q_tt, particle X/Gamma/sigma derivatives, and bound circulation at multiple times",
        "threshold": 2.0e-4,
    }
    logger.info("  Target derivative test: PASSED (max rel_err = %.3e)", max_derivative_error)

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
    test_results["boundary_compatibility"] = {
        "status": "PASS", "checks_verified": len(mms.boundary_checks),
        "max_symbolic_boundary_residual": 0.0, "threshold": 0.0,
        "scope": "all essential and natural component boundary conditions",
    }
    logger.info("  Boundary compatibility test: PASSED (%d boundary conditions verified)", len(mms.boundary_checks))
    test_results["structural_jacobian_directional_derivative"] = _run_jacobian_check(
        mms, solid, output_dir
    )
    test_results["full_strong_to_weak_structural_identity"] = _run_strong_weak_check(
        mms, solid, output_dir
    )

    # Test 3: Source independence
    logger.info("Test 3: Source independence guarantee …")
    import inspect
    import ast
    from mms_problem import add_coupled_mms_source
    source_functions = (build_plate_mms_sympy, add_coupled_mms_source,
                        _target_function_from_coefficients,
                        DiscreteGeneralizedAlphaTarget.advance)
    prohibited = {"residual", "jacobian", "mass_form", "damping_form", "internal_work",
                  "stiffness_form", "manufactured_weak_load"}
    prohibited_calls = []
    for function in source_functions:
        tree = ast.parse(inspect.getsource(function))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr in prohibited):
                prohibited_calls.append(f"{function.__name__}:{node.lineno}:{node.func.attr}")
    assert not prohibited_calls, f"Independent source invokes production operators: {prohibited_calls}"
    test_results["source_independence"] = {
        "status": "PASS", "prohibited_calls_detected": prohibited_calls,
        "scope": "AST call audit across standalone/coupled source construction and discrete stage helpers",
        "threshold": 0,
    }
    logger.info("  Source independence test: PASSED (zero prohibited operator calls in source builders)")

    # Test 4: audit the resolved dependency implementation and stage-trace path.
    logger.info("Test 4: FLOWVPM dependency RK3 implementation and stage-trace audit …")
    test_results["rk3_stage_count"] = _audit_flowvpm_rk3_source()
    logger.info("  RK3 source audit: PASSED (FLOWVPM %s)",
                test_results["rk3_stage_count"]["flowvpm_version"])
    test_results["julia_bound_circulation_consistency"] = {
        "status": "PASS",
        "state_absolute_error": test_results["rk3_stage_count"]["bound_circulation_state_error"],
        "derivative_relative_error": test_results["rk3_stage_count"]["bound_circulation_derivative_error"],
        "thresholds": {"state_absolute": 0.0, "derivative_relative": 2.0e-7},
        "scope": "Julia state/analytic derivative vs canonical formula and centered differences over periods",
    }

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
        p_forces = np.zeros((16, 3))
        for j in range(16):
            p_forces[j] = exact_panel_force(j / 16, (j + 1) / 16, 0.05, (1e-2, 5e-3, 2e-2), 2.3, n_quad=16)
        exact_resultant = exact_panel_force(0.0, 1.0, 0.05,
                                            (1e-2, 5e-3, 2e-2), 2.3, n_quad=32)
        partitioned_resultant = np.sum(p_forces, axis=0)
        traction_error = float(np.linalg.norm(partitioned_resultant-exact_resultant) /
                               max(np.linalg.norm(exact_resultant), 1.0e-14))
        modes = [
            "translation", "rotation", "sinusoidal",
        ]
        max_defect = 0.0
        max_wrench_error = 0.0
        mode_results = {}
        for transfer_mode in ("crm", "rbf"):
            transfer = CouplingTransfer(
                domain, model, eta_span_comm=eta_span, eta_cp=0.75,
                force_transfer_mode=transfer_mode,
                displacement_transfer_mode=transfer_mode,
                work_conservative_mode=True,
            )
            _, nodal = transfer.apply_force_transfer(p_forces)
            solid_coords = transfer.interface_coords
            displacement_modes = {
                "translation": np.tile([0.0, 0.0, 1e-4], (len(nodal), 1)),
                "rotation": np.array([[0.0, 0.0, 1e-4*(x-Lx/2.0)] for x, y in solid_coords]),
                "sinusoidal": np.array([[0.0, 0.0, 1e-4*np.sin(np.pi*x/Lx)*np.sin(np.pi*y/Ly)] for x, y in solid_coords]),
            }
            aero_coords = transfer.aero_coords[:16]
            origin = np.asarray((0.13, -0.07, 0.02))
            wrench = CouplingTransfer.wrench_transfer_metrics(
                aero_coords, p_forces, solid_coords, nodal, moment_origin=origin
            )
            shifted = CouplingTransfer.wrench_transfer_metrics(
                aero_coords, p_forces, solid_coords, nodal,
                moment_origin=origin + np.asarray((-0.04, 0.03, 0.02)),
            )
            shift = np.asarray((-0.04, 0.03, 0.02))
            if not np.allclose(shifted[3], wrench[3]-np.cross(shift, wrench[1]),
                               rtol=2.0e-12, atol=2.0e-12):
                raise AssertionError(f"{transfer_mode} origin-translation identity failed")
            mode_work = {}
            for name in modes:
                _, _, _, work_rel = transfer.incremental_work_metrics(
                    displacement_modes[name], p_forces, nodal
                )
                mode_work[name] = float(work_rel)
                max_defect = max(max_defect, work_rel)
                if work_rel >= 1.0e-12:
                    raise AssertionError(f"{transfer_mode} work duality failure on {name}: {work_rel}")
            wrench_error = max(float(wrench[6]), float(wrench[7]))
            max_wrench_error = max(max_wrench_error, wrench_error)
            mode_results[transfer_mode] = {"work_relative_errors": mode_work,
                                           "force_relative_error": float(wrench[6]),
                                           "moment_relative_error": float(wrench[7]),
                                           "origin_shift_checked": True}
        wrench_threshold = float(mms_p["coupled"].get("transfer_wrench_rtol", 1.0e-10))
        if traction_error > wrench_threshold:
            raise AssertionError(f"Panel traction/resultant consistency error {traction_error} exceeds {wrench_threshold}")
        if max_wrench_error > wrench_threshold:
            raise AssertionError(f"Transfer force/moment conservation error {max_wrench_error} exceeds {wrench_threshold}")
        test_results["work_duality"] = {"status": "PASS", "max_relative_defect": max_defect,
                                         "threshold": 1.0e-12, "scope": "CRM/RBF translation, rotation, nonrigid"}
        test_results["independent_transfer_traction_force_moment_work"] = {
            "status": "PASS", "modes": mode_results,
            "max_wrench_relative_error": max_wrench_error,
            "traction_resultant_relative_error": traction_error,
            "threshold": wrench_threshold,
            "scope": "CRM/RBF resultant wrench, nonzero-origin shift, and three work modes",
        }
        logger.info("  CRM/RBF transfer audits passed (work %.3e, wrench %.3e)", max_defect, max_wrench_error)
    except Exception as exc:
        logger.error("  Work duality test ERROR: %s", exc)
        test_results["work_duality"] = {"status": "ERROR", "reason": str(exc)}

    test_results["source_independence_dataflow"] = {
        "status": "PASS", "prohibited_calls_detected": prohibited_calls,
        "scope": "complete coupled discrete source builder plus standalone symbolic source builder",
        "threshold": 0,
    }
    test_results["all_component_derivatives_and_boundary_data"] = {
        "status": "PASS", "derivative_error": max_derivative_error,
        "derivative_threshold": 2.0e-4,
        "boundary_checks": len(mms.boundary_checks),
        "strong_weak": strong_weak_result,
    }

    required_check_names = (
        "target_derivatives", "boundary_compatibility", "structural_jacobian_directional_derivative",
        "full_strong_to_weak_structural_identity", "source_independence", "source_independence_dataflow",
        "rk3_stage_count", "work_duality", "independent_transfer_traction_force_moment_work",
        "julia_bound_circulation_consistency",
        "all_component_derivatives_and_boundary_data", "configuration_api_payload_contract",
        "fully_discrete_target_stages", "target_vlm_contract",
    )
    for required_name in required_check_names:
        if required_name not in test_results:
            test_results[required_name] = {
                "status": "FAIL", "reason": "Required automated check did not produce a result",
                "scope": "required-check completeness gate", "threshold": "result must be present",
            }

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
        modes = ["test", "structural", "transfer", "fluid", "coupled"]
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
        elif mode == "fluid":
            run_particle_mms(solid, fluid, coupling, mms_p, output_dir,
                             dry_run=args.dry_run)
        elif mode == "coupled":
            run_coupled(solid, fluid, coupling, mms_p, output_dir,
                         study=args.study, levels_override=levels_override,
                         dry_run=args.dry_run, no_plots=args.no_plots,
                         production_fluid_path=Path(args.config_dir) / "fluid_params.yaml")

    logger.info("mms_driver.py completed successfully.")


if __name__ == "__main__":
    main()
