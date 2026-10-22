#!/usr/bin/env python3
"""Instrumented independent plate strong/weak diagnostics.

``continuous`` compares the independent analytic strong load with production
weak operators evaluated on exact, mesh-attached UFL fields. ``production``
repeats the comparison using a CG2 projection of the same fields. Neither mode
is part of the primary plate MMS driver.
"""

import argparse
import csv
import json
import os
import platform
import signal
import sys
import traceback
from pathlib import Path
from time import perf_counter

import yaml

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "solid"))

from progress import log, operation
from plate import make_plate_mms

try:
    import dolfin
    from dolfin import (Constant, Expression, FunctionSpace, VectorFunctionSpace,
                        SpatialCoordinate, assemble, as_vector, cos, dot, dx,
                        interpolate, pi, sin)
except ImportError as exc:
    raise SystemExit(f"Strong/weak plate check requires legacy DOLFIN: {exc}") from exc

from utils import ReissnerMindlinModel, SolidDomain
from fenics_plate import scalar_expr, vector_expr

TESTS = (
    ("test_a", "(1+x[0]/Lx)*sin(pi*x[1]/(2*Ly))", "(1-x[0]/Lx)*sin(pi*x[1]/(2*Ly))"),
    ("test_b", "sin(2*pi*x[0]/Lx)*sin(pi*x[1]/Ly)", "cos(pi*x[0]/Lx)*sin(pi*x[1]/Ly)"),
    ("test_c", "pow(x[0]/Lx,2)*sin(3*pi*x[1]/(2*Ly))", "sin(pi*x[0]/Lx)*sin(3*pi*x[1]/(2*Ly))"),
)
THRESHOLD = 2.0e-5


class DiagnosticTimeout(TimeoutError):
    pass


def timeout_handler(_signum, _frame):
    raise DiagnosticTimeout("strong/weak diagnostic exceeded --timeout-seconds")


def resolution_arg(value):
    try:
        nx, ny = (int(part) for part in value.lower().split("x", 1))
    except (ValueError, TypeError):
        raise argparse.ArgumentTypeError("resolution must look like 12x24")
    if nx < 2 or ny < 2:
        raise argparse.ArgumentTypeError("mesh dimensions must each be >= 2")
    return nx, ny


def save_results(path, rows, summary):
    path = Path(path)
    with operation("write_incremental_csv_json", heartbeat_s=0, detail=f"rows={len(rows)} path={path}"):
        path.parent.mkdir(parents=True, exist_ok=True)
        columns = list(dict.fromkeys(k for row in rows for k in row)) or ["status"]
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        path.with_suffix(".json").write_text(json.dumps(summary, indent=2))


def validate_config(cfg):
    required_positive = ("span", "root_chord", "tip_chord", "thickness_ratio", "E", "rho_s", "kappa_shear")
    missing = [key for key in required_positive if key not in cfg]
    if missing:
        raise ValueError(f"Missing solid configuration keys: {missing}")
    invalid = [key for key in required_positive if float(cfg[key]) <= 0.0]
    if invalid:
        raise ValueError(f"Configuration values must be positive: {invalid}")
    if not -1.0 < float(cfg["nu"]) < 0.5:
        raise ValueError("Poisson ratio must be in (-1, 0.5)")


def state_projection(source_fields, time, spaces):
    return (
        interpolate(vector_expr(source_fields[0], time), spaces[0]),
        interpolate(scalar_expr(source_fields[1], time), spaces[1]),
        interpolate(vector_expr(source_fields[2], time), spaces[2]),
    )


def analytic_ufl_state(mesh, config, time, derivative=0):
    """Return exact manufactured fields as mesh-attached UFL expressions."""
    x = SpatialCoordinate(mesh)
    bump = sin(pi * x[0] / config["root_chord"]) ** 4 * sin(pi * x[1] / config["span"]) ** 4
    omega = 2.3
    phase = omega * time
    amp_u, amp_w = 1.0e-4, 1.0e-3
    if derivative == 0:
        c, s = cos(phase), sin(phase)
    elif derivative == 1:
        c, s = -omega * sin(phase), omega * cos(phase)
    else:
        c, s = -omega**2 * cos(phase), -omega**2 * sin(phase)
    return (
        as_vector((amp_u * bump * c, amp_u * bump * s)),
        amp_w * bump * s,
        as_vector((amp_w * bump * c, -amp_w * bump * s)),
    )


def run(args):
    started = perf_counter()
    log(f"STARTUP | argv={sys.argv!r} | python={sys.version.split()[0]} | executable={sys.executable}")
    log(f"ENVIRONMENT | platform={platform.platform()} | dolfin={dolfin.__version__} | output={args.output}")
    log(f"CONFIGURATION | config_dir={args.config_dir} | heartbeat_s={args.heartbeat_seconds} "
        f"timeout_s={args.timeout_seconds} | detailed={args.detailed}")
    if args.detailed:
        log(f"DETAILED SETTINGS | threshold={THRESHOLD} | quadrature_degree=24 | tests={[name for name, *_ in TESTS]}")
    if args.disable_strong_weak_diagnostic:
        summary = {"status": "NOT RUN", "reason": "disabled by command-line option", "checks": []}
        save_results(args.output, [], summary)
        log(f"FINAL | status=NOT RUN | elapsed={perf_counter()-started:.3f}s | output={args.output}")
        return summary

    with operation("load_and_validate_configuration", args.heartbeat_seconds,
                   detail=f"config={args.config_dir}"):
        config_path = Path(args.config_dir) / "solid_params.yaml"
        cfg = yaml.safe_load(config_path.read_text())
        validate_config(cfg)
    formulations = ("linear", "nonlinear") if args.formulation == "both" else (args.formulation,)
    modes = ("continuous", "production") if args.mode == "both" else (args.mode,)
    if args.quick:
        resolutions = [args.quick_mesh]
        log(f"QUICK DIAGNOSTIC | resolution={resolutions[0]} | not a final verification")
    else:
        resolutions = args.resolution or [(12, 24)]
    rows = []
    first_failure = None
    summary = {"status": "RUNNING", "config": str(config_path.resolve()), "checks": rows,
               "modes": modes, "formulations": formulations, "threshold": THRESHOLD,
               "heartbeat_seconds": args.heartbeat_seconds}
    save_results(args.output, rows, summary)

    for formulation in formulations:
        nonlinear = formulation == "nonlinear"
        detail = f"formulation={formulation}"
        with operation("construct_manufactured_solution", args.heartbeat_seconds, detail=detail):
            mms = make_plate_mms(
                cfg, nonlinear, cfg.get("include_quadratic_inplane_strain", True),
                operation_factory=lambda stage, **kw: operation(
                    stage, args.heartbeat_seconds, detail=detail
                ),
            )
        with operation("evaluate_symbolic_boundary_conditions", args.heartbeat_seconds, detail=detail):
            boundary_values = [value for group in mms.boundary_checks.values()
                               for value in (group if isinstance(group, list) else [group])]
            nonzero_boundary = [str(value) for value in boundary_values if value != 0]
            if nonzero_boundary:
                raise AssertionError(f"nonzero manufactured boundary data: {nonzero_boundary}")
        with operation("compile_independent_strong_load_expressions", args.heartbeat_seconds,
                       detail=detail):
            strong_coefficients = {}
            for component, source_fields in mms.load_components.items():
                strong_coefficients[component] = (
                    vector_expr(source_fields[0], 0.37, degree=8),
                    scalar_expr(source_fields[1], 0.37, degree=8),
                    vector_expr(source_fields[2], 0.37, degree=8),
                )
        fields = mms.fields
        velocity = tuple(v.diff(mms.t) for v in fields)
        acceleration = tuple(v.diff(mms.t, 2) for v in fields)

        for mode in modes:
            for nx, ny in resolutions:
                case_detail = f"mode={mode} formulation={formulation} mesh={nx}x{ny} time=0.37"
                # CG2 is the production representation. The continuous-mode
                # projection is deliberately higher order, but remains a
                # numerical approximation and is labelled as such in output.
                projection_degree = 4 if mode == "continuous" else 2
                with operation("construct_production_model_and_ufl_forms", args.heartbeat_seconds,
                               detail=case_detail):
                    domain = SolidDomain(
                        cfg["span"], cfg["root_chord"], cfg["tip_chord"], cfg["thickness_ratio"],
                        cfg.get("leading_edge_sweep", 0.0), nx, ny, min(ny, 40), "midpoint",
                    )
                    model = ReissnerMindlinModel(
                        domain, cfg["E"], cfg["nu"], cfg["rho_s"], cfg["eta_m"], cfg["eta_k"],
                        cfg["kappa_shear"], Constant(0.001), cfg.get("alpha_m", 0.1),
                        cfg.get("alpha_f", 0.2), geometric_nonlinearity=nonlinear,
                        include_quadratic_inplane_strain=cfg.get("include_quadratic_inplane_strain", True),
                        quadrature_degree=24,
                    )
                    ref_spaces = (
                        VectorFunctionSpace(domain.mesh, "CG", projection_degree, dim=2),
                        FunctionSpace(domain.mesh, "CG", projection_degree),
                        VectorFunctionSpace(domain.mesh, "CG", projection_degree, dim=2),
                    )
                    test_spaces = [model.V.sub(i).collapse() for i in range(3)]
                if mode == "continuous":
                    with operation("evaluate_exact_mesh_attached_UFL_displacement_velocity_acceleration",
                                   args.heartbeat_seconds, detail=case_detail):
                        q_state = analytic_ufl_state(domain.mesh, cfg, 0.37, 0)
                        v_state = analytic_ufl_state(domain.mesh, cfg, 0.37, 1)
                        a_state = analytic_ufl_state(domain.mesh, cfg, 0.37, 2)
                else:
                    with operation("interpolate_exact_displacement_velocity_acceleration",
                                   args.heartbeat_seconds,
                                   detail=f"{case_detail} projection=CG{projection_degree}"):
                        q_state = state_projection(fields, 0.37, ref_spaces)
                        v_state = state_projection(velocity, 0.37, ref_spaces)
                        a_state = state_projection(acceleration, 0.37, ref_spaces)

                weak_assemblies = 0
                for test_name, shape_x, shape_y in TESTS:
                    test_detail = f"{case_detail} test={test_name}"
                    with operation("interpolate_virtual_test", args.heartbeat_seconds, detail=test_detail):
                        test_u = Expression((f"1e-4*({shape_x})", f"1e-4*({shape_y})"), degree=8,
                                            Lx=cfg["root_chord"], Ly=cfg["span"])
                        test_w = Expression(f"1e-3*({shape_x})", degree=8,
                                            Lx=cfg["root_chord"], Ly=cfg["span"])
                        test_theta = Expression((f"1e-3*({shape_y})", f"-1e-3*({shape_x})"), degree=8,
                                                Lx=cfg["root_chord"], Ly=cfg["span"])
                        test = (interpolate(test_u, test_spaces[0]), interpolate(test_w, test_spaces[1]),
                                interpolate(test_theta, test_spaces[2]))
                    with operation("construct_production_weak_form", args.heartbeat_seconds,
                                   detail=test_detail):
                        weak_components = {
                            "inertia": model.mass_form(a_state, test),
                            "mass_damping": model.eta_m * model.mass_form(v_state, test),
                            "reference_stiffness_damping": model.eta_k * model.stiffness_form(v_state, test),
                        }
                        internal_parts = model.internal_work_components(q_state, test)
                        membrane_label = "nonlinear_membrane" if nonlinear else "linear_membrane"
                        weak_components[membrane_label] = internal_parts["membrane"]
                        weak_components["bending"] = internal_parts["bending"]
                        weak_components["shear"] = internal_parts["shear"]
                    with operation("construct_independent_strong_component_forms", args.heartbeat_seconds,
                                   detail=test_detail):
                        strong_components = {}
                        for component, force_fields in strong_coefficients.items():
                            label = ("nonlinear_membrane" if component == "membrane" and nonlinear else
                                     "linear_membrane" if component == "membrane" else component)
                            strong_components[label] = (
                                dot(force_fields[0], test[0])
                                + force_fields[1] * test[1]
                                + dot(force_fields[2], test[2])
                            ) * model.dx

                    test_start = perf_counter()
                    weak_values, strong_values = {}, {}
                    first_weak = weak_assemblies == 0
                    for component, weak_form in weak_components.items():
                        stage = (f"first_weak_assembly_JIT_included_{component}" if first_weak
                                 else f"subsequent_weak_assembly_{component}")
                        with operation(stage, args.heartbeat_seconds,
                                       detail=test_detail):
                            weak_values[component] = float(assemble(weak_form))
                        first_weak = False
                    weak_assemblies += 1
                    for component, strong_form in strong_components.items():
                        with operation(f"assemble_independent_strong_{component}", args.heartbeat_seconds,
                                       detail=test_detail):
                            strong_values[component] = float(assemble(strong_form))

                    comparisons = []
                    for component in weak_components:
                        weak_value = weak_values[component]
                        strong_value = strong_values[component]
                        discrepancy = abs(weak_value - strong_value)
                        total_scale = sum(abs(value) for value in weak_values.values()) + sum(
                            abs(value) for value in strong_values.values()
                        )
                        # Near-zero components are dominated by quadrature and
                        # floating-point noise. Use a small fraction of the
                        # total force-work scale as a floor, while retaining
                        # component-relative normalization for meaningful terms.
                        scale = max(abs(weak_value), abs(strong_value), total_scale * 1.0e-8, 1.0e-20)
                        relative = discrepancy / scale
                        comparisons.append({
                            "mode": mode, "formulation": formulation, "test": test_name,
                            "component": component, "nx": nx, "ny": ny,
                            "projection_space": "analytic UFL" if mode == "continuous" else f"CG{projection_degree}", "physical_time_s": 0.37,
                            "production_weak_value": weak_value, "independent_strong_value": strong_value,
                            "signed_difference": weak_value - strong_value,
                            "absolute_discrepancy": discrepancy, "normalization_scale": scale,
                            "relative_discrepancy": relative, "threshold": THRESHOLD,
                            "exceeds_threshold": relative > THRESHOLD,
                            "elapsed_s": perf_counter() - test_start,
                            "status": ("PASS" if relative <= THRESHOLD else "FAIL")
                            if mode == "continuous" else ("WITHIN_THRESHOLD" if relative <= THRESHOLD else "EXCEEDS_THRESHOLD"),
                        })
                    weak_total = sum(weak_values.values())
                    strong_total = sum(strong_values.values())
                    total_error = abs(weak_total - strong_total)
                    total_scale = max(abs(weak_total), abs(strong_total), 1.0e-20)
                    comparisons.append({
                        "mode": mode, "formulation": formulation, "test": test_name,
                        "component": "total", "nx": nx, "ny": ny,
                        "projection_space": "analytic UFL" if mode == "continuous" else f"CG{projection_degree}", "physical_time_s": 0.37,
                        "production_weak_value": weak_total, "independent_strong_value": strong_total,
                        "signed_difference": weak_total - strong_total,
                        "absolute_discrepancy": total_error, "normalization_scale": total_scale,
                        "relative_discrepancy": total_error / total_scale, "threshold": THRESHOLD,
                        "exceeds_threshold": total_error / total_scale > THRESHOLD,
                        "elapsed_s": perf_counter() - test_start,
                        "status": ("PASS" if total_error / total_scale <= THRESHOLD else "FAIL")
                        if mode == "continuous" else ("WITHIN_THRESHOLD" if total_error / total_scale <= THRESHOLD else "EXCEEDS_THRESHOLD"),
                    })
                    for row in comparisons:
                        rows.append(row)
                        log(f"COMPARE | {case_detail} test={test_name} component={row['component']} "
                            f"weak={row['production_weak_value']:.9e} strong={row['independent_strong_value']:.9e} "
                            f"signed={row['signed_difference']:.3e} abs={row['absolute_discrepancy']:.3e} "
                            f"rel={row['relative_discrepancy']:.3e} scale={row['normalization_scale']:.3e} "
                            f"elapsed={row['elapsed_s']:.3f}s status={row['status']}")
                        save_results(args.output, rows, summary)
                    if args.fail_after == f"{formulation}:{test_name}":
                        first_failure = {"formulation": formulation, "stage": "injected_diagnostic_failure",
                                         "test": test_name, "component": "total"}
                        summary.update(status="FAIL", first_failure=first_failure)
                        save_results(args.output, rows, summary)
                        log(f"FIRST FAILURE | {first_failure}")
                        raise RuntimeError(f"deliberately injected failure at {first_failure}")
                    for row in comparisons:
                        if mode == "continuous" and row["status"] == "FAIL":
                            first_failure = {"formulation": formulation, "stage": "strong_weak_comparison",
                                             "test": test_name, "component": row["component"]}
                            summary.update(status="FAIL", first_failure=first_failure)
                            save_results(args.output, rows, summary)
                            log(f"FIRST FAILURE | {first_failure}")
                            raise AssertionError(f"strong/weak mismatch: {first_failure}, row={row}")
    identity_status = "PASS" if "continuous" in modes else "NOT RUN"
    summary.update(
        status=identity_status,
        first_failure=None,
        production_interpretation=(
            "projection discrepancies are recorded per row and are not treated as formulation failures"
            if "production" in modes else None
        ),
        continuous_identity=("passed" if "continuous" in modes else "not requested"),
    )
    save_results(args.output, rows, summary)
    log(f"FINAL | status={identity_status} | checks={len(rows)} | "
        f"elapsed={perf_counter()-started:.3f}s | output={args.output}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", default=str(ROOT / "config"))
    parser.add_argument("--output", default=str(HERE / "results" / "plate_strong_weak.csv"))
    parser.add_argument("--formulation", choices=("linear", "nonlinear", "both"), default="both")
    parser.add_argument("--mode", choices=("continuous", "production", "both"), default="both")
    parser.add_argument("--resolution", type=resolution_arg, action="append",
                        help="mesh as NXxNY; repeat to test representation refinement")
    parser.add_argument("--quick", action="store_true", help="run a reduced diagnostic mesh only")
    parser.add_argument("--quick-mesh", type=resolution_arg, default=(4, 8))
    parser.add_argument("--heartbeat-seconds", type=float, default=10.0)
    parser.add_argument("--timeout-seconds", type=int, default=0,
                        help="optional wall-clock timeout for the diagnostic; 0 disables it")
    parser.add_argument("--detailed", action="store_true", help="log threshold, quadrature, and test settings")
    parser.add_argument("--disable-strong-weak-diagnostic", action="store_true")
    parser.add_argument("--fail-after", metavar="FORMULATION:TEST",
                        help="inject a diagnostic failure to verify failure reporting")
    args = parser.parse_args()
    started = perf_counter()
    if args.timeout_seconds > 0:
        if not hasattr(signal, "SIGALRM"):
            parser.error("--timeout-seconds is supported only on platforms with SIGALRM")
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(args.timeout_seconds)
    try:
        run(args)
    except BaseException:
        error_text = traceback.format_exc()
        output = Path(args.output)
        summary_path = output.with_suffix(".json")
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text())
            except (OSError, json.JSONDecodeError):
                summary = {"checks": []}
        else:
            summary = {"checks": []}
        final_status = "TIMED OUT" if isinstance(sys.exc_info()[1], DiagnosticTimeout) else "FAIL"
        summary.update(status=final_status, traceback=error_text,
                       elapsed_s=perf_counter()-started)
        if "first_failure" not in summary:
            summary["first_failure"] = {
                "stage": "timeout" if final_status == "TIMED OUT" else "see_traceback",
                "formulation": None, "test": None,
            }
        output.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2))
        log(f"FINAL | status={final_status} | elapsed={perf_counter()-started:.3f}s "
            f"| first_failure={summary['first_failure']}")
        log(f"TRACEBACK | elapsed={perf_counter()-started:.3f}s | outputs_preserved={output},{summary_path}\n{error_text}")
        raise
    finally:
        if args.timeout_seconds > 0 and hasattr(signal, "SIGALRM"):
            signal.alarm(0)


if __name__ == "__main__":
    main()
