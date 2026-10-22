#!/usr/bin/env python3
"""Run analytic identities and component checks, without a PDE solve.

This command does NOT refine meshes or time steps and does NOT launch
``run.sh``. It checks symbolic fluid/plate identities, the nonlinear membrane
variation, and an algebraic force-transfer invariant. Julia's FLOWVPM/FLOWVLM
checks are a separate command; their existing CSV is included in this report
if present. Use ``refinement.py --execute`` for actual solid convergence.
"""

import argparse
import csv
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import yaml

from fluid import taylor_green_source
from plate import make_plate_mms
from scalar_generalized_alpha import run_scalar_test
from progress import log, operation

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "results"


def verify_taylor_green():
    data = taylor_green_source()
    residuals = [data[k] for k in ("momentum_source", "vorticity_source")]
    maximum = max(float(abs(sp.N(v.subs({data["coords"][i]: q for i, q in enumerate((0.31, 0.47, 0.0, 0.23))})))) for expr in residuals for v in expr)
    curl_error = max(float(abs(sp.N(v.subs({data["coords"][i]: q for i, q in enumerate((0.31, 0.47, 0.0, 0.23))})))) for v in data["curl_residual"])
    div = sp.simplify(data["divergence"])
    if maximum > 1e-10 or curl_error > 1e-10 or div != 0:
        raise AssertionError(f"Taylor-Green identity failed: residual={maximum}, curl={curl_error}, div={div}")
    return {"status": "PASS", "momentum_source_max_abs": maximum,
            "vorticity_source_max_abs": maximum, "curl_definition_max_abs": curl_error,
            "divergence": str(div)}


def verify_nonlinear_membrane_variation():
    x, y, epsilon = sp.symbols("x y epsilon", real=True)
    u = sp.Matrix([x**2 + x*y, y**2 - x])
    du = sp.Matrix([x*y, x**2])
    w, dw = x**2*y, x*y**2

    def strain(um, wm):
        gu = um.jacobian((x, y))
        gw = sp.Matrix([sp.diff(wm, x), sp.diff(wm, y)])
        return (gu + gu.T)/2 + (gu.T*gu + gw*gw.T)/2

    exact = sp.diff(strain(u + epsilon*du, w + epsilon*dw), epsilon).subs(epsilon, 0)
    gu = u.jacobian((x, y))
    gdu = du.jacobian((x, y))
    gw = sp.Matrix([sp.diff(w, x), sp.diff(w, y)])
    gdw = sp.Matrix([sp.diff(dw, x), sp.diff(dw, y)])
    variation = (gdu + gdu.T)/2 + (gu.T*gdu + gdu.T*gu)/2 + (gw*gdw.T + gdw*gw.T)/2
    discrepancy = sp.simplify(exact - variation)
    if discrepancy != sp.zeros(2):
        raise AssertionError(f"Nonlinear membrane variation mismatch: {discrepancy}")
    return {"status": "PASS", "directional_derivative_max_abs": 0.0}


def verify_plate_symbolics(config, heartbeat_s=10.0):
    result = {}
    for label, nonlinear in (("linear", False), ("nonlinear", True)):
        mms = make_plate_mms(
            config, nonlinear, config.get("include_quadratic_inplane_strain", True),
            operation_factory=lambda stage, **kw: operation(
                stage, heartbeat_s, detail=f"formulation={label}"
            ),
        )
        check_values = []
        for group in mms.boundary_checks.values():
            values = group if isinstance(group, list) else [group]
            check_values.extend(float(sp.N(v)) for v in values)
        if any(abs(v) > 1e-11 for v in check_values):
            raise AssertionError(f"{label} plate MMS violates edge conditions: {check_values}")
        substitution = {mms.x: config["root_chord"] * 0.41, mms.y: config["span"] * 0.57, mms.t: 0.37}
        sampled = []
        for group in mms.loads:
            expressions = list(group) if isinstance(group, sp.MatrixBase) else [group]
            sampled.extend(float(sp.N(expr.subs(substitution))) for expr in expressions)
        if not np.isfinite(sampled).all() or max(abs(v) for v in sampled) == 0:
            raise AssertionError(f"{label} plate MMS source is invalid")
        result[label] = {
            "status": "PASS",
            "edge_condition_max_abs": max((abs(v) for v in check_values), default=0.0),
            "source_sample_max_abs": max(abs(v) for v in sampled),
            "source_components": len(sampled),
            "source_derivation": "strong equilibrium from independently assembled membrane, bending, shear, inertia, and Rayleigh damping laws",
        }
        if label == "nonlinear":
            X = np.linspace(0, config["root_chord"], 80)
            Y = np.linspace(0, config["span"], 160)
            xx, yy = np.meshgrid(X, Y)
            w_expr = mms.fields[1]
            w_func = sp.lambdify((mms.x, mms.y, mms.t), w_expr, "numpy")
            values = w_func(xx, yy, 0.37)
            fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
            image = ax.pcolormesh(xx, yy, values * 1e3, shading="auto", cmap="viridis")
            ax.set(xlabel="Chordwise x [m]", ylabel="Spanwise y [m]", title="Analytical manufactured transverse displacement")
            fig.colorbar(image, ax=ax, label="w* [mm]")
            fig.savefig(OUT / "plate_exact_displacement.png", dpi=180)
            plt.close(fig)
    return result


def verify_rbf_component(heartbeat_s=10.0):
    module_path = ROOT / "coupling" / "utils" / "rbf_transfer.py"
    spec = importlib.util.spec_from_file_location("verification_rbf_transfer", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    RBFTransfer = module.RBFTransfer

    ns, nf = 81, 21
    y_s = np.linspace(0, 1, ns)
    y_f = np.linspace(0.025, 0.975, nf)
    solid = np.column_stack((np.zeros(ns), y_s))
    fluid = np.column_stack((np.zeros(nf), y_f))
    with operation("construct_RBF_transfer_map", heartbeat_s,
                   detail=f"fluid={nf} solid={ns} radius=0.12 neighbors=12"):
        nbr, weights = RBFTransfer.build_local_rbf_map(fluid, solid, 0.12, n_neighbors=12)
    A = np.full(nf, 1.0 / nf)
    lumped = RBFTransfer.compute_S_lumped(ns, nbr, weights, A)
    forces = (
        np.column_stack((np.ones(nf) * 2.0, np.ones(nf) * -3.0, np.ones(nf) * 0.5)),
        np.column_stack((1.0 + 2*y_f, -3.0 + 0.5*y_f, 0.25 - y_f)),
    )
    force_results = []
    for force in forces:
        with operation("apply_RBF_conservative_force_map", heartbeat_s,
                       detail=f"force_shape={force.shape}"):
            mapped_density, _ = RBFTransfer.apply_Tf_operator(force, ns, nbr, weights, A, lumped)
        integral = np.sum(mapped_density * lumped[:, None], axis=0)
        reference = np.sum(force * A[:, None], axis=0)
        resultant_error = float(np.linalg.norm(integral - reference) / max(np.linalg.norm(reference), 1e-16))
        fluid_xyz = np.column_stack((np.zeros(nf), y_f, np.zeros(nf)))
        solid_xyz = np.column_stack((np.zeros(ns), y_s, np.zeros(ns)))
        fluid_moment = np.sum(np.cross(fluid_xyz, force * A[:, None]), axis=0)
        solid_moment = np.sum(np.cross(solid_xyz, mapped_density * lumped[:, None]), axis=0)
        moment_error = float(np.linalg.norm(solid_moment-fluid_moment) / max(np.linalg.norm(fluid_moment), 1e-16))
        if resultant_error > 1e-12:
            raise AssertionError(f"RBF weighted resultant conservation failed: {resultant_error}")
        force_results.append({"resultant_relative_error": resultant_error,
                              "moment_relative_discrepancy_measured_not_constrained": moment_error})

    solid_virtual = np.column_stack((np.sin(np.pi*y_s), y_s**2, np.cos(np.pi*y_s)))
    with operation("apply_RBF_kinematic_interpolation", heartbeat_s,
                   detail=f"virtual_shape={solid_virtual.shape}"):
        fluid_virtual = RBFTransfer.apply_rbf_interpolation(solid_virtual, nbr, weights)
    force = forces[1]
    mapped_density, _ = RBFTransfer.apply_Tf_operator(force, ns, nbr, weights, A, lumped)
    work_fluid = float(np.sum(force * A[:, None] * fluid_virtual))
    work_solid = float(np.sum(mapped_density * lumped[:, None] * solid_virtual))
    work_error = abs(work_fluid-work_solid) / max(abs(work_fluid), 1e-16)
    if work_error > 1e-12:
        raise AssertionError(f"RBF paired virtual-work identity failed: {work_error}")
    return {"status": "PASS", "force_cases": force_results,
            "paired_virtual_work_relative_error": work_error,
            "stations_fluid": nf, "stations_solid": ns,
            "scope": "RBF backend algebra; resultant and paired work are checked; moment discrepancy is measured, not assumed conserved; not a full FEniCS/CRM interface test"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", default=str(ROOT / "config"))
    parser.add_argument("--heartbeat-seconds", type=float, default=10.0)
    args = parser.parse_args()
    started = perf_counter()
    log(f"MMS SUITE START | argv={sys.argv!r} python={sys.executable} config={args.config_dir} output={OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    with operation("load_solid_configuration", args.heartbeat_seconds):
        with open(Path(args.config_dir) / "solid_params.yaml") as stream:
            solid_cfg = yaml.safe_load(stream)
    tests = {}
    stages = (
        ("taylor_green_analytic_identity", lambda: verify_taylor_green()),
        ("nonlinear_membrane_energy_variation", verify_nonlinear_membrane_variation),
        ("plate_symbolic_manufactured_loads", lambda: verify_plate_symbolics(solid_cfg, args.heartbeat_seconds)),
        ("rbf_weighted_resultant_component", lambda: verify_rbf_component(args.heartbeat_seconds)),
        ("scalar_generalized_alpha", lambda: run_scalar_test(OUT)),
    )
    for name, check in stages:
        with operation(f"MMS_{name}", args.heartbeat_seconds):
            tests[name] = check()
    tests["full_rvpm_equation_mms"] = {
        "status": "NOT_IMPLEMENTED",
        "reason": "FLOWVPM's production particle integrator has no general manufactured vorticity-source injection; adding a classical Navier-Stokes source would not match its filtered rVPM equations.",
    }
    tests["fenics_plate_discrete_convergence"] = {
        "status": "NOT_RUN",
        "reason": "Not executed by this run_mms invocation; use the documented fenics-env commands for DOLFIN-dependent checks.",
    }
    tests["plate_production_weak_vs_independent_strong"] = {
        "status": "NOT_RUN",
        "command": "conda run -n fenics-env python verification/mms/strong_weak_check.py",
        "reason": "Requires legacy DOLFIN; compares production weak mass/damping/internal operators with the independent symbolic strong equations and natural-edge checks.",
    }
    tests["plate_residual_jacobian_directional_derivative"] = {
        "status": "NOT_RUN",
        "command": "conda run -n fenics-env python verification/mms/jacobian_check.py",
        "reason": "Requires legacy DOLFIN; finite-differences the complete production residual against its UFL tangent.",
    }
    fluid_component_csv = OUT / "flowvpm_component_metrics.csv"
    if fluid_component_csv.is_file():
        with open(fluid_component_csv, newline="") as stream:
            fluid_component_rows = list(csv.DictReader(stream))
        tests["flowvpm_kernel_core_components"] = {
            "status": "PASS" if fluid_component_rows and all(row["status"] == "PASS" for row in fluid_component_rows) else "FAIL",
            "metrics_csv": str(fluid_component_csv),
            "checks": fluid_component_rows,
        }
    else:
        tests["flowvpm_kernel_core_components"] = {
            "status": "NOT_RUN", "command": "julia --project=/path/to/FLOWUnsteady verification/mms/fluid_components.jl"
        }
    summary = {"suite": "VarFlExI verification components", "tests": tests, "full_coupled_continuum_mms": "NOT_CLAIMED"}
    with open(OUT / "mms_summary.json", "w") as stream:
        json.dump(summary, stream, indent=2)
    with open(OUT / "mms_metrics.csv", "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("test", "status", "metric", "value"))
        writer.writerow(("taylor_green_analytic_identity", "PASS", "source_max_abs", tests["taylor_green_analytic_identity"]["momentum_source_max_abs"]))
        writer.writerow(("taylor_green_analytic_identity", "PASS", "curl_definition_max_abs", tests["taylor_green_analytic_identity"]["curl_definition_max_abs"]))
        writer.writerow(("nonlinear_membrane_energy_variation", "PASS", "directional_derivative_discrepancy", 0.0))
        writer.writerow(("plate_linear_symbolic", "PASS", "source_sample_max_abs", tests["plate_symbolic_manufactured_loads"]["linear"]["source_sample_max_abs"]))
        writer.writerow(("plate_nonlinear_symbolic", "PASS", "source_sample_max_abs", tests["plate_symbolic_manufactured_loads"]["nonlinear"]["source_sample_max_abs"]))
        writer.writerow(("rbf_paired_virtual_work_component", "PASS", "relative_error", tests["rbf_weighted_resultant_component"]["paired_virtual_work_relative_error"]))
        for i, case in enumerate(tests["rbf_weighted_resultant_component"]["force_cases"]):
            writer.writerow((f"rbf_resultant_case_{i+1}", "PASS", "relative_error", case["resultant_relative_error"]))
            writer.writerow((f"rbf_moment_case_{i+1}", "MEASURED_NOT_CONSTRAINED", "relative_discrepancy", case["moment_relative_discrepancy_measured_not_constrained"]))
        for row in tests["scalar_generalized_alpha"]["levels"]:
            writer.writerow(("scalar_generalized_alpha", "PASS", "dt_s", row["dt_s"]))
            writer.writerow(("scalar_generalized_alpha", "PASS", "time_integrated_l2_error", row["time_integrated_l2_error"]))
            writer.writerow(("scalar_generalized_alpha", "PASS", "final_time_absolute_error", row["final_time_absolute_error"]))
        for row in tests["flowvpm_kernel_core_components"].get("checks", []):
            writer.writerow((row["test"], row["status"], "absolute_error", row["absolute_error"]))
    lines = ["# MMS and Component Verification", "", "Generated by `python3 verification/mms/run_mms.py`.", "",
             "| Test | Status | Scope |", "|---|---|---|"]
    lines.extend([
        "| Taylor-Green analytic identities | PASS | Exact incompressible Navier-Stokes and vorticity residual identities; not an rVPM solve. |",
        "| Nonlinear membrane directional derivative | PASS | Symbolic derivative of the quadratic Green-Lagrange strain matches its variation. |",
        "| Reissner-Mindlin plate source derivation (linear/nonlinear) | PASS | Independent symbolic strong-form source and essential/natural edge checks; no FEniCS solve. |",
        "| RBF resultant and paired virtual work | PASS | Actual local-RBF backend algebra for constant/nonuniform force and nontrivial displacement; moment discrepancy measured, not presumed conserved. |",
        f"| Installed FLOWVLM/FLOWVPM component checks | {tests['flowvpm_kernel_core_components']['status']} | Kernel limits, direct Gaussian-erf induced velocity, core growth, prescribed-U/zero-J RK3 state update, and freestream registration; no wake integration. |",
        "| Scalar generalized-alpha continuous-forcing time refinement | PASS | Six-level oscillator test uses the same coefficient/update helpers as the production integrator; see the dedicated CSV for measured orders. |",
        "| rVPM equation MMS | NOT IMPLEMENTED | No consistent source injection for the installed filtered rVPM evolution. |",
        "| Discrete FEniCS plate convergence | NOT RUN | Not run by this analytic/component driver; use `fenics-env` commands below. |",
        "| Production weak form vs independent strong form | NOT RUN | Executable test added; requires legacy DOLFIN. |",
        "| Production nonlinear residual/Jacobian finite-difference check | NOT RUN | Executable test added; requires legacy DOLFIN. |",
        "| Full coupled FSI MMS | NOT CLAIMED | No compatible manufactured sources for both solvers and interface conditions. |",
        "", "Measured analytic/component metrics:",
        "", "| Quantity | Value | Observed order |", "|---|---:|---|"])
    lines.extend([
        f"| Taylor-Green momentum residual max | {tests['taylor_green_analytic_identity']['momentum_source_max_abs']:.3e} | N/A, identity check |",
        f"| Taylor-Green curl-definition residual max | {tests['taylor_green_analytic_identity']['curl_definition_max_abs']:.3e} | N/A, identity check |",
        f"| Taylor-Green vorticity residual max | {tests['taylor_green_analytic_identity']['vorticity_source_max_abs']:.3e} | N/A, identity check |",
        f"| Linear plate source sample max | {tests['plate_symbolic_manufactured_loads']['linear']['source_sample_max_abs']:.6e} | N/A, no discrete refinement |",
        f"| Nonlinear plate source sample max | {tests['plate_symbolic_manufactured_loads']['nonlinear']['source_sample_max_abs']:.6e} | N/A, no discrete refinement |",
        f"| RBF paired virtual-work relative discrepancy | {tests['rbf_weighted_resultant_component']['paired_virtual_work_relative_error']:.3e} | N/A, algebraic identity |",
        *[
            f"| RBF resultant case {i+1} relative discrepancy | {case['resultant_relative_error']:.3e} | N/A, algebraic identity |"
            for i, case in enumerate(tests["rbf_weighted_resultant_component"]["force_cases"])
        ],
        *[
            f"| RBF moment case {i+1} relative discrepancy | {case['moment_relative_discrepancy_measured_not_constrained']:.3e} | measured, not constrained by operator |"
            for i, case in enumerate(tests["rbf_weighted_resultant_component"]["force_cases"])
        ],
        *[
            f"| {row['test']} absolute discrepancy | {float(row['absolute_error']):.3e} | N/A, component identity |"
            for row in tests["flowvpm_kernel_core_components"].get("checks", [])
        ],
        *[
            f"| Scalar oscillator {row['dt_s']:.6g} s: integrated L2 / final error | {row['time_integrated_l2_error']:.6e} / {row['final_time_absolute_error']:.6e} | measured p={row['time_integrated_l2_error_observed_order'] or 'n/a'} / {row['final_time_absolute_error_observed_order'] or 'n/a'} |"
            for row in tests["scalar_generalized_alpha"]["levels"]
        ],
        "", "The plate source uses configured constitutive constants, Green-Lagrange membrane strain including quadratic in-plane strain, Mindlin shear, rotary inertia, and the implemented mass/initial-stiffness Rayleigh damping law. Continuous forcing evaluates manufactured acceleration, velocity, and displacement at the same force time; the production residual retains its distinct generalized-alpha stages. Discrete plate convergence remains to be measured in `fenics-env`. The original strain implementation omitted the in-plane quadratic term even though its variation and documentation included it; `solid/utils/reisner_mindlin_model.py` now makes the strain and variation consistent.",
        "", "Outputs: `mms_metrics.csv`, `mms_summary.json`, `flowvpm_component_metrics.csv`, `scalar_generalized_alpha.csv`, `scalar_generalized_alpha.png`, `plate_exact_displacement.png`."
    ])
    (OUT / "MMS_REPORT.md").write_text("\n".join(lines) + "\n")
    log(f"MMS SUITE DONE | status=PASS | elapsed={perf_counter()-started:.3f}s "
        f"report={OUT / 'MMS_REPORT.md'} summary={OUT / 'mms_summary.json'}")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
