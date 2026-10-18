#!/usr/bin/env python3
"""Run dependency-light analytic MMS and transfer component checks."""

import argparse
import csv
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import yaml

from fluid import taylor_green_source
from plate import make_plate_mms

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "results"


def verify_taylor_green():
    data = taylor_green_source()
    residuals = [data[k] for k in ("momentum_source", "vorticity_source")]
    maximum = max(float(abs(sp.N(v.subs({data["coords"][i]: q for i, q in enumerate((0.31, 0.47, 0.0, 0.23))})))) for expr in residuals for v in expr)
    div = sp.simplify(data["divergence"])
    if maximum > 1e-10 or div != 0:
        raise AssertionError(f"Taylor-Green identity failed: residual={maximum}, div={div}")
    return {"status": "PASS", "momentum_source_max_abs": maximum, "vorticity_source_max_abs": maximum, "divergence": str(div)}


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


def verify_plate_symbolics(config):
    result = {}
    for label, nonlinear in (("linear", False), ("nonlinear", True)):
        mms = make_plate_mms(config, nonlinear, config.get("include_quadratic_inplane_strain", True))
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


def verify_rbf_component():
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
    nbr, weights = RBFTransfer.build_local_rbf_map(fluid, solid, 0.12, n_neighbors=12)
    A = np.full(nf, 1.0 / nf)
    lumped = RBFTransfer.compute_S_lumped(ns, nbr, weights, A)
    force = np.column_stack((np.ones(nf) * 2.0, np.ones(nf) * -3.0, np.ones(nf) * 0.5))
    mapped_density, _ = RBFTransfer.apply_Tf_operator(force, ns, nbr, weights, A, lumped)
    integral = np.sum(mapped_density * lumped[:, None], axis=0)
    reference = np.sum(force * A[:, None], axis=0)
    error = float(np.linalg.norm(integral - reference) / max(np.linalg.norm(reference), 1e-16))
    if error > 1e-12:
        raise AssertionError(f"RBF weighted resultant conservation failed: {error}")
    return {"status": "PASS", "weighted_resultant_relative_error": error, "stations_fluid": nf, "stations_solid": ns,
            "scope": "algebraic RBF backend test; not a FEniCS geometry/force-transfer integration test"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", default=str(ROOT / "config"))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with open(Path(args.config_dir) / "solid_params.yaml") as stream:
        solid_cfg = yaml.safe_load(stream)
    tests = {
        "taylor_green_analytic_identity": verify_taylor_green(),
        "nonlinear_membrane_energy_variation": verify_nonlinear_membrane_variation(),
        "plate_symbolic_manufactured_loads": verify_plate_symbolics(solid_cfg),
        "rbf_weighted_resultant_component": verify_rbf_component(),
    }
    tests["full_rvpm_equation_mms"] = {
        "status": "NOT_IMPLEMENTED",
        "reason": "FLOWVPM's production particle integrator has no general manufactured vorticity-source injection; adding a classical Navier-Stokes source would not match its filtered rVPM equations.",
    }
    tests["fenics_plate_discrete_convergence"] = {
        "status": "NOT_RUN",
        "reason": "legacy DOLFIN is unavailable: default Python cannot import dolfin, and fenics-env reports that it cannot find the DOLFIN pkg-config file.",
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
        writer.writerow(("nonlinear_membrane_energy_variation", "PASS", "directional_derivative_discrepancy", 0.0))
        writer.writerow(("plate_linear_symbolic", "PASS", "source_sample_max_abs", tests["plate_symbolic_manufactured_loads"]["linear"]["source_sample_max_abs"]))
        writer.writerow(("plate_nonlinear_symbolic", "PASS", "source_sample_max_abs", tests["plate_symbolic_manufactured_loads"]["nonlinear"]["source_sample_max_abs"]))
        writer.writerow(("rbf_weighted_resultant_component", "PASS", "relative_error", tests["rbf_weighted_resultant_component"]["weighted_resultant_relative_error"]))
        for row in tests["flowvpm_kernel_core_components"].get("checks", []):
            writer.writerow((row["test"], row["status"], "absolute_error", row["absolute_error"]))
    lines = ["# MMS and Component Verification", "", "Generated by `python3 verification/mms/run_mms.py`.", "",
             "| Test | Status | Scope |", "|---|---|---|"]
    lines.extend([
        "| Taylor-Green analytic identities | PASS | Exact incompressible Navier-Stokes and vorticity residual identities; not an rVPM solve. |",
        "| Nonlinear membrane directional derivative | PASS | Symbolic derivative of the quadratic Green-Lagrange strain matches its variation. |",
        "| Reissner-Mindlin plate source derivation (linear/nonlinear) | PASS | Independent symbolic strong-form source and essential/natural edge checks; no FEniCS solve. |",
        "| RBF weighted resultant | PASS | Actual local-RBF backend algebra; not full mesh transfer. |",
        f"| Installed FLOWVLM/FLOWVPM component checks | {tests['flowvpm_kernel_core_components']['status']} | Stored freestream registration, kernel normalization and RK3 core spreading; no circulation/wake integration. |",
        "| rVPM equation MMS | NOT IMPLEMENTED | No consistent source injection for the installed filtered rVPM evolution. |",
        "| Discrete FEniCS plate convergence | NOT RUN | DOLFIN is unavailable: `fenics-env` cannot locate its pkg-config file. |",
        "| Full coupled FSI MMS | NOT CLAIMED | No compatible manufactured sources for both solvers and interface conditions. |",
        "", "Measured analytic/component metrics:",
        "", "| Quantity | Value | Observed order |", "|---|---:|---|"])
    lines.extend([
        f"| Taylor-Green momentum residual max | {tests['taylor_green_analytic_identity']['momentum_source_max_abs']:.3e} | N/A, identity check |",
        f"| Taylor-Green vorticity residual max | {tests['taylor_green_analytic_identity']['vorticity_source_max_abs']:.3e} | N/A, identity check |",
        f"| Linear plate source sample max | {tests['plate_symbolic_manufactured_loads']['linear']['source_sample_max_abs']:.6e} | N/A, no discrete refinement |",
        f"| Nonlinear plate source sample max | {tests['plate_symbolic_manufactured_loads']['nonlinear']['source_sample_max_abs']:.6e} | N/A, no discrete refinement |",
        f"| RBF weighted resultant relative discrepancy | {tests['rbf_weighted_resultant_component']['weighted_resultant_relative_error']:.3e} | N/A, algebraic identity |",
        *[
            f"| {row['test']} absolute discrepancy | {float(row['absolute_error']):.3e} | N/A, component identity |"
            for row in tests["flowvpm_kernel_core_components"].get("checks", [])
        ],
        "", "The plate source uses configured constitutive constants, Green-Lagrange membrane strain including quadratic in-plane strain, Mindlin shear, rotary inertia, and the implemented mass/initial-stiffness Rayleigh damping law. Time forcing is continuous; generalized-alpha temporal convergence remains to be measured in the FEniCS environment. The original strain implementation omitted the in-plane quadratic term even though its variation and documentation included it; `solid/utils/reisner_mindlin_model.py` now makes the strain and variation consistent.",
        "", "Outputs: `mms_metrics.csv`, `mms_summary.json`, `flowvpm_component_metrics.csv`, `plate_exact_displacement.png`."
    ])
    (OUT / "MMS_REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
