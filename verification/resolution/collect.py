#!/usr/bin/env python3
"""Collect completed resolution runs and compare histories to the finest case."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUNS = Path(__file__).resolve().parent / "runs"


def read_csv(path):
    if not path.is_file():
        return []
    with open(path, newline="") as stream:
        return list(csv.DictReader(stream))


def history_error(a, b, time_key, value_key):
    ta = np.asarray([float(row[time_key]) for row in a])
    tb = np.asarray([float(row[time_key]) for row in b])
    va = np.asarray([float(row[value_key]) for row in a])
    vb = np.asarray([float(row[value_key]) for row in b])
    for label, times, values in (("case", ta, va), ("reference", tb, vb)):
        if len(times) < 2 or len(values) != len(times):
            raise ValueError(f"{label} history needs at least two paired samples")
        if not np.isfinite(times).all() or not np.isfinite(values).all():
            raise ValueError(f"{label} history contains non-finite values")
        if np.any(np.diff(times) <= 0.0):
            raise ValueError(f"{label} time samples must be strictly increasing")
    end = min(ta[-1], tb[-1])
    start = max(ta[0], tb[0])
    if end <= start:
        raise ValueError("case and reference histories have no common time interval")
    samples = np.linspace(start, end, max(200, min(2000, 10 * max(len(ta), len(tb)))))
    xa, xb = np.interp(samples, ta, va), np.interp(samples, tb, vb)
    numerator = np.sqrt(np.trapezoid((xa-xb)**2, samples))
    denominator = np.sqrt(np.trapezoid(xb**2, samples))
    if denominator > 1e-14:
        return float(numerator / denominator), "relative_L2"
    return float(numerator), "absolute_L2_reference_norm_below_1e-14"


def verify_snapshot(case):
    config_dir = Path(case["config_dir"])
    digest = hashlib.sha256()
    for filename in ("fluid_params.yaml", "solid_params.yaml", "coupling_params.yaml"):
        path = config_dir / filename
        if not path.is_file():
            return "MISSING"
        digest.update(path.read_bytes())
    return "MATCH" if digest.hexdigest() == case.get("config_sha256") else "HASH_MISMATCH"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=("structural", "panels", "particles", "time", "transfer", "fmm"), required=True)
    args = parser.parse_args()
    study_dir = RUNS / args.study
    manifest_path = study_dir / "manifest.csv"
    if not manifest_path.is_file():
        raise SystemExit(f"Missing {manifest_path}; regenerate the study plan before collection.")
    with manifest_path.open(newline="") as stream:
        manifest = list(csv.DictReader(stream))
    completed = []
    for case in manifest:
        case["config_snapshot_status"] = verify_snapshot(case)
        root = ROOT / case.get("results_dir", f"results/resolution_{args.study}/{case['case']}")
        solid = read_csv(root / "solid" / "solid_v18_diagnostics.csv")
        fluid = read_csv(root / "fluid" / f"{args.study}_{case['case']}_coupling_diagnostics.csv")
        case["solid_csv"] = str(root / "solid" / "solid_v18_diagnostics.csv")
        case["fluid_csv"] = str(root / "fluid" / f"{args.study}_{case['case']}_coupling_diagnostics.csv")
        case["status"] = "COMPLETE" if solid and fluid else "NO_OUTPUT"
        if solid and fluid:
            case["tip_displacement_final_m"] = float(solid[-1]["u_tip"])
            case["tip_displacement_max_abs_m"] = max(abs(float(row["u_tip"])) for row in solid)
            case["lift_time_integral_Ns"] = float(np.trapezoid([float(row["lift"]) for row in fluid],
                                                               [float(row["time"]) for row in fluid]))
            case["runtime_s"] = sum(float(row["solid_solve_time"]) for row in solid) + sum(float(row["fluid_step_time"]) for row in fluid)
            for key, column in (("force_transfer_relative_error_max", "force_transfer_relative_error"),
                                 ("geometry_relative_error_max", "geometry_relative_error"),
                                 ("work_relative_error_max", "work_rel_error")):
                values = [float(row[column]) for row in solid if row.get(column, "") not in ("", "nan")]
                case[key] = max(values, default=float("nan"))
            completed.append((case, solid, fluid))
    if completed:
        if args.study == "time":
            reference = min(completed, key=lambda item: float(item[0]["dt_s"]))
        elif args.study == "structural":
            reference = min(completed, key=lambda item: float(item[0]["h_s_span_m"]))
        elif args.study == "panels":
            reference = min(completed, key=lambda item: float(item[0]["h_vlm_span_m"]))
        elif args.study == "transfer":
            reference = min(completed, key=lambda item: float(item[0]["h_c_span_m"]))
        elif args.study == "particles":
            reference = min(completed, key=lambda item: float(item[0]["h_particle_release_m"]))
        else:
            reference = min(completed, key=lambda item: float(item["factor_or_value"]))
        ref_case, ref_solid, ref_fluid = reference
        for case, solid, fluid in completed:
            case["reference_case"] = ref_case["case"]
            try:
                tip_error, tip_type = history_error(solid, ref_solid, "time", "u_tip")
                lift_error, lift_type = history_error(fluid, ref_fluid, "time", "lift")
                case["tip_displacement_history_error"] = tip_error
                case["tip_displacement_history_metric"] = tip_type
                case["lift_history_error"] = lift_error
                case["lift_history_metric"] = lift_type
            except (KeyError, ValueError) as exc:
                case["comparison_status"] = f"INVALID_HISTORY: {exc}"
            else:
                case["comparison_status"] = "PASS"

    output = study_dir / "convergence_results.csv"
    columns = sorted({key for row in manifest for key in row})
    with open(output, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(manifest)
    (study_dir / "convergence_results.json").write_text(json.dumps(manifest, indent=2))
    valid = [row for row in manifest if row.get("status") == "COMPLETE"]
    if valid:
        xkey = {"time": "dt_s", "structural": "h_s_span_m", "panels": "h_vlm_span_m",
                "transfer": "h_c_span_m", "particles": "h_particle_release_m",
                "fmm": "factor_or_value"}[args.study]
        fig, ax = plt.subplots(figsize=(6.8, 4.4), constrained_layout=True)
        for key, label in (("tip_displacement_history_error", "tip displacement"),
                           ("lift_history_error", "lift")):
            pairs = sorted((float(row[xkey]), float(row[key])) for row in valid if row.get(key) not in (None, ""))
            if pairs:
                ax.loglog([v[0] for v in pairs], [max(v[1], 1e-16) for v in pairs], "o-", label=label)
        ax.set(xlabel=f"{xkey} [SI units or tolerance]", ylabel="Relative history error vs finest completed case",
               title=f"{args.study} study: estimated discretization difference")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend()
        fig.savefig(study_dir / "convergence_results.png", dpi=180)
    print(f"Collected {len(valid)}/{len(manifest)} completed cases to {output}")


if __name__ == "__main__":
    main()
