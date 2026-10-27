#!/usr/bin/env python3
"""Plan or execute separate spatial/temporal plate MMS convergence studies.

Spatial mode refines both in-plane mesh directions while keeping dt fixed.
Temporal mode refines dt on a fixed mesh. Each solve compares the numerical
plate fields to the same analytic manufactured solution; adjacent error ratios
are used to estimate observed order. Nothing runs unless ``--execute`` is set.
"""

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
sys.path.insert(0, str(HERE))
from progress import log, operation


def cases(study):
    # These levels intentionally refine a single variable family at a time.
    # Spatial: five simultaneous uniform refinements with dt held small enough
    # that the second-order time error is intended to sit below spatial error.
    # Temporal: six halvings of dt at fixed mesh and fixed physical final time.
    # For the nonlinear formulation, dt=0.01 did not converge even with
    # residual backtracking; dt=0.005 was verified to converge on this mesh.
    if study == "spatial":
        return [{"nx": n, "ny": 4*n//3, "dt": 0.0000625, "duration": 0.01}
                for n in (6, 12, 24, 48, 96)]
    return [{"nx": 48, "ny": 128, "dt": 0.005 / (2**level), "duration": 0.05}
            for level in range(6)]


def pair_order(coarse_error, fine_error, ratio):
    """Return a measured pairwise rate only for valid positive finite data."""
    coarse_error, fine_error, ratio = map(float, (coarse_error, fine_error, ratio))
    if not all(math.isfinite(value) for value in (coarse_error, fine_error, ratio)):
        return ""
    if coarse_error <= 0.0 or fine_error <= 0.0 or fine_error >= coarse_error or ratio <= 1.0:
        return ""
    return math.log(coarse_error / fine_error) / math.log(ratio)


def assess_error_sequence(errors, ratios):
    """Only publish rates when a multi-level, decreasing asymptotic regime is plausible."""
    errors = [float(value) for value in errors]
    if len(errors) < 3 or len(ratios) != len(errors) - 1:
        return [""] * max(len(errors) - 1, 0), "INCONCLUSIVE: fewer than three levels"
    if any(not math.isfinite(value) or value <= 100.0 * sys.float_info.epsilon for value in errors):
        return [""] * (len(errors) - 1), "INCONCLUSIVE: non-finite, non-positive, or roundoff-scale error"
    if any(fine >= coarse for coarse, fine in zip(errors[:-1], errors[1:])):
        return [""] * (len(errors) - 1), "INCONCLUSIVE: error is not strictly decreasing"
    candidates = [pair_order(coarse, fine, ratio)
                  for coarse, fine, ratio in zip(errors[:-1], errors[1:], ratios)]
    if any(not isinstance(rate, float) or not math.isfinite(rate) for rate in candidates):
        return [""] * (len(errors) - 1), "INCONCLUSIVE: invalid pairwise rate"
    # Near-zero rates indicate a floor from an unrefined error source; a highly
    # scattered tail does not support one asymptotic order either.
    tail = candidates[-min(3, len(candidates)):]
    if min(tail) < 0.1:
        return [""] * (len(errors) - 1), "INCONCLUSIVE: error plateau/non-refined floor suspected"
    if max(tail) - min(tail) > 1.0:
        return [""] * (len(errors) - 1), "INCONCLUSIVE: pairwise rates have not stabilized"
    return candidates, "MEASURED: decreasing multi-level sequence; interpret as observed, not formal order proof"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=("spatial", "temporal"), required=True)
    parser.add_argument("--execute", action="store_true", help="Run cases using the active Python/FEniCS environment.")
    parser.add_argument(
        "--formulation", choices=("production", "linear", "nonlinear", "both"),
        default="production",
        help="production follows geometric_nonlinearity in solid_params.yaml; both is a comparison study",
    )
    parser.add_argument("--config-dir", default=str(ROOT / "config"))
    parser.add_argument("--output-dir", default=str(OUT),
                        help="results directory; choose a fresh path to preserve previous runs")
    parser.add_argument("--heartbeat-seconds", type=float, default=10.0)
    parser.add_argument("--case-timeout-seconds", type=float, default=0.0,
                        help="optional per-case subprocess timeout; 0 disables it")
    args = parser.parse_args()
    with open(Path(args.config_dir) / "solid_params.yaml") as stream:
        import yaml
        solid_cfg = yaml.safe_load(stream)
    chord = float(solid_cfg["root_chord"])
    span = float(solid_cfg["span"])
    planned = cases(args.study)
    formulations = ("linear", "nonlinear") if args.formulation == "both" else (args.formulation,)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = Path(args.config_dir) / "solid_params.yaml"
    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    plan_path = output_dir / f"plate_{args.study}_plan.json"
    plan_path.write_text(json.dumps({
        "executed": args.execute,
        "formulations": formulations,
        "cases": planned,
        "spatial_spacing_definition": "max(root_chord/nx, span/ny)",
        "root_chord_m": chord,
        "span_m": span,
        "solid_config_path": str(config_path.resolve()),
        "solid_config_sha256": config_hash,
        "production_formulation_from_config": {
            "geometric_nonlinearity": bool(solid_cfg.get("geometric_nonlinearity", True)),
            "include_quadratic_inplane_strain": bool(solid_cfg.get("include_quadratic_inplane_strain", True)),
            "alpha_m": float(solid_cfg.get("alpha_m", 0.10)),
            "alpha_f": float(solid_cfg.get("alpha_f", 0.20)),
            "eta_m": float(solid_cfg["eta_m"]),
            "eta_k": float(solid_cfg["eta_k"]),
        },
        "shared_production_model": "solid.utils.reisner_mindlin_model.ReissnerMindlinModel",
        "manufactured_load_note": "Independent symbolic strong-form load; this is plate MMS, not coupled-FSI forcing.",
        "error_for_order": "pairwise rates are emitted only for finite, positive, strictly decreasing sequences with at least three levels, non-plateau rates, and a reasonably stable tail; otherwise the sequence is marked INCONCLUSIVE",
        "spatial_note": "hx and hy refine together; report both, and use max(hx,hy) only for a scalar rate axis",
        "mesh_aspect_ratio_note": "hx/hy is recorded per run; both nx and ny refine together",
        "temporal_note": "six dt levels starting at 0.005 s; fixed 48x128 mesh and 0.05 s final time",
    }, indent=2))
    if not args.execute:
        print(f"Wrote plan {plan_path}; no simulations run. Add --execute in fenics-env.")
        return
    raw = output_dir / f"plate_{args.study}_raw.csv"
    raw.unlink(missing_ok=True)
    manifest_path = output_dir / f"plate_{args.study}_manifest.json"
    manifest = {
        "study": args.study,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "python_executable": sys.executable,
        "solid_config_path": str(config_path.resolve()),
        "solid_config_sha256": config_hash,
        "production_formulation_from_config": {
            "geometric_nonlinearity": bool(solid_cfg.get("geometric_nonlinearity", True)),
            "include_quadratic_inplane_strain": bool(solid_cfg.get("include_quadratic_inplane_strain", True)),
            "alpha_m": float(solid_cfg.get("alpha_m", 0.10)),
            "alpha_f": float(solid_cfg.get("alpha_f", 0.20)),
            "eta_m": float(solid_cfg["eta_m"]),
            "eta_k": float(solid_cfg["eta_k"]),
        },
        "shared_production_model": "solid.utils.reisner_mindlin_model.ReissnerMindlinModel",
        "manufactured_load_note": "Independent symbolic strong-form load; this is plate MMS, not coupled-FSI forcing.",
        "cases": [],
    }
    case_index = 0
    total_cases = len(planned) * len(formulations)
    for formulation in formulations:
        for case in planned:
            case_index += 1
            case_start = perf_counter()
            command = [sys.executable, "-u", str(HERE / "fenics_plate.py"), "--config-dir", args.config_dir,
                       "--nx", str(case["nx"]), "--ny", str(case["ny"]), "--dt", str(case["dt"]),
                       "--duration", str(case["duration"]), "--output", str(raw),
                       "--heartbeat-seconds", str(args.heartbeat_seconds)]
            if formulation == "nonlinear":
                command.append("--nonlinear")
            elif formulation == "linear":
                command.append("--linear")
            elif formulation == "production":
                command.append("--production-formulation")
            record = {"formulation": formulation, **case, "command": command,
                      "status": "running", "return_code": None}
            manifest["cases"].append(record)
            manifest_path.write_text(json.dumps(manifest, indent=2))
            try:
                with operation("refinement_case", args.heartbeat_seconds,
                               detail=f"case={case_index}/{total_cases} formulation={formulation} "
                                      f"mesh={case['nx']}x{case['ny']} dt={case['dt']} "
                                      f"final_time={case['duration']}"):
                    completed = subprocess.run(
                        command, cwd=ROOT, check=False, bufsize=1,
                        timeout=args.case_timeout_seconds or None,
                    )
                record["return_code"] = completed.returncode
                record["status"] = "completed" if completed.returncode == 0 else "failed"
                record["elapsed_s"] = perf_counter() - case_start
                manifest_path.write_text(json.dumps(manifest, indent=2))
                if completed.returncode != 0:
                    raise subprocess.CalledProcessError(completed.returncode, command)
            except Exception as exc:
                record["status"] = "TIMED OUT" if isinstance(exc, subprocess.TimeoutExpired) else "FAIL"
                record["error"] = repr(exc)
                record["traceback"] = traceback.format_exc()
                record["elapsed_s"] = perf_counter() - case_start
                manifest_path.write_text(json.dumps(manifest, indent=2))
                log(f"CASE {record['status'].upper()} | case={case_index}/{total_cases} "
                    f"formulation={formulation} mesh={case['nx']}x{case['ny']} dt={case['dt']} "
                    f"final_time={case['duration']} elapsed={record['elapsed_s']:.3f}s "
                    f"traceback={record['traceback']}")
                raise
    rows = list(csv.DictReader(raw.open()))
    if not rows:
        raise RuntimeError(f"No plate MMS rows were produced; see {manifest_path}")
    expected_count = len(planned) * len(formulations)
    if len(rows) != expected_count:
        raise RuntimeError(f"Expected {expected_count} completed case rows, found {len(rows)}")
    case_keys = set()
    for row in rows:
        key = (row["formulation"], int(row["nx"]), int(row["ny"]), float(row["dt_s"]))
        if key in case_keys:
            raise RuntimeError(f"Duplicate refinement result row: {key}")
        case_keys.add(key)
        if row.get("run_status") != "completed" or row.get("solid_config_sha256") != config_hash:
            raise RuntimeError(f"Incomplete or config-mismatched result row: {key}")
    expected_keys = {
        (formulation, case["nx"], case["ny"], case["dt"])
        for formulation in formulations for case in planned
    }
    if case_keys != expected_keys:
        raise RuntimeError("Completed case rows do not match the requested refinement levels")
    expected_durations = {round(float(case["duration"]), 12) for case in planned}
    actual_durations = {round(float(row["duration_s"]), 12) for row in rows}
    if actual_durations != expected_durations:
        raise RuntimeError(
            f"Cases do not share the intended final time(s): expected {expected_durations}, got {actual_durations}"
        )
    error_keys = [key for key in rows[0] if key.endswith("_relative_l2")]
    for formulation in formulations:
        selected = [row for row in rows if row["formulation"] == formulation]
        # Order levels coarse-to-fine so both log(error ratio) and log(refinement
        # ratio) have the same sign for a convergent solution.
        selected.sort(
            key=lambda row: float(row["dt_s"]) if args.study == "temporal" else int(row["nx"]),
            reverse=args.study == "temporal",
        )
        for error_key in error_keys:
            ratios = []
            for coarse, fine in zip(selected[:-1], selected[1:]):
                if args.study == "spatial":
                    coarse_h = max(chord / int(coarse["nx"]), span / int(coarse["ny"]))
                    fine_h = max(chord / int(fine["nx"]), span / int(fine["ny"]))
                    ratios.append(coarse_h / fine_h)
                else:
                    ratios.append(float(coarse["dt_s"]) / float(fine["dt_s"]))
            rates, assessment = assess_error_sequence(
                [row[error_key] for row in selected], ratios
            )
            for index, row in enumerate(selected):
                row[f"{error_key}_observed_order"] = rates[index - 1] if index else ""
                row[f"{error_key}_order_assessment"] = assessment
    with open(output_dir / f"plate_{args.study}_convergence.csv", "w", newline="") as stream:
        columns = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["convergence_csv"] = str((output_dir / f"plate_{args.study}_convergence.csv").resolve())
    manifest["all_cases_completed"] = all(case["status"] == "completed" for case in manifest["cases"])
    manifest_path.write_text(json.dumps(manifest, indent=2))
    fig, ax = plt.subplots(figsize=(6.8, 4.4), constrained_layout=True)
    for formulation in formulations:
        selected = [row for row in rows if row["formulation"] == formulation]
        x = [max(chord/int(row["nx"]), span/int(row["ny"])) if args.study == "spatial" else float(row["dt_s"]) for row in selected]
        y = [float(row["transverse_w_relative_l2"]) for row in selected]
        order = np.argsort(x)
        ax.loglog(np.asarray(x)[order], np.asarray(y)[order], "o-", label=formulation)
    ax.set(xlabel="max(hx, hy) [m]" if args.study == "spatial" else "Time step dt [s]",
           ylabel="Relative transverse-displacement L2 error", title=f"Structural MMS {args.study} refinement")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.savefig(output_dir / f"plate_{args.study}_convergence.png", dpi=180)
    log(f"REFINEMENT DONE | study={args.study} cases={len(rows)} status=PASS "
        f"csv={output_dir / f'plate_{args.study}_convergence.csv'} "
        f"plot={output_dir / f'plate_{args.study}_convergence.png'} manifest={manifest_path}")


if __name__ == "__main__":
    main()
