#!/usr/bin/env python3
"""Plan or run spatial/temporal Reissner-Mindlin MMS refinements."""

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / "results"


def cases(study):
    if study == "spatial":
        return [{"nx": n, "ny": 4*n//3, "dt": 0.0005, "duration": 0.01} for n in (6, 12, 24)]
    return [{"nx": 24, "ny": 64, "dt": dt, "duration": 0.05} for dt in (0.01, 0.005, 0.0025, 0.00125)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=("spatial", "temporal"), required=True)
    parser.add_argument("--execute", action="store_true", help="Run cases using the active Python/FEniCS environment.")
    parser.add_argument("--formulation", choices=("linear", "nonlinear", "both"), default="both")
    parser.add_argument("--config-dir", default=str(ROOT / "config"))
    args = parser.parse_args()
    planned = cases(args.study)
    formulations = ("linear", "nonlinear") if args.formulation == "both" else (args.formulation,)
    OUT.mkdir(parents=True, exist_ok=True)
    plan_path = OUT / f"plate_{args.study}_plan.json"
    plan_path.write_text(json.dumps({"executed": args.execute, "formulations": formulations, "cases": planned}, indent=2))
    if not args.execute:
        print(f"Wrote plan {plan_path}; no simulations run. Add --execute in fenics-env.")
        return
    raw = OUT / f"plate_{args.study}_raw.csv"
    raw.unlink(missing_ok=True)
    for formulation in formulations:
        for case in planned:
            command = [sys.executable, str(HERE / "fenics_plate.py"), "--config-dir", args.config_dir,
                       "--nx", str(case["nx"]), "--ny", str(case["ny"]), "--dt", str(case["dt"]),
                       "--duration", str(case["duration"]), "--output", str(raw)]
            if formulation == "nonlinear":
                command.append("--nonlinear")
            subprocess.run(command, cwd=ROOT, check=True)
    rows = list(csv.DictReader(raw.open()))
    value_key = "transverse_w_relative_l2"
    for formulation in formulations:
        selected = [row for row in rows if row["formulation"] == formulation]
        selected.sort(key=lambda row: float(row["dt_s"]) if args.study == "temporal" else int(row["nx"]))
        if args.study == "spatial":
            selected.sort(key=lambda row: int(row["nx"]), reverse=True)
        for i, row in enumerate(selected):
            if i == 0:
                row["observed_order"] = ""
                continue
            prev = selected[i - 1]
            if args.study == "spatial":
                coarse_h = max(0.24/int(prev["nx"]), 0.96/int(prev["ny"]))
                fine_h = max(0.24/int(row["nx"]), 0.96/int(row["ny"]))
                ratio = coarse_h / fine_h
            else:
                ratio = float(prev["dt_s"]) / float(row["dt_s"])
            row["observed_order"] = math.log(float(prev[value_key])/float(row[value_key])) / math.log(ratio)
    with open(OUT / f"plate_{args.study}_convergence.csv", "w", newline="") as stream:
        columns = list(rows[0]) + (["observed_order"] if "observed_order" not in rows[0] else [])
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    fig, ax = plt.subplots(figsize=(6.8, 4.4), constrained_layout=True)
    for formulation in formulations:
        selected = [row for row in rows if row["formulation"] == formulation]
        x = [max(0.24/int(row["nx"]), 0.96/int(row["ny"])) if args.study == "spatial" else float(row["dt_s"]) for row in selected]
        y = [float(row[value_key]) for row in selected]
        order = np.argsort(x)
        ax.loglog(np.asarray(x)[order], np.asarray(y)[order], "o-", label=formulation)
    ax.set(xlabel="Mesh spacing h [m]" if args.study == "spatial" else "Time step dt [s]",
           ylabel="Relative transverse-displacement L2 error", title=f"Structural MMS {args.study} refinement")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.savefig(OUT / f"plate_{args.study}_convergence.png", dpi=180)
    print(f"Saved measured errors and convergence plot under {OUT}")


if __name__ == "__main__":
    main()
