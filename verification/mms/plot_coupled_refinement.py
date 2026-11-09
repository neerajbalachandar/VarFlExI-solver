#!/usr/bin/env python3
"""Plot the coupled MMS refinement report and finest-level target histories."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METRICS = [
    ("structural_state_relative_space_time_l2", "Structural state"),
    ("particle_position_relative_space_time_l2", "Particle position"),
    ("particle_circulation_relative_space_time_l2", "Particle circulation"),
    ("particle_core_relative_space_time_l2", "Particle core"),
    ("bound_circulation_relative_space_time_l2", "Bound circulation"),
    ("force_relative_space_time_l2", "Panel force"),
    ("probe_velocity_relative_space_time_l2", "Probe velocity"),
    ("probe_vorticity_relative_space_time_l2", "Probe vorticity"),
]


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def plot_errors(cases, output):
    fig, axes = plt.subplots(len(METRICS), 2, figsize=(12, 21), sharex="col")
    for column, study in enumerate(("time", "spatial")):
        series = sorted((row for row in cases if row["study"] == study),
                        key=lambda row: row["level"])
        if not series:
            raise ValueError(f"report has no {study} cases")
        x = ([row["dt_s"] for row in series] if study == "time"
             else [1.0 / row["level"] for row in series])
        for row_index, (key, title) in enumerate(METRICS):
            ax = axes[row_index, column]
            y = [row[key] for row in series]
            ax.plot(x, y, "o-", color="#1769aa", linewidth=1.8, markersize=5)
            ax.set_yscale("log")
            ax.set_title(title if column == 0 else "", loc="left", fontsize=10)
            ax.grid(True, which="both", alpha=0.25)
            if column == 0:
                ax.set_ylabel("Relative space-time L2")
            if row_index == 0:
                ax.set_title("Time refinement" if study == "time" else "Spatial refinement",
                             fontsize=12, loc="center")
            if row_index == len(METRICS)-1:
                ax.set_xlabel("dt [s]" if study == "time" else "relative mesh scale 1/level")
        # x axes are shared by column, so invert once. Calling this from every
        # subplot toggles the shared direction repeatedly and can reverse it.
        axes[0, column].invert_xaxis()
    fig.suptitle("Coupled discrete-model MMS: error against the analytic target\n"
                 "Each panel has its own vertical scale; falling curves indicate improvement",
                 fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_histories(root, output):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    for row_index, (study, case_name, title) in enumerate((
            ("time", "time_level_4", "Finest temporal case"),
            ("spatial", "spatial_level_4", "Finest spatial case"))):
        case = root / case_name
        fluid_path = case / "fluid" / "coupled_mms_force_payload_history.jsonl"
        rows = load_jsonl(fluid_path)
        times = np.asarray([item["fluid_time_s"] for item in rows])
        lift = np.asarray([item["lift"] for item in rows])
        target_lift = np.asarray([sum(item["target_force"][2]) for item in rows])
        axes[row_index, 0].plot(times, lift, "o-", ms=3, label="numerical")
        axes[row_index, 0].plot(times, target_lift, "--", label="analytic target")
        axes[row_index, 0].set_title(f"{title}: total lift")
        axes[row_index, 0].set_ylabel("Lift [N]")
        axes[row_index, 0].grid(alpha=0.25)
        axes[row_index, 0].legend(fontsize=8)

        final = rows[-1]
        nspan = len(final["bound_circulation"])
        eta = (np.arange(nspan) + 0.5) / nspan
        axes[row_index, 1].plot(eta, final["bound_circulation"], "o-", label="numerical")
        axes[row_index, 1].plot(eta, final["target_bound_circulation"], "--", label="analytic target")
        axes[row_index, 1].set_title(f"{title}: bound circulation at final time")
        axes[row_index, 1].set_ylabel("Circulation")
        axes[row_index, 1].grid(alpha=0.25)
        axes[row_index, 1].legend(fontsize=8)

        u_num = np.asarray([item["numerical_probe_velocity_m_s"][0][0] for item in rows])
        u_target = np.asarray([item["target_probe_velocity_m_s"][0][0] for item in rows])
        axes[row_index, 2].plot(times, u_num, "o-", ms=3, label="numerical")
        axes[row_index, 2].plot(times, u_target, "--", label="analytic target")
        axes[row_index, 2].set_title(f"{title}: probe 1 x-velocity")
        axes[row_index, 2].set_ylabel("Velocity [m/s]")
        axes[row_index, 2].grid(alpha=0.25)
        axes[row_index, 2].legend(fontsize=8)

    for ax in axes[-1, :]:
        ax.set_xlabel("Time [s] or span fraction")
    # Correct the mixed x-axis label on circulation plots.
    axes[0, 1].set_xlabel("Span fraction")
    axes[1, 1].set_xlabel("Span fraction")
    axes[0, 0].set_xlabel("Time [s]")
    axes[0, 2].set_xlabel("Time [s]")
    axes[1, 0].set_xlabel("Time [s]")
    axes[1, 2].set_xlabel("Time [s]")
    fig.suptitle("Numerical histories compared with their analytic target", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(output, dpi=180)
    plt.close(fig)


def write_orders(cases, output):
    columns = ["study", "metric", "level1_to_level2_order", "level2_to_level4_order",
               "errors_decrease_monotonically"]
    rows = []
    for study in ("time", "spatial"):
        series = sorted((row for row in cases if row["study"] == study),
                        key=lambda row: row["level"])
        for key, label in METRICS:
            errors = [row[key] for row in series]
            order12 = (np.log(errors[0] / errors[1]) / np.log(2)
                       if errors[0] > 0 and errors[1] > 0 else np.nan)
            order24 = (np.log(errors[1] / errors[2]) / np.log(2)
                       if errors[1] > 0 and errors[2] > 0 else np.nan)
            rows.append({"study": study, "metric": label,
                         "level1_to_level2_order": order12,
                         "level2_to_level4_order": order24,
                         "errors_decrease_monotonically": errors[0] > errors[1] > errors[2]})
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(
        "verification/mms/results/pipeline/coupled_refinement"))
    args = parser.parse_args()
    root = args.results.resolve()
    report = json.loads((root / "coupled_refinement_report.json").read_text())
    cases = report["cases"]
    plot_errors(cases, root / "coupled_refinement_error_metrics.png")
    plot_histories(root, root / "coupled_refinement_target_histories.png")
    write_orders(cases, root / "coupled_refinement_observed_orders.csv")
    print(f"Wrote coupled refinement plots and pairwise-order table under {root}")


if __name__ == "__main__":
    main()
