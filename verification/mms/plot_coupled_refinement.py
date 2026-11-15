#!/usr/bin/env python3
"""Plot analytical-target coupled MMS refinement metrics and histories."""
import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METRICS = [
    ("structural_state_relative_space_time_l2", "Structural state", "structural", 2),
    ("particle_position_relative_space_time_l2", "Particle position", "particle", 3),
    ("particle_circulation_relative_space_time_l2", "Particle circulation", "particle", 3),
    ("particle_core_relative_space_time_l2", "Particle core", "particle", 3),
    ("probe_velocity_relative_space_time_l2", "Probe velocity", "probe", 3),
    ("probe_vorticity_relative_space_time_l2", "Probe vorticity", "probe", 3),
    ("bound_circulation_relative_space_time_l2", "Bound circulation", "algebraic", "mixed"),
    ("force_relative_space_time_l2", "Panel force", "algebraic", "mixed"),
]
RATE_METRICS = METRICS[:6]
TINY = 1.0e-30


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def case_path(row):
    return Path(row["case_dir"]).resolve()


def _triangle(ax, order, xlo, xhi, ylo, yhi):
    """Draw a dotted p-order guide with horizontal and vertical legs."""
    if xlo <= 0 or xhi <= xlo or ylo <= 0 or yhi <= ylo:
        return
    x0 = xlo * 1.18
    x1 = min(x0 * 1.7, xhi / 1.08)
    y0 = ylo * 1.5
    y1 = y0 * (x1 / x0) ** order
    if x1 <= x0 or y1 >= yhi / 1.15:
        # Preserve the expected slope while shrinking the guide to fit.
        x1 = min(x0 * (yhi / (y0 * 1.5)) ** (1.0 / order), xhi / 1.08)
        y1 = y0 * (x1 / x0) ** order
    if x1 <= x0 or y1 <= y0 or y1 > yhi:
        return
    ax.plot([x0, x1, x1], [y0, y0, y1], ":", color="0.3", linewidth=1.0)
    ax.text(math.sqrt(x0*x1), y0 * (y1/y0)**0.12, f"p={order}", fontsize=7,
            rotation=math.degrees(math.atan2(math.log(y1/y0), math.log(x1/x0))),
            color="0.25", ha="center", va="bottom")


def plot_errors(cases, output):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.5))
    colors = {"structural": "#1f77b4", "particle": "#d95f02", "probe": "#2ca02c"}
    markers = {"structural": "o", "particle": "s", "probe": "^"}
    for ax, study in zip(axes, ("time", "spatial")):
        series = sorted((row for row in cases if row["study"] == study), key=lambda row: row["level"])
        if not series:
            raise ValueError(f"report has no {study} cases")
        x = np.asarray([row["dt_s"] if study == "time" else 1.0 / float(row["level"])
                        for row in series], dtype=float)
        for key, label, group, _theory in RATE_METRICS:
            y = np.asarray([float(row[key]) for row in series], dtype=float)
            ax.loglog(x, np.maximum(y, TINY), marker=markers[group], linewidth=1.5,
                      markersize=4, color=colors[group], alpha=0.88, label=label)
        ax.set_title("Temporal refinement" if study == "time" else "Coupled spatial refinement")
        ax.set_xlabel("Time step dt [s]" if study == "time" else "Relative mesh scale 1/level")
        ax.set_ylabel("Relative space-time L2 error")
        ax.grid(True, which="both", alpha=0.25)
        # Both axes increase from fine (left) to coarse (right), as requested.
        xlo, xhi = float(np.min(x)), float(np.max(x))
        all_y = [float(row[key]) for row in series for key, *_ in RATE_METRICS]
        positive = [value for value in all_y if value > 0 and math.isfinite(value)]
        if positive:
            ylo, yhi = min(positive), max(positive)
            ax.set_xlim(xlo / 1.5, xhi * 1.5)
            ax.set_ylim(ylo / 2.5, yhi * 4.0)
            _triangle(ax, 2, xlo / 1.5, xhi * 1.5, ylo / 2.5, yhi * 4.0)
            _triangle(ax, 3, xlo / 1.5, xhi * 1.5, ylo / 2.5, yhi * 4.0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=7, frameon=False,
               bbox_to_anchor=(0.5, 0.01))
    fig.suptitle("Coupled discrete MMS errors against analytical targets", fontsize=11)
    fig.tight_layout(rect=(0, 0.13, 1, 0.94))
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_histories(root, cases, output):
    fig, axes = plt.subplots(2, 3, figsize=(9.0, 6.0))
    colors = plt.get_cmap("viridis")
    for row_index, study in enumerate(("time", "spatial")):
        series = sorted((row for row in cases if row["study"] == study), key=lambda row: row["level"])
        if not series:
            raise ValueError(f"report has no {study} cases")
        all_rows = [(case, load_jsonl(case_path(case) / "fluid" / "coupled_mms_force_payload_history.jsonl"))
                    for case in series]
        target_case, target_rows = all_rows[-1]
        target_time = np.asarray([item["fluid_time_s"] for item in target_rows], dtype=float)
        target_lift = np.asarray([np.sum(np.asarray(item["target_force"])[2]) for item in target_rows])
        target_probe = np.asarray([item["target_probe_velocity_m_s"][0][0] for item in target_rows])
        for level_index, (case, rows) in enumerate(all_rows):
            color = colors(level_index / max(1, len(all_rows)-1))
            times = np.asarray([item["fluid_time_s"] for item in rows], dtype=float)
            lift = np.asarray([item["lift"] for item in rows], dtype=float)
            probe = np.asarray([item["numerical_probe_velocity_m_s"][0][0] for item in rows], dtype=float)
            label = f"level {case['level']}"
            axes[row_index, 0].plot(times, lift, color=color, marker=".", ms=3, label=label)
            axes[row_index, 2].plot(times, probe, color=color, marker=".", ms=3, label=label)
        axes[row_index, 0].plot(target_time, target_lift, "k--", linewidth=2.0, label="analytic target")
        axes[row_index, 2].plot(target_time, target_probe, "k--", linewidth=2.0, label="analytic target")
        final = target_rows[-1]
        gamma = np.asarray(final["bound_circulation"], dtype=float)
        gamma_target = np.asarray(final["target_bound_circulation"], dtype=float)
        eta = (np.arange(len(gamma)) + 0.5) / len(gamma)
        axes[row_index, 1].plot(eta, gamma, "o-", ms=3, label=f"level {target_case['level']}")
        axes[row_index, 1].plot(eta, gamma_target, "k--", linewidth=2.0, label="analytic target")
        axes[row_index, 0].set_title(f"{study.title()}: lift")
        axes[row_index, 1].set_title(f"{study.title()}: final bound circulation")
        axes[row_index, 2].set_title(f"{study.title()}: probe 1 x-velocity")
        axes[row_index, 0].set_ylabel("Lift [N]")
        axes[row_index, 1].set_ylabel("Circulation")
        axes[row_index, 2].set_ylabel("Velocity [m/s]")
        for col in range(3):
            axes[row_index, col].grid(alpha=0.25)
            axes[row_index, col].legend(fontsize=6)
    for ax in axes[0]:
        ax.set_xlabel("Time [s]" if ax is not axes[0, 1] else "Span fraction")
    for ax in axes[1]:
        ax.set_xlabel("Time [s]" if ax is not axes[1, 1] else "Span fraction")
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _metric_status(report, study, metric):
    assessments = report.get("rate_assessment", {})
    if isinstance(assessments, dict):
        study_data = assessments.get(study, assessments)
        if isinstance(study_data, dict):
            value = study_data.get(metric)
            if isinstance(value, dict):
                value = value.get("status")
            if value is not None:
                return str(value)
    # Some reports store the assessment alongside each study's first row.
    for row in report.get("cases", []):
        if row.get("study") == study and isinstance(row.get("rate_assessment"), dict):
            value = row["rate_assessment"].get(metric)
            if isinstance(value, dict):
                value = value.get("status")
            if value is not None:
                return str(value)
    return "n/a"


def _pair_order(coarse, fine, h_coarse, h_fine):
    if (coarse is None or fine is None or h_coarse is None or h_fine is None
            or coarse <= 0 or fine <= 0 or h_coarse <= h_fine or h_fine <= 0):
        return ""
    return math.log(float(coarse) / float(fine)) / math.log(float(h_coarse) / float(h_fine))


def write_orders(report, output):
    columns = ["study", "metric", "label", "theory", "p_1_2", "p_2_4", "status"]
    rows = []
    for study in ("time", "spatial"):
        series = sorted((row for row in report["cases"] if row["study"] == study),
                        key=lambda row: int(row["level"]))
        by_level = {int(row["level"]): row for row in series}
        for metric, label, _group, theory in METRICS:
            theory_label = str(theory) if theory == "mixed" else ("—" if theory == "algebraic" else theory)
            values = {level: by_level.get(level, {}).get(metric) for level in (1, 2, 4)}
            h = {}
            for level in (1, 2, 4):
                row = by_level.get(level)
                h[level] = (float(row["dt_s"]) if study == "time" else 1.0 / level) if row else None
            rows.append({
                "study": study, "metric": metric, "label": label, "theory": theory_label,
                "p_1_2": _pair_order(values[1], values[2], h[1], h[2]),
                "p_2_4": _pair_order(values[2], values[4], h[2], h[4]),
                "status": _metric_status(report, study, metric),
            })
    with Path(output).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _relative_pointwise_error(numerical, target):
    numerical = np.asarray(numerical, dtype=float)
    target = np.asarray(target, dtype=float)
    if numerical.shape != target.shape:
        raise ValueError(f"pointwise vectors differ in shape: {numerical.shape} != {target.shape}")
    return float(np.linalg.norm(numerical - target) / max(float(np.linalg.norm(target)), 1.0e-14))


def plot_error_vs_time(cases, output):
    panels = (("time", "structural_state_relative_space_time_l2", "Structural"),
              ("time", "particle_position_relative_space_time_l2", "Particle position"),
              ("spatial", "structural_state_relative_space_time_l2", "Structural"),
              ("spatial", "particle_position_relative_space_time_l2", "Particle position"))
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.0))
    colors = plt.get_cmap("viridis")
    for ax, (study, _metric, title) in zip(axes.flat, panels):
        series = sorted((row for row in cases if row["study"] == study), key=lambda row: row["level"])
        for idx, case in enumerate(series):
            directory = case_path(case)
            color = colors(idx / max(1, len(series)-1))
            if title == "Structural":
                with (directory / "solid_mms_history.csv").open(newline="", encoding="utf-8") as stream:
                    history = list(csv.DictReader(stream))
                times = np.asarray([float(row["time_s"]) for row in history])
                errors = []
                for row in history:
                    numerical = json.loads(row["numerical_state"])
                    target = json.loads(row["target_state"])
                    errors.append(_relative_pointwise_error(numerical, target))
            else:
                fluid_path = directory / "fluid" / "coupled_mms_force_payload_history.jsonl"
                history = load_jsonl(fluid_path)
                times = np.asarray([row["fluid_time_s"] for row in history])
                errors = []
                for row in history:
                    numerical = row.get("numerical_particle_state", {}).get("position")
                    target = row.get("target_particle_state", {}).get("position")
                    errors.append(_relative_pointwise_error(numerical, target))
            ax.semilogy(times, np.maximum(errors, TINY), marker=".", ms=3,
                        label=f"level {case['level']}", color=color)
        ax.set_title(f"{study.title()}: {title}")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Pointwise relative L2 error")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_bound_circulation_residual(cases, smoke_path, output):
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for study, marker in (("time", "o"), ("spatial", "s")):
        series = sorted((row for row in cases if row["study"] == study), key=lambda row: row["level"])
        levels, residuals = [], []
        for case in series:
            history_path = case_path(case) / "fluid" / "coupled_mms_force_payload_history.jsonl"
            history = load_jsonl(history_path)
            levels.append(int(case["level"]))
            residuals.append(max(float(row["target_circulation_residual"]) for row in history))
        ax.semilogy(levels, np.maximum(residuals, TINY), marker=marker, label=study.title())
    if not Path(smoke_path).is_file():
        raise FileNotFoundError(f"coupled smoke summary is required for the residual reference line: {smoke_path}")
    smoke = json.loads(Path(smoke_path).read_text())
    reference = float(smoke["target_circulation_residual_max"])
    ax.axhline(reference, color="black", linestyle="--", linewidth=1.2,
               label="coupled smoke reference")
    ax.set_xlabel("Refinement level")
    ax.set_ylabel("Maximum target circulation residual")
    ax.set_xticks(sorted({int(row["level"]) for row in cases}))
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(
        "verification/mms/results/pipeline/coupled_refinement"))
    parser.add_argument("--smoke-summary", type=Path)
    args = parser.parse_args()
    root = args.results.resolve()
    report = json.loads((root / "coupled_refinement_report.json").read_text())
    cases = report["cases"]
    plot_errors(cases, root / "coupled_refinement_error_metrics.png")
    plot_histories(root, cases, root / "coupled_refinement_target_histories.png")
    write_orders(report, root / "coupled_refinement_observed_orders.csv")
    plot_error_vs_time(cases, root / "coupled_refinement_error_vs_time.png")
    smoke_path = args.smoke_summary or root.parent / "coupled_smoke" / "mms_summary.json"
    plot_bound_circulation_residual(
        cases, smoke_path, root / "coupled_refinement_bound_circulation_residual.png"
    )
    print(f"Wrote coupled refinement figures and order table under {root}")


if __name__ == "__main__":
    main()
