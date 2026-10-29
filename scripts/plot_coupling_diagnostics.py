#!/usr/bin/env python3
"""
Examples
--------
python scripts/plot_coupling_diagnostics.py "coupling_history (1).csv"
python scripts/plot_coupling_diagnostics.py rbf_rbf.csv crm_crm.csv \
    --labels RBF-RBF CRM-CRM --output-dir plots/transfer_comparison

The current socket schedule writes the solid geometry, solid solve, and solid
step timings in the geometry message sent *after* the force-transfer audit.
Consequently, they appear one CSV row later than their matching F->S load
transfer.  For component timing plots this script shifts those three fields
back by one row; the raw CSV is never modified.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

# File-only plots must not depend on a desktop/Tk installation.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


WORK_COLUMNS = (
    "incremental_work_Wf",
    "incremental_work_Ws",
    "incremental_work_defect",
    "incremental_work_relative_error",
)
BALANCE_COLUMNS = (
    "force_transfer_relative_error",
    "moment_transfer_relative_error",
    "force_transfer_residual_norm",
    "moment_transfer_residual_norm",
)
TIMING_COLUMNS = (
    "force_transfer_apply_time",
    "geometry_transfer_apply_time",
    "solid_solve_time",
    "solid_io_time",
    "solid_step_time",
    "fluid_step_time",
)
# These are emitted in the outgoing geometry message and belong to the
# immediately preceding solid solve / F->S load-transfer audit.
SOLID_MESSAGE_TIMINGS = (
    "geometry_transfer_apply_time",
    "solid_solve_time",
    "solid_io_time",
    "solid_step_time",
)


def read_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path} contains no data rows")
    out: dict[str, np.ndarray] = {}
    for key in rows[0]:
        values = []
        for row in rows:
            try:
                values.append(float(row.get(key, "nan")))
            except (TypeError, ValueError):
                values.append(np.nan)
        out[key] = np.asarray(values, dtype=float)
    for required in ("step", "time", *WORK_COLUMNS):
        if required not in out:
            raise ValueError(f"{path} is missing required column '{required}'")
    return out


def aligned(data: dict[str, np.ndarray], column: str) -> np.ndarray:
    values = data.get(column)
    if values is None:
        return np.full_like(data["time"], np.nan)
    if column not in SOLID_MESSAGE_TIMINGS:
        return values
    # CSV row i+1 contains the timing emitted with geometry generated after
    # solid step i.  Shift it onto the work audit / F->S transfer at row i.
    return np.r_[values[1:], np.nan]


def finite(values: np.ndarray) -> np.ndarray:
    return values[np.isfinite(values)]


def positive_for_log(values: np.ndarray, floor: float = 1.0e-18) -> np.ndarray:
    return np.where(np.isfinite(values), np.maximum(np.abs(values), floor), np.nan)


def save(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}.png", dpi=220)
    fig.savefig(output_dir / f"{stem}.pdf")
    plt.close(fig)


def plot_work(runs: list[tuple[str, dict[str, np.ndarray]]], output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 6.4), sharex=True)
    for label, data in runs:
        time = data["time"]
        axes[0].plot(time, data["incremental_work_Wf"], label=f"{label}: $\Delta W_f$")
        axes[0].plot(time, data["incremental_work_Ws"], "--", label=f"{label}: $\Delta W_s$")
        axes[1].semilogy(time, positive_for_log(data["incremental_work_relative_error"]), label=label)
    axes[0].set_ylabel("incremental work")
    axes[0].set_title("Discrete virtual-work audit")
    axes[0].legend(ncol=2, fontsize=8)
    axes[1].set_xlabel("coupling time [s]")
    axes[1].set_ylabel(r"$|\Delta W_f-\Delta W_s|/\max(|\Delta W_f|,|\Delta W_s|)$")
    axes[1].set_title("relative work defect")
    axes[1].grid(True, which="both", alpha=0.25)
    save(fig, output_dir, "01_work_conservation")


def plot_balance(runs: list[tuple[str, dict[str, np.ndarray]]], output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 6.4), sharex=True)
    for label, data in runs:
        time = data["time"]
        axes[0].semilogy(time, positive_for_log(data.get("force_transfer_relative_error", np.nan)), label=label)
        axes[1].semilogy(time, positive_for_log(data.get("moment_transfer_relative_error", np.nan)), label=label)
    axes[0].set_ylabel("relative force residual")
    axes[0].set_title("Global force conservation")
    axes[1].set_xlabel("coupling time [s]")
    axes[1].set_ylabel("relative moment residual")
    axes[1].set_title("Global moment conservation")
    for axis in axes:
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(fontsize=8)
    save(fig, output_dir, "02_force_moment_conservation")


def plot_timing(runs: list[tuple[str, dict[str, np.ndarray]]], output_dir: Path) -> None:
    fig, axes = plt.subplots(len(runs), 1, figsize=(9, 3.0 * len(runs)), sharex=False)
    axes = np.atleast_1d(axes)
    for axis, (label, data) in zip(axes, runs):
        time = data["time"]
        series = {
            "F→S load transfer": aligned(data, "force_transfer_apply_time"),
            "S→F geometry transfer": aligned(data, "geometry_transfer_apply_time"),
            "solid nonlinear solve": aligned(data, "solid_solve_time"),
            "solid output": aligned(data, "solid_io_time"),
            "fluid callback": aligned(data, "fluid_step_time"),
            "fluid advance between callbacks": aligned(
                data, "fluid_advance_between_callbacks_time"
            ),
        }
        for name, values in series.items():
            axis.plot(time, values, label=name, linewidth=1.0)
        axis.set_title(f"Per-step measured compute times — {label}")
        axis.set_xlabel("coupling time [s]")
        axis.set_ylabel("wall time [s]")
        axis.set_yscale("log")
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(fontsize=8, ncol=2)
    save(fig, output_dir, "03_component_timing")


def plot_displacement_transfer_error(runs: list[tuple[str, dict[str, np.ndarray]]], output_dir: Path) -> None:
    """Plot S->F displacement-transfer error at control points.

    New runs provide the direct structural-field vs reconstructed geometry
    error at CPs. Older files only contain the fluid geometry echo check; it
    is plotted as a clearly labelled legacy proxy, not as the CP error.
    """
    if not any(
        "cp_geometry_rms_error" in data or "geometry_transfer_relative_error" in data
        for _, data in runs
    ):
        return
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 6.0), sharex=True)
    for label, data in runs:
        time = data["time"]
        if "cp_geometry_rms_error" in data:
            axes[0].semilogy(time, positive_for_log(data["cp_geometry_rms_error"]), label=f"{label}: RMS")
        if "cp_geometry_max_error" in data:
            axes[0].semilogy(time, positive_for_log(data["cp_geometry_max_error"]), "--", label=f"{label}: max")
        if "cp_geometry_relative_rms_error" in data:
            axes[1].semilogy(time, positive_for_log(data["cp_geometry_relative_rms_error"]), label=label)
        elif "geometry_transfer_relative_error" in data:
            axes[1].semilogy(
                time,
                positive_for_log(data["geometry_transfer_relative_error"]),
                label=f"{label}: legacy fluid-geometry echo",
            )
    axes[0].set_ylabel("CP geometry error [m]")
    axes[0].set_title("Solid-to-fluid displacement reconstruction at control points")
    axes[1].set_xlabel("coupling time [s]")
    axes[1].set_ylabel("relative displacement-transfer error")
    for axis in axes:
        axis.grid(True, which="both", alpha=0.25)
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(fontsize=8)
    if not any("cp_geometry_rms_error" in data for _, data in runs):
        axes[0].text(
            0.5,
            0.5,
            "CP displacement error was not logged in this legacy CSV.",
            transform=axes[0].transAxes,
            ha="center",
            va="center",
        )
    save(fig, output_dir, "03_displacement_transfer_error")


def plot_timing_comparison(runs: list[tuple[str, dict[str, np.ndarray]]], output_dir: Path) -> None:
    labels = []
    values = []
    for run_label, data in runs:
        for component, column in (
            ("F→S", "force_transfer_apply_time"),
            ("S→F", "geometry_transfer_apply_time"),
            ("solid solve", "solid_solve_time"),
            ("fluid callback", "fluid_step_time"),
            ("fluid advance", "fluid_advance_between_callbacks_time"),
        ):
            sample = finite(aligned(data, column))
            if sample.size:
                labels.append(f"{run_label}\n{component}")
                values.append(sample)
    fig, axis = plt.subplots(figsize=(max(8, 0.95 * len(labels)), 4.8))
    axis.boxplot(values, showfliers=False)
    axis.set_xticks(np.arange(1, len(labels) + 1), labels, rotation=35, ha="right")
    axis.set_yscale("log")
    axis.set_ylabel("wall time [s]")
    axis.set_title("Timing distribution comparison (median and interquartile range)")
    axis.grid(True, axis="y", which="both", alpha=0.25)
    save(fig, output_dir, "04_timing_distribution")


def plot_solver_health(runs: list[tuple[str, dict[str, np.ndarray]]], output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 6.0), sharex=True)
    for label, data in runs:
        time = data["time"]
        if "newton_iterations" in data:
            axes[0].plot(time, data["newton_iterations"], label=label)
        if "newton_relative_residual" in data:
            axes[1].semilogy(time, positive_for_log(data["newton_relative_residual"]), label=label)
    axes[0].set_ylabel("nonlinear iterations")
    axes[0].set_title("Solid nonlinear-solver health")
    axes[1].set_xlabel("coupling time [s]")
    axes[1].set_ylabel("relative residual")
    for axis in axes:
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(fontsize=8)
    save(fig, output_dir, "05_nonlinear_solver_health")


def plot_orchestrator_stages(runs: list[tuple[str, dict[str, np.ndarray]]], output_dir: Path) -> None:
    """Plot scheduler stages; these are not presented as network latency."""
    columns = (
        ("wait: solid geometry", "wait_for_solid_geometry_time"),
        ("send: geometry", "server_send_geometry_time"),
        ("wait: fluid force", "wait_for_fluid_force_time"),
        ("server force processing", "server_force_processing_time"),
        ("send: force", "server_send_force_time"),
        ("wait: solid diagnostic", "wait_for_solid_diagnostic_time"),
    )
    if not any(column in data for _, data in runs for _, column in columns):
        return
    fig, axes = plt.subplots(len(runs), 1, figsize=(9, 3.0 * len(runs)), sharex=False)
    for axis, (label, data) in zip(np.atleast_1d(axes), runs):
        for title, column in columns:
            if column in data:
                axis.plot(data["time"], data[column], label=title, linewidth=1.0)
        axis.set_yscale("log")
        axis.set_xlabel("coupling time [s]")
        axis.set_ylabel("orchestrator stage [s]")
        axis.set_title(f"Coupling-server scheduling stages — {label}")
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(fontsize=8, ncol=2)
    save(fig, output_dir, "06_orchestrator_stage_timing")


def write_summary(runs: list[tuple[str, dict[str, np.ndarray]]], output_dir: Path) -> None:
    metrics = [
        "incremental_work_relative_error",
        "force_transfer_relative_error",
        "moment_transfer_relative_error",
        "force_transfer_apply_time",
        "geometry_transfer_apply_time",
        "solid_solve_time",
        "fluid_step_time",
        "fluid_advance_between_callbacks_time",
        "cp_geometry_rms_error",
        "cp_geometry_max_error",
        "cp_geometry_relative_rms_error",
        "wait_for_solid_geometry_time",
        "server_send_geometry_time",
        "wait_for_fluid_force_time",
        "server_force_processing_time",
        "server_send_force_time",
        "wait_for_solid_diagnostic_time",
    ]
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run", "metric", "count", "mean", "median", "p95", "max"])
        for label, data in runs:
            for metric in metrics:
                values = finite(aligned(data, metric))
                if values.size:
                    writer.writerow([
                        label, metric, len(values), np.mean(values), np.median(values),
                        np.percentile(values, 95), np.max(values),
                    ])
    (output_dir / "timing_scope.txt").write_text(
        "Measured timing definitions\n"
        "==========================\n"
        "force_transfer_apply_time: fluid CP/panel loads -> solid nodal loads only.\n"
        "geometry_transfer_apply_time: solid interface displacement -> fluid LE/TE geometry only.\n"
        "solid_solve_time: nonlinear structural solve only.\n"
        "solid_io_time: solid field-output writes only.\n"
        "solid_step_time: broad solid-side post-receive work; not a pure solver time.\n"
        "fluid_step_time: force/diagnostic work inside the coupling callback, excluding socket exchange.\n"
        "fluid_advance_between_callbacks_time: elapsed work outside consecutive callbacks; it is the closest\n"
        "available proxy for fluid advancement, but may include framework/output overhead.\n"
        "fluid_interstep_time is deliberately excluded: it is elapsed time between callback starts, not communication latency.\n"
        "wait_for_* fields are scheduler-stage durations. They include peer computation and blocking I/O and are\n"
        "not pure network latency. Pure one-way latency needs synchronized endpoint clocks or packet-level tracing.\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="+", type=Path, help="one or more coupling-history CSV files")
    parser.add_argument("--labels", nargs="*", help="labels matching CSV order")
    parser.add_argument("--output-dir", type=Path, default=Path("plots/coupling_diagnostics"))
    args = parser.parse_args()
    if args.labels and len(args.labels) != len(args.csv):
        parser.error("--labels must contain exactly one label per CSV")
    labels = args.labels or [path.stem for path in args.csv]
    runs = [(label, read_csv(path)) for label, path in zip(labels, args.csv)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_work(runs, args.output_dir)
    plot_balance(runs, args.output_dir)
    plot_displacement_transfer_error(runs, args.output_dir)
    plot_timing(runs, args.output_dir)
    plot_timing_comparison(runs, args.output_dir)
    plot_solver_health(runs, args.output_dir)
    plot_orchestrator_stages(runs, args.output_dir)
    write_summary(runs, args.output_dir)
    print(f"Wrote plots and summary to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
