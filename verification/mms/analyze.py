#!/usr/bin/env python3
"""Compare completed standalone FLOWUnsteady fluid-resolution cases."""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent


def read_history(case):
    path = Path(case["result"]["history_path"])
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) < 2:
        raise ValueError(f"{path}: expected at least two output times")
    data = {key: np.asarray([float(row[key]) for row in rows]) for key in rows[0]}
    time = data["time_s"]
    if not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
        raise ValueError(f"{path}: invalid time history")
    for key, values in data.items():
        if not np.isfinite(values).all():
            raise ValueError(f"{path}: nonfinite values in {key}")
    return data


def relative_integrated_error(t, value, tref, reference):
    lo, hi = max(t[0], tref[0]), min(t[-1], tref[-1])
    if hi <= lo:
        raise ValueError("histories have no common physical time interval")
    samples = np.unique(np.concatenate((t[(t >= lo) & (t <= hi)],
                                        tref[(tref >= lo) & (tref <= hi)], [lo, hi])))
    a, b = np.interp(samples, t, value), np.interp(samples, tref, reference)
    numerator = math.sqrt(float(np.trapezoid((a - b) ** 2, samples)))
    denominator = math.sqrt(float(np.trapezoid(b ** 2, samples)))
    return (numerator / denominator, "relative_L2") if denominator > 1e-14 else (
        numerator, "absolute_L2_reference_norm_below_1e-14")


def relative_space_time_error(run, ref, nspan, tref):
    """Map Gamma histories to a shared eta grid, then integrate in time/span."""
    t, tr = run["time_s"], ref["time_s"]
    end, start = min(t[-1], tr[-1]), max(t[0], tr[0])
    if end <= start:
        raise ValueError("Gamma histories have no common time interval")
    times = np.unique(np.concatenate((t[(t >= start) & (t <= end)],
                                      tr[(tr >= start) & (tr <= end)], [start, end])))
    eta = (np.arange(nspan) + 0.5) / nspan

    def mapped(history):
        gamma_cols = sorted((key for key in history if key.startswith("Gamma_")),
                            key=lambda key: int(key.split("_")[1]))
        n = len(gamma_cols)
        old_eta = (np.arange(n) + 0.5) / n
        gamma = np.column_stack([history[key] for key in gamma_cols])
        eta_data = np.vstack([np.interp(eta, old_eta, row) for row in gamma])
        return np.column_stack([np.interp(times, history["time_s"], eta_data[:, j])
                                for j in range(nspan)])

    a, b = mapped(run), mapped(ref)
    numerator = math.sqrt(float(np.trapezoid(np.trapezoid((a-b)**2, eta, axis=1), times)))
    denominator = math.sqrt(float(np.trapezoid(np.trapezoid(b*b, eta, axis=1), times)))
    return (numerator / denominator, "relative_space_time_L2") if denominator > 1e-14 else (
        numerator, "absolute_space_time_L2_reference_norm_below_1e-14")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True,
                        choices=("particles", "time", "core", "panels", "fmm", "wake", "all"))
    parser.add_argument("--results", type=Path, default=HERE / "results")
    args = parser.parse_args()
    studies = ["particles", "time", "core", "panels", "fmm", "wake"] if args.study == "all" else [args.study]
    for study in studies:
        directory = args.results / study
        manifest_path = directory / "manifest.json"
        if not manifest_path.is_file():
            print(f"[{study}] no execution manifest, skipping")
            continue
        manifest = json.loads(manifest_path.read_text())
        completed = [c for c in manifest["cases"] if c["status"] == "COMPLETE" and
                     c.get("result", {}).get("status") == "COMPLETE"]
        failures = [c["case_id"] for c in manifest["cases"] if c["status"] != "COMPLETE"]
        if len(completed) < 2:
            print(f"[{study}] need at least two completed cases; completed={len(completed)}, failed={failures}")
            continue
        data = {c["case_id"]: read_history(c) for c in completed}
        for case in completed:
            ident = case["case_id"]
            history = data[ident]
            effective = case["result"]["effective"]
            times = history["time_s"]
            expected_steps = int(effective["nsteps"])
            dt = float(effective["dt_s"])
            if len(times) != expected_steps + 1:
                raise ValueError(f"{ident}: history has {len(times)} rows; expected {expected_steps + 1}")
            if not np.allclose(np.diff(times), dt, rtol=1e-8, atol=1e-12):
                raise ValueError(f"{ident}: time increments do not match effective dt={dt}")
            if not math.isclose(float(times[-1]), float(effective["duration_s"]), rel_tol=1e-8, abs_tol=1e-12):
                raise ValueError(f"{ident}: final history time does not match the configured physical duration")
            ngamma = sum(key.startswith("Gamma_") for key in history)
            if ngamma != int(effective["n_span"]):
                raise ValueError(f"{ident}: Gamma history has {ngamma} stations, expected n_span={effective['n_span']}")
        geometries = [c["result"]["geometry"] for c in completed]
        for key in ("span_m", "root_chord_m", "tip_chord_m", "leading_edge_sweep_m",
                    "aoa_deg", "twist_root_deg", "twist_tip_deg"):
            if len({float(g[key]) for g in geometries}) != 1:
                raise ValueError(f"{study}: physical geometry differs across cases ({key})")
        # The finest completed case is a numerical reference, never an exact solution.
        if study == "time":
            reference_case = min(completed, key=lambda c: float(c["result"]["effective"]["dt_s"]))
        elif study == "particles":
            reference_case = max(completed, key=lambda c: int(c["result"]["effective"]["particles_per_step"]))
        elif study == "panels":
            reference_case = max(completed, key=lambda c: int(c["result"]["effective"]["n_span"]))
        elif study == "core":
            base_lambda = float(manifest["baseline_parameters"]["lambda_vpm"])
            reference_case = min(completed, key=lambda c: abs(float(c["result"]["effective"]["lambda_vpm"]) - base_lambda))
        elif study == "fmm":
            reference_case = min(completed, key=lambda c: float(c["result"]["effective"]["fmm_relative_tolerance"]))
        elif study == "wake":
            reference_case = max(completed, key=lambda c: int(c["result"]["final_particle_count"]))
        else:
            reference_case = max(completed, key=lambda c: len(data[c["case_id"]]["time_s"]))
        ref = data[reference_case["case_id"]]
        nspan_ref = len([k for k in ref if k.startswith("Gamma_")])
        rows = []
        qois = ["CL", "CD", "lift_N", "drag_N"] + [f"u_probe{j}_{axis}" for j in range(1,4) for axis in "xyz"] + [
            f"omega_probe{j}_{axis}" for j in range(1,4) for axis in "xyz"]
        for case in completed:
            ident, cur = case["case_id"], data[case["case_id"]]
            common_start = max(float(cur["time_s"][0]), float(ref["time_s"][0]))
            common_end = min(float(cur["time_s"][-1]), float(ref["time_s"][-1]))
            if common_end <= common_start:
                raise ValueError(f"{ident}: no shared physical time interval with reference")
            record = {"case_id": ident, "status": "COMPLETE",
                      "reference_case_id": reference_case["case_id"],
                      "is_reference": ident == reference_case["case_id"],
                      "common_time_start_s": common_start, "common_time_end_s": common_end,
                      "runtime_s": case["result"]["runtime_s"],
                      "final_particle_count": case["result"]["final_particle_count"],
                      "dt_s": case["result"]["effective"]["dt_s"],
                      "particles_per_step": case["result"]["effective"]["particles_per_step"],
                      "lambda_vpm": case["result"]["effective"]["lambda_vpm"],
                      "sigma_vpm_initial_m": case["result"]["effective"]["sigma_vpm_initial_m"],
                      "release_spacing_nominal_m": case["result"]["effective"]["release_spacing_nominal_m"],
                      "n_span": case["result"]["effective"]["n_span"],
                      "fmm_relative_tolerance": case["result"]["effective"]["fmm_relative_tolerance"],
                      "fmm_direct_relative_velocity_error": case["result"].get("fmm_direct_relative_velocity_error"),
                      "particles_inserted_observed": case["result"].get("particles_inserted_observed"),
                      "particles_removed_observed": case["result"].get("particles_removed_observed")}
            for qoi in qois:
                if qoi in cur:
                    record[qoi + "_error"], record[qoi + "_error_type"] = relative_integrated_error(
                        cur["time_s"], cur[qoi], ref["time_s"], ref[qoi])
            if study == "panels":
                record["Gamma_space_time_error"], record["Gamma_error_type"] = relative_space_time_error(cur, ref, nspan_ref, ref["time_s"])
            else:
                record["Gamma_space_time_error"], record["Gamma_error_type"] = relative_space_time_error(cur, ref, nspan_ref, ref["time_s"])
            rows.append(record)
        if study in ("time", "particles", "panels"):
            refine_key = {"time": "dt_s", "particles": "particles_per_step", "panels": "n_span"}[study]
            ordered = sorted(rows, key=lambda r: float(r[refine_key]), reverse=(study == "time"))
            for coarse, fine in zip(ordered[:-1], ordered[1:]):
                if study == "time":
                    ratio = float(coarse[refine_key]) / float(fine[refine_key])
                else:
                    ratio = float(fine[refine_key]) / float(coarse[refine_key])
                if ratio <= 1:
                    continue
                for qoi in ("CL_error", "CD_error", "Gamma_space_time_error"):
                    ec, ef = coarse.get(qoi), fine.get(qoi)
                    if ec is not None and ef is not None and float(ec) > 0 and float(ef) > 0:
                        coarse[qoi + "_pairwise_order_to_next"] = math.log(float(ec)/float(ef))/math.log(ratio)
        summary = {"study": study, "reference_case": reference_case["case_id"],
                   "reference_interpretation": "finest completed numerical reference; not exact",
                   "freestream": manifest["freestream_override"], "failed_cases": failures,
                   "cases": rows}
        (directory / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
        fields = sorted({key for row in rows for key in row})
        with (directory / "metrics.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
        fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
        xkey = {"time":"dt_s", "particles":"particles_per_step", "panels":"n_span",
                "core":"lambda_vpm", "fmm":"fmm_relative_tolerance", "wake":"final_particle_count"}[study]
        for key, label in (("CL_error", "lift coefficient"), ("CD_error", "drag coefficient"),
                           ("Gamma_space_time_error", "circulation")):
            pairs = sorted((float(r[xkey]), float(r[key])) for r in rows
                           if r.get(key) is not None and not r["is_reference"] and float(r[xkey]) > 0)
            if pairs:
                ax.loglog([p[0] for p in pairs], [max(p[1], 1e-16) for p in pairs], "o-", label=label)
        ax.set(xlabel=xkey, ylabel="Integrated difference vs finest completed numerical reference",
               title=f"{study}: fluid resolution / sensitivity")
        ax.grid(True, which="both", alpha=.25); ax.legend()
        fig.savefig(directory / "error_vs_resolution.png", dpi=180); plt.close(fig)
        fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
        for r in rows:
            e = r.get("CL_error")
            if e is not None and float(e) > 0:
                ax.loglog(float(e), float(r["runtime_s"]), "o", label=r["case_id"])
        ax.set(xlabel="Lift-coefficient history difference", ylabel="Runtime [s]", title=f"{study}: cost vs difference")
        ax.grid(True, which="both", alpha=.25); ax.legend(fontsize=7)
        fig.savefig(directory / "runtime_vs_error.png", dpi=180); plt.close(fig)
        fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
        for r in rows:
            e = r.get("CL_error")
            if e is not None and float(e) > 0 and float(r["final_particle_count"]) > 0:
                ax.loglog(float(r["final_particle_count"]), float(e), "o", label=r["case_id"])
        ax.set(xlabel="Retained particles at final time", ylabel="Lift-coefficient history difference",
               title=f"{study}: particle cost vs difference")
        ax.grid(True, which="both", alpha=.25); ax.legend(fontsize=7)
        fig.savefig(directory / "particles_vs_error.png", dpi=180); plt.close(fig)
        order_note = ("Pairwise rates are descriptive estimates for the stated refinement variable; particle study co-refines release and core scales."
                      if study in ("particles", "time", "panels") else
                      "This is a sensitivity study; no formal convergence order is reported.")
        report = [f"# Fluid {study} study", "", f"Numerical reference: `{reference_case['case_id']}` (not an exact solution).",
                  f"Freestream: {manifest['freestream_override']}.", f"Failed cases excluded: {failures or 'none'}.",
                  "", order_note, "", "The particle-vorticity probe is reconstructed from the Gaussian-erf particle kernel. Probe locations are fixed in physical coordinates. rVPM has no Eulerian solver grid; any postprocessing coordinates are samples only.",
                  "", "| Case | dt [s] | p/step | lambda | n_span | runtime [s] | particles | CL difference | CL pair rate | Gamma space-time difference |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for r in rows:
            rate = r.get("CL_error_pairwise_order_to_next")
            rate_text = "n/a" if rate is None else f"{rate:.4g}"
            report.append(f"| {r['case_id']} | {r['dt_s']:.5g} | {r['particles_per_step']} | {r['lambda_vpm']:.4g} | {r['n_span']} | {r['runtime_s']:.4g} | {r['final_particle_count']} | {r.get('CL_error', float('nan')):.5g} | {rate_text} | {r['Gamma_space_time_error']:.5g} |")
        report += ["", "Continuum context: incompressible Navier-Stokes/vorticity equations; this is production rVPM self-convergence or numerical sensitivity, not continuum MMS. Particle-core regularization, bound VLM resolution, FMM evaluation, and wake management are distinct numerical effects.", ""]
        (directory / "report.md").write_text("\n".join(report))
        print(f"[{study}] analyzed {len(rows)} completed cases; reference={reference_case['case_id']}; failed={failures}")


if __name__ == "__main__":
    main()
