#!/usr/bin/env python3
"""Post-process FLOWUnsteady/FEniCS runs for nonlinear response diagnostics.

Run from the repository root, for example:
    python3 notebooks/nonlinear_analysis.py --run-dir results/rect_nonlinear_regime

Outputs a long-form ``coupling/nonlinear_analysis.csv`` containing time samples
and scalar diagnostics, plus PNG figures in ``coupling/nonlinear_analysis_plots``.
The script reads existing outputs only; it never launches a simulation.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
from scipy.spatial import cKDTree


def config_number(path: Path, key: str, default: float) -> float:
    if not path.exists():
        return default
    match = re.search(rf"^\s*{re.escape(key)}\s*:\s*([-+0-9.eE]+)", path.read_text(), re.M)
    return float(match.group(1)) if match else default


def xdmf_series(path: Path):
    """Read nodal vector/scalar attributes from temporal XDMF/HDF5."""
    root = ET.parse(path).getroot()
    grids = [g for g in root.iter() if g.tag.endswith("Grid") and g.attrib.get("GridType") != "Collection"]
    # Includes may be present in later time grids; the first grid has geometry.
    geom_item = next((x for x in grids[0].iter() if x.tag.endswith("DataItem") and "/geometry" in (x.text or "")), None) if grids else None
    if geom_item is None:
        raise ValueError(f"No mesh geometry DataItem in {path}")
    h5name, key = (geom_item.text or "").strip().split(":", 1)
    h5path = (path.parent / h5name).resolve()
    with h5py.File(h5path, "r") as h5:
        coords = h5[key][...]
        times, frames, field_names = [], [], []
        for grid in grids:
            ti = next((n for n in grid if n.tag.endswith("Time")), None)
            if ti is None:
                continue
            attr = next((n for n in grid if n.tag.endswith("Attribute")), None)
            if attr is None:
                continue
            item = next((n for n in attr.iter() if n.tag.endswith("DataItem")), None)
            if item is None or not item.text or ":" not in item.text:
                continue
            fn, ds = item.text.strip().split(":", 1)
            if Path(fn).name != h5path.name:
                continue
            frames.append(h5[ds][...])
            times.append(float(ti.attrib["Value"]))
            field_names.append(attr.attrib.get("Name", "field"))
    if not frames:
        raise ValueError(f"No time-series frames found in {path}")
    return np.asarray(times), coords, np.asarray(frames), field_names[0]


def nearest_nodes(coords):
    xy = np.asarray(coords)
    x, y = xy[:, 0], xy[:, 1]
    xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()
    chord, span = xmax - xmin, ymax - ymin
    def nearest(tx, ty):
        return int(np.argmin((x - tx) ** 2 + (y - ty) ** 2))
    # Plate axis convention in this repository: x is chord, y is span.
    return {
        "tip": nearest(xmin + 0.75 * chord, ymax),
        "mid": nearest(xmin + 0.50 * chord, ymin + 0.50 * span),
        "leading_edge": nearest(xmin, ymin + 0.50 * span),
        "root": nearest(xmin + 0.75 * chord, ymin),
    }


def finite_derivative(y, t):
    return np.gradient(y, t, axis=0, edge_order=2 if len(t) > 2 else 1)


def psd_metrics(x, dt):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 8 or not np.any(np.abs(x - np.mean(x)) > 0):
        return np.array([]), np.array([]), np.nan, np.nan
    f, p = signal.welch(x - np.mean(x), fs=1 / dt, nperseg=min(4096, len(x)))
    i = int(np.argmax(p[1:]) + 1) if len(p) > 1 else 0
    return f, p, float(f[i]) if i else np.nan, float(p[i]) if i else np.nan


def recurrence_metrics(x, max_points=1500):
    """Simple recurrence quantification with fixed 10%-distance threshold."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 20:
        return np.nan, np.nan, np.nan
    if len(x) > max_points:
        x = signal.resample(x, max_points)
    sd = np.std(x)
    if sd == 0:
        return 1.0, 1.0, 1.0
    z = (x - np.mean(x)) / sd
    # Delay embedding (m=3, tau=1), then recurrence matrix.
    n_emb = len(z) - 2
    emb = np.column_stack([z[i:i + n_emb] for i in range(3)])
    d = np.linalg.norm(emb[:, None, :] - emb[None, :, :], axis=2)
    r = d < 0.1 * np.sqrt(3)
    np.fill_diagonal(r, False)
    recurrence_rate = float(r.mean())
    # DET: recurrence points in diagonal lines at least 2 points long.
    diag_points = 0
    for k in range(-len(r) + 1, len(r)):
        diag = np.diag(r, k)
        if len(diag) < 2:
            continue
        padded = np.r_[False, diag, False].astype(int)
        changes = np.diff(padded)
        lengths = np.where(changes == -1)[0] - np.where(changes == 1)[0]
        diag_points += int(lengths[lengths >= 2].sum())
    det = diag_points / max(int(r.sum()), 1)
    # Laminar fraction: recurrence points in vertical lines >= 2.
    vertical = 0
    for col in r.T:
        padded = np.r_[False, col, False].astype(int)
        changes = np.diff(padded)
        lengths = np.where(changes == -1)[0] - np.where(changes == 1)[0]
        vertical += int(lengths[lengths >= 2].sum())
    lam = vertical / max(int(r.sum()), 1)
    return float(det), float(lam), recurrence_rate


def zero_one_test(x, c_count=40):
    """Gottwald-Melbourne 0-1 test; returns median K over deterministic c values."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 50:
        return np.nan
    x = (x - x.mean()) / (x.std() + 1e-30)
    n = len(x)
    # Avoid resonant c near zero and pi; deterministic reproducibility.
    cs = np.linspace(0.2, 2.8, c_count)
    ks = []
    j = np.arange(1, n // 10 + 1)
    for c in cs:
        phase = np.cumsum(x * 0 + c)  # phase n*c
        p = np.cumsum(x * np.cos(phase))
        q = np.cumsum(x * np.sin(phase))
        ms = np.array([np.mean((p[k:] - p[:-k]) ** 2 + (q[k:] - q[:-k]) ** 2) for k in j])
        if np.std(ms) > 0:
            ks.append(np.corrcoef(j, ms)[0, 1])
    return float(np.median(ks)) if ks else np.nan


def lyapunov_rosenstein(x, dt, max_points=2500):
    """Lightweight Rosenstein largest-exponent estimate (1/s), diagnostic only."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 100:
        return np.nan
    if len(x) > max_points:
        original_n = len(x)
        x = signal.resample(x, max_points)
        dt *= original_n / max_points
    # Normalize and use m=4, tau=2; exclude neighbors within one characteristic period.
    x = (x - x.mean()) / (x.std() + 1e-30)
    tau, m = 2, 4
    n = len(x) - (m - 1) * tau - 50
    if n < 50:
        return np.nan
    emb = np.column_stack([x[i * tau:i * tau + n] for i in range(m)])
    tree = cKDTree(emb)
    nn = np.full(n, -1, int)
    for i in range(n):
        for _, j in zip(*tree.query(emb[i], k=min(16, n))):
            if abs(int(j) - i) > max(10, int(0.02 / dt)):
                nn[i] = int(j); break
    valid = np.where(nn >= 0)[0]
    if len(valid) < 30:
        return np.nan
    horizon = min(40, n // 5)
    curve = np.zeros(horizon); count = np.zeros(horizon)
    for i in valid:
        upto = min(horizon, n - i, n - nn[i])
        if upto <= 1:
            continue
        dist = np.linalg.norm(emb[i:i + upto] - emb[nn[i]:nn[i] + upto], axis=1)
        curve[:upto] += np.log(np.maximum(dist, 1e-12)); count[:upto] += 1
    ok = count > 0
    meanlog = curve[ok] / count[ok]
    tt = np.arange(horizon)[ok] * dt
    fit = (tt >= 2 * dt) & (tt <= min(tt.max(), 0.05))
    if fit.sum() < 5:
        fit = np.arange(len(tt)) < min(12, len(tt))
    return float(np.polyfit(tt[fit], meanlog[fit], 1)[0]) if fit.sum() >= 3 else np.nan


def bursts(x, t, threshold_sigma=2.5):
    env = np.abs(signal.hilbert(np.asarray(x, float) - np.mean(x))) if len(x) >= 8 else np.abs(x)
    threshold = np.nanmean(env) + threshold_sigma * np.nanstd(env)
    ix, props = signal.find_peaks(env, height=threshold, distance=max(1, int(0.03 / np.median(np.diff(t)))))
    return env, t[ix], env[ix]


def add_row(rows, run, category, metric, value, units="", source="", status="ok", time=np.nan, frequency=np.nan):
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = np.nan
    rows.append({"run": run, "category": category, "metric": metric, "time_s": time, "frequency_hz": frequency,
                 "value": value, "units": units, "source": source, "status": status})


def plot_line(path, t, series, title, ylabel):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for label, y in series.items():
        ax.plot(t, y, label=label, lw=1)
    ax.set(xlabel="Time (s)", ylabel=ylabel, title=title)
    ax.grid(True, alpha=.25)
    if len(series) > 1: ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def analyze_run(run_dir: Path, args, rows):
    run = run_dir.name
    outdir = run_dir / "coupling"
    outdir.mkdir(parents=True, exist_ok=True)
    plots = outdir / "nonlinear_analysis_plots"
    plots.mkdir(exist_ok=True)
    transient = args.transient_fraction
    solid_files = list(run_dir.rglob("*.xdmf")) + list(run_dir.rglob("*.xmf"))
    solid_file = next((p for p in solid_files if "elastodynamics" in p.name.lower()), None)
    struct = None
    if solid_file:
        try:
            t, coords, state, field = xdmf_series(solid_file)
            order = np.argsort(t); t, state = t[order], state[order]
            # state fields in current writer: u,v,w,theta_x,theta_y
            nodes = nearest_nodes(coords)
            ncomp = state.shape[-1] if state.ndim == 3 else 1
            names = ["u", "v", "w", "rotation_x", "rotation_y"]
            signals = {}
            for loc, idx in nodes.items():
                for j in range(min(ncomp, len(names))):
                    signals[f"{loc}_{names[j]}"] = state[:, idx, j] if ncomp > 1 else state[:, idx]
            cut = int(len(t) * transient)
            ti = t[cut:]
            for name, val in signals.items():
                add_row(rows, run, "structure", name, np.nan, "m" if name.endswith(("_u", "_v", "_w")) else "rad", str(solid_file), "time_series")
                for tt, vv in zip(t, val): add_row(rows, run, "structure", name, vv, "m" if name.endswith(("_u", "_v", "_w")) else "rad", str(solid_file), "ok", tt)
            tip = signals.get("tip_w")
            if tip is not None:
                vel = finite_derivative(tip, t)
                for tt, vv in zip(t, vel): add_row(rows, run, "structure", "tip_w_velocity", vv, "m/s", str(solid_file), "derived_time_derivative", tt)
                plot_line(plots / "structure_response.png", t,
                          {k: v for k, v in signals.items() if k in ("tip_u", "tip_v", "tip_w", "mid_w", "leading_edge_w")},
                          "Structural displacement", "Displacement (m)")
                fig, ax = plt.subplots(figsize=(5, 5)); ax.plot(tip[cut:], vel[cut:], lw=.7); ax.set(xlabel="Tip w (m)", ylabel="Tip velocity (m/s)", title="Tip phase portrait"); ax.grid(True, alpha=.25); fig.tight_layout(); fig.savefig(plots / "phase_portrait.png", dpi=150); plt.close(fig)
                f, p, peak, _ = psd_metrics(tip[cut:], np.median(np.diff(t)))
                if len(f):
                    fig, ax = plt.subplots(figsize=(8, 4)); ax.semilogy(f, np.maximum(p, 1e-30)); ax.set(xlabel="Frequency (Hz)", ylabel="PSD (m²/Hz)", title="Tip displacement PSD"); ax.grid(True, alpha=.25); fig.tight_layout(); fig.savefig(plots / "tip_psd.png", dpi=150); plt.close(fig)
                    for ff, pp in zip(f, p): add_row(rows, run, "spectrum", "tip_w_psd", pp, "m²/Hz", str(solid_file), "welch", frequency=ff)
                env, bt, bv = bursts(tip[cut:], ti)
                plot_line(plots / "tip_envelope.png", ti, {"Hilbert amplitude": env}, "Tip displacement amplitude envelope", "Amplitude (m)")
                for tt, vv in zip(ti, env): add_row(rows, run, "structure_nonlinear", "tip_hilbert_envelope", vv, "m", str(solid_file), "hilbert", tt)
                burst_mask = env > (np.nanmean(env) + 2.5 * np.nanstd(env))
                padded = np.r_[False, burst_mask, False].astype(int)
                changes = np.diff(padded)
                burst_starts, burst_ends = np.where(changes == 1)[0], np.where(changes == -1)[0]
                durations = (burst_ends - burst_starts) * np.median(np.diff(ti)) if len(ti) > 1 else np.array([])
                for start_ix, end_ix, duration in zip(burst_starts, burst_ends, durations):
                    add_row(rows, run, "intermittency", "tip_burst_duration", duration, "s", str(solid_file), "envelope_threshold_2_5sigma", ti[start_ix])
                if len(durations):
                    fig, ax = plt.subplots(figsize=(6, 4)); ax.hist(durations, bins="auto"); ax.set(xlabel="Burst duration (s)", ylabel="Count", title="Tip envelope burst-duration distribution"); ax.grid(True, alpha=.25); fig.tight_layout(); fig.savefig(plots / "burst_duration_histogram.png", dpi=150); plt.close(fig)
                freq_gust = args.gust_frequency
                period = 1 / freq_gust if freq_gust > 0 else np.nan
                poincare_t = np.arange(ti[0] + period, ti[-1], period) if len(ti) and np.isfinite(period) else []
                poincare = np.interp(poincare_t, t, tip) if len(poincare_t) else np.array([])
                fig, ax = plt.subplots(figsize=(6, 4)); ax.plot(poincare_t, poincare, "."); ax.set(xlabel="Time (s)", ylabel="Tip w (m)", title=f"Poincaré samples ({freq_gust:g} Hz gust)"); ax.grid(True, alpha=.25); fig.tight_layout(); fig.savefig(plots / "poincare.png", dpi=150); plt.close(fig)
                for tt, vv in zip(poincare_t, poincare): add_row(rows, run, "structure_nonlinear", "poincare_tip_w", vv, "m", str(solid_file), "gust_period_sample", tt)
                det, lam, rr = recurrence_metrics(tip[cut:])
                metrics = [("dominant_frequency", peak, "Hz"), ("tip_rms", np.sqrt(np.mean((tip[cut:] - np.mean(tip[cut:])) ** 2)), "m"),
                           ("tip_peak_to_peak", np.ptp(tip[cut:]), "m"), ("rqa_determinism", det, "fraction"),
                           ("rqa_laminar_fraction", lam, "fraction"), ("recurrence_rate", rr, "fraction"),
                           ("zero_one_K", zero_one_test(tip[cut:]), "dimensionless"),
                           ("largest_lyapunov_rosenstein", lyapunov_rosenstein(tip[cut:], np.median(np.diff(t))), "1/s"),
                           ("burst_count", len(bt), "count"), ("poincare_sample_count", len(poincare), "count")]
                for name, val, unit in metrics: add_row(rows, run, "structure_nonlinear", name, val, unit, str(solid_file), "diagnostic_estimate")
                for tt, vv in zip(bt, bv): add_row(rows, run, "structure_nonlinear", "tip_envelope_burst_peak", vv, "m", str(solid_file), "ok", tt)
                if len(ti) > 100:
                    win = max(32, len(ti) // 4)
                    for wi, start in enumerate(range(0, len(ti) - win + 1, win)):
                        _, _, fp, _ = psd_metrics(tip[cut + start:cut + start + win], np.median(np.diff(t)))
                        add_row(rows, run, "stationarity", f"window_{wi}_dominant_frequency", fp, "Hz", str(solid_file), "windowed_spectrum")
            struct = (t, signals, nodes, state, coords)
        except Exception as exc:
            add_row(rows, run, "structure", "load_error", np.nan, "", str(solid_file), f"error: {exc}")
    else:
        add_row(rows, run, "structure", "structural_timeseries", np.nan, "", "", "missing_elastodynamics_xdmf")

    # Coupling CSV: force coefficients, transfer residuals, and actual stage times.
    coupling_csv = next(iter(run_dir.rglob("coupling_history.csv")), None)
    cdf = None
    if coupling_csv:
        try:
            cdf = pd.read_csv(coupling_csv)
            for col in cdf.columns:
                if col in ("step", "time") or not pd.api.types.is_numeric_dtype(cdf[col]): continue
                unit = "s" if "time" in col else "dimensionless" if col.lower() in ("cl", "cd", "force_relative_error", "geometry_relative_error") else ""
                for _, r in cdf[["time", col]].iterrows():
                    add_row(rows, run, "coupling", col, r[col], unit, str(coupling_csv), "ok", r["time"])
            for col in cdf.columns:
                if col in ("step", "time") or not pd.api.types.is_numeric_dtype(cdf[col]): continue
                add_row(rows, run, "coupling_summary", f"{col}_mean", cdf[col].mean(), "", str(coupling_csv))
                add_row(rows, run, "coupling_summary", f"{col}_max_abs", cdf[col].abs().max(), "", str(coupling_csv))
            if "time" in cdf:
                for col in ["lift", "drag", "cl", "cd", "cm", "sample_fz"]:
                    if col in cdf:
                        plot_line(plots / f"{col}_history.png", cdf.time, {col: cdf[col]}, f"{col} history", col)
                        if len(cdf) >= 8:
                            load_env = np.abs(signal.hilbert(cdf[col].to_numpy(float) - cdf[col].mean()))
                            env_threshold = np.nanmean(load_env) + 2.5 * np.nanstd(load_env)
                            peak_ix, _ = signal.find_peaks(load_env, height=env_threshold)
                            plot_line(plots / f"{col}_envelope.png", cdf.time, {"Hilbert amplitude": load_env}, f"{col} amplitude envelope", col)
                            for tt, vv in zip(cdf.time, load_env): add_row(rows, run, "load_envelope", f"{col}_hilbert_envelope", vv, col, str(coupling_csv), "hilbert", tt)
                            for ix in peak_ix: add_row(rows, run, "load_bursts", f"{col}_burst_peak", load_env[ix], col, str(coupling_csv), "threshold_2_5sigma", cdf.time.iloc[ix])
                        _, _, peak, _ = psd_metrics(cdf[col].to_numpy(), np.median(np.diff(cdf.time)))
                        add_row(rows, run, "loads", f"{col}_dominant_frequency", peak, "Hz", str(coupling_csv), "diagnostic_estimate")
                        ff, pp, _, _ = psd_metrics(cdf[col].to_numpy(), np.median(np.diff(cdf.time)))
                        # Recompute the spectral density itself for output rows.
                        if len(ff):
                            freq, power = signal.welch(cdf[col].to_numpy(float) - cdf[col].mean(), fs=1 / np.median(np.diff(cdf.time)), nperseg=min(4096, len(cdf)))
                            for fbin, pbin in zip(freq, power): add_row(rows, run, "spectrum", f"{col}_psd", pbin, f"{col}²/Hz", str(coupling_csv), "welch", frequency=fbin)
                if struct and "tip_w" in struct[1]:
                    st, ss = struct[0], struct[1]
                    tip = np.interp(cdf.time, st, ss["tip_w"])
                    dt_load = np.median(np.diff(cdf.time)) if len(cdf) > 1 else np.nan
                    for col in ("lift", "sample_fz"):
                        if col not in cdf: continue
                        force = cdf[col].to_numpy(float)
                        fig, ax = plt.subplots(figsize=(5, 4)); ax.plot(tip, force, lw=.7); ax.set(xlabel="Tip w (m)", ylabel=col, title=f"Force-displacement loop: {col}"); ax.grid(True, alpha=.25); fig.tight_layout(); fig.savefig(plots / f"{col}_lissajous.png", dpi=150); plt.close(fig)
                        work = np.trapz(force, tip)
                        add_row(rows, run, "loads", f"{col}_displacement_loop_integral", work, "J/m (force convention dependent)", str(coupling_csv), "signed_loop_integral")
                        # Estimate phase lag from cross spectrum; positive means force lags tip.
                        if len(force) > 16:
                            freq, csd = signal.csd(tip - np.mean(tip), force - np.mean(force), fs=1 / dt_load, nperseg=min(1024, len(force)))
                            coherence_f, coh = signal.coherence(tip, force, fs=1 / dt_load, nperseg=min(1024, len(force)))
                            for fbin, value in zip(freq, csd):
                                add_row(rows, run, "cross_spectrum", f"tip_w_{col}_cross_spectrum_real", np.real(value), "mixed/Hz", str(coupling_csv), "csd", frequency=fbin)
                                add_row(rows, run, "cross_spectrum", f"tip_w_{col}_cross_spectrum_imag", np.imag(value), "mixed/Hz", str(coupling_csv), "csd", frequency=fbin)
                            for fbin, value in zip(coherence_f, coh): add_row(rows, run, "cross_spectrum", f"tip_w_{col}_coherence", value, "fraction", str(coupling_csv), "welch_coherence", frequency=fbin)
                            if len(csd) > 1:
                                band = np.argmax(np.abs(csd[1:])) + 1
                                phase = float(np.angle(csd[band]))
                                add_row(rows, run, "loads", f"tip_w_{col}_phase_lag", phase, "rad", str(coupling_csv), f"at_{freq[band]:g}_Hz")
                        # Work per gust cycle, computed on phase-aligned cycle intervals.
                        period = 1 / args.gust_frequency if args.gust_frequency > 0 else np.nan
                        if np.isfinite(period):
                            starts = np.arange(cdf.time.iloc[0], cdf.time.iloc[-1] - period, period)
                            for cycle_no, start in enumerate(starts):
                                mask = (cdf.time >= start) & (cdf.time < start + period)
                                if mask.sum() >= 3:
                                    cycle_work = np.trapz(force[mask], tip[mask])
                                    add_row(rows, run, "loads", f"{col}_work_per_gust_cycle", cycle_work, "J/m (force convention dependent)", str(coupling_csv), f"cycle_{cycle_no}", start)
        except Exception as exc:
            add_row(rows, run, "coupling", "csv_load_error", np.nan, "", str(coupling_csv), f"error: {exc}")
    else:
        add_row(rows, run, "coupling", "coupling_history", np.nan, "", "", "missing")

    station_csv = next(iter(run_dir.rglob("*_aerodynamic_station_history.csv")), None)
    if station_csv:
        try:
            adf = pd.read_csv(station_csv)
            coefficient_names = ("cl_section", "cd_section", "cm_section")
            for col in adf.columns:
                if col in ("step", "time_s", "station") or not pd.api.types.is_numeric_dtype(adf[col]):
                    continue
                unit = "m" if col.endswith("_m") else "m/s²" if "circulation" in col else "N" if col.endswith("_N") else "N m" if col.endswith("_Nm") else ""
                for _, r in adf[["time_s", "station", col]].iterrows():
                    add_row(rows, run, "aerodynamic_station", f"station_{int(r['station'])}_{col}", r[col], unit, str(station_csv), "simulation_sample", r["time_s"])
            for coef in coefficient_names:
                if coef not in adf: continue
                fig, ax = plt.subplots(figsize=(9, 4.5))
                for station in sorted(adf.station.unique())[::max(1, len(adf.station.unique()) // 6)]:
                    part = adf[adf.station == station]
                    ax.plot(part.time_s, part[coef], lw=.8, label=f"station {int(station)}")
                ax.set(xlabel="Time (s)", ylabel=coef, title=f"Sectional {coef} histories"); ax.grid(True, alpha=.25); ax.legend(ncol=2); fig.tight_layout(); fig.savefig(plots / f"{coef}_stations.png", dpi=150); plt.close(fig)
                for station, part in adf.groupby("station"):
                    if len(part) < 8: continue
                    freq, power, peak, _ = psd_metrics(part[coef].to_numpy(), np.median(np.diff(part.time_s)))
                    add_row(rows, run, "aerodynamic_station_spectrum", f"station_{int(station)}_{coef}_dominant_frequency", peak, "Hz", str(station_csv), "diagnostic_estimate")
                    for fb, pb in zip(freq, power): add_row(rows, run, "spectrum", f"station_{int(station)}_{coef}_psd", pb, f"{coef}²/Hz", str(station_csv), "welch", frequency=fb)
        except Exception as exc:
            add_row(rows, run, "aerodynamic_station", "station_history_read_error", np.nan, "", str(station_csv), f"error: {exc}")

    # Direct simulation-time structural samples (state, velocity, acceleration,
    # strain/stress extrema, energy, and Newton convergence diagnostics).
    state_csv = next(iter(run_dir.rglob("solid_state_history.csv")), None)
    if state_csv:
        try:
            sdf = pd.read_csv(state_csv)
            for col in sdf.columns:
                if col in ("step", "time_s") or not pd.api.types.is_numeric_dtype(sdf[col]):
                    continue
                rotational = "theta_" in col
                unit = ("J" if col.endswith("_J") else "N/m" if "stress" in col else
                        ("rad" if rotational else "m") if "_displacement" in col else
                        ("rad/s" if rotational else "m/s") if "_velocity" in col else
                        ("rad/s²" if rotational else "m/s²") if "_acceleration" in col else "")
                cat = "solid_nonlinear_solver" if col.startswith("newton_") else "solid_energy" if col.startswith("E_") else "structure_state"
                for _, r in sdf[["time_s", col]].iterrows():
                    add_row(rows, run, cat, col, r[col], unit, str(state_csv), "simulation_sample", r["time_s"])
            for metric in ("newton_iterations", "newton_residual_norm", "newton_relative_residual", "membrane_stress_max_N_per_m", "surface_strain_max"):
                if metric in sdf:
                    plot_line(plots / f"{metric}.png", sdf.time_s, {metric: sdf[metric]}, metric.replace("_", " "), metric)
            if "E_total_J" in sdf and len(sdf):
                drift = float(sdf.E_total_J.iloc[-1] - sdf.E_total_J.iloc[0])
                add_row(rows, run, "solid_energy", "total_energy_drift", drift, "J", str(state_csv), "last_minus_first")
                add_row(rows, run, "solid_energy", "total_energy_relative_drift", drift / max(abs(float(sdf.E_total_J.iloc[0])), 1.0e-30), "fraction", str(state_csv), "last_minus_first_over_initial")
            disp_cols = [c for c in ("tip_u_displacement", "tip_v_displacement", "tip_w_displacement",
                                      "midspan_w_displacement", "leading_edge_midspan_w_displacement") if c in sdf]
            if disp_cols:
                plot_line(plots / "structural_displacements_simulation_csv.png", sdf.time_s,
                          {c.replace("_displacement", ""): sdf[c] for c in disp_cols},
                          "Structural displacement histories (simulation samples)", "Displacement (m)")
            if "tip_w_displacement" in sdf and "tip_w_velocity" in sdf:
                cut_state = int(len(sdf) * transient)
                fig, ax = plt.subplots(figsize=(5, 5)); ax.plot(sdf.tip_w_displacement.iloc[cut_state:], sdf.tip_w_velocity.iloc[cut_state:], lw=.7)
                ax.set(xlabel="Tip w (m)", ylabel="Tip velocity (m/s)", title="Tip phase portrait (solver velocity)"); ax.grid(True, alpha=.25)
                fig.tight_layout(); fig.savefig(plots / "phase_portrait_solver_velocity.png", dpi=150); plt.close(fig)
        except Exception as exc:
            add_row(rows, run, "structure", "state_history_read_error", np.nan, "", str(state_csv), f"error: {exc}")
    else:
        add_row(rows, run, "structure", "solid_state_history", np.nan, "", "", "missing")

    probe_csv = next(iter(run_dir.rglob("*_wake_probe_history.csv")), None)
    if probe_csv:
        try:
            pdf = pd.read_csv(probe_csv)
            for col in pdf.columns:
                if col in ("step", "time_s") or not pd.api.types.is_numeric_dtype(pdf[col]):
                    continue
                unit = "m/s" if col.endswith("_m_s") else "m" if col.endswith("_m") else ""
                for _, r in pdf[["time_s", col]].iterrows():
                    add_row(rows, run, "flow_probe", col, r[col], unit, str(probe_csv), "fixed_spatial_probe", r["time_s"])
                if col.endswith("_speed_m_s"):
                    f, p, peak, _ = psd_metrics(pdf[col].to_numpy(), np.median(np.diff(pdf.time_s))) if len(pdf) > 1 else ([], [], np.nan, np.nan)
                    add_row(rows, run, "flow_probe", col + "_dominant_frequency", peak, "Hz", str(probe_csv), "diagnostic_estimate")
                    if np.isfinite(peak):
                        chord = config_number(Path("config/solid_params.yaml"), "root_chord", np.nan)
                        vref = config_number(Path("config/fluid_params.yaml"), "vinf", np.nan)
                        st = peak * chord / vref if np.isfinite(chord) and np.isfinite(vref) and vref != 0 else np.nan
                        add_row(rows, run, "flow_probe", col + "_strouhal", st, "dimensionless", str(probe_csv), "fD_over_V")
                    if len(f):
                        for fb, pb in zip(f, p): add_row(rows, run, "spectrum", col + "_psd", pb, "(m/s)²/Hz", str(probe_csv), "welch", frequency=fb)
                    values = pdf[col].to_numpy(float)
                    values = values[np.isfinite(values)]
                    if len(values) > 2:
                        values = values - np.mean(values)
                        corr = signal.correlate(values, values, mode="full", method="fft")[len(values)-1:]
                        corr = corr / max(corr[0], 1e-30)
                        lag_dt = float(np.median(np.diff(pdf.time_s)))
                        max_lag = min(len(corr), max(2, int(2.0 / lag_dt)))
                        for lag, value in enumerate(corr[:max_lag]):
                            add_row(rows, run, "flow_probe_autocorrelation", col, value, "fraction", str(probe_csv), "normalized", lag * lag_dt)
                    plot_line(plots / f"{col}.png", pdf.time_s, {col: pdf[col]}, col.replace("_", " "), "m/s")
        except Exception as exc:
            add_row(rows, run, "flow_probe", "probe_history_read_error", np.nan, "", str(probe_csv), f"error: {exc}")

    # Per-span force arrays in JSONL, when available. These are dimensional forces.
    force_jsonl = next(iter(run_dir.rglob("forces_sent_received_history.jsonl")), None)
    if force_jsonl:
        try:
            recs = []
            with force_jsonl.open() as fh:
                for line in fh:
                    try: recs.append(json.loads(line))
                    except json.JSONDecodeError: continue
            tt = np.array([r.get("time", np.nan) for r in recs])
            for key in ("force_received", "force_sent"):
                if not any(key in r for r in recs): continue
                arrays = [np.asarray(r.get(key, []), float) for r in recs]
                if not arrays or not arrays[0].ndim == 2: continue
                length = min(map(len, arrays)); arr = np.asarray([a[:length] for a in arrays])
                total = np.nansum(arr, axis=1)
                for comp, label in enumerate(("x", "y", "z")):
                    if comp >= total.shape[-1]: continue
                    plot_line(plots / f"{key}_{label}.png", tt, {f"total_{label}": total[:, comp]}, f"{key} total force {label}", "Force")
                    for tval, value in zip(tt, total[:, comp]): add_row(rows, run, "loads", f"{key}_total_{label}", value, "N", str(force_jsonl), "per_step_sum", tval)
                    add_row(rows, run, "loads_summary", f"{key}_total_{label}_rms", np.sqrt(np.nanmean(total[:, comp] ** 2)), "N", str(force_jsonl))
                # Span-station force histories (bounded number of rows and plot traces).
                for station in sorted(set([0, length // 4, length // 2, 3 * length // 4, length - 1])):
                    for comp, label in enumerate(("x", "y", "z")):
                        if comp >= arr.shape[-1]: continue
                        for tval, value in zip(tt, arr[:, station, comp]):
                            add_row(rows, run, "loads_station", f"{key}_station_{station}_{label}", value, "N", str(force_jsonl), "per_step", tval)
        except Exception as exc:
            add_row(rows, run, "loads", "force_jsonl_error", np.nan, "", str(force_jsonl), f"error: {exc}")

    # Particle fields: use saved particle velocity/vorticity/sigma/circulation.
    # The default stride limits I/O for large wake dumps; set --particle-stride 1 for all.
    particle_files = sorted(run_dir.rglob("*pfield.*.xmf"), key=lambda p: int(re.search(r"\.(\d+)\.xmf$", p.name).group(1)) if re.search(r"\.(\d+)\.xmf$", p.name) else -1)
    particle_files = [p for p in particle_files if "staticpfield" not in p.name.lower()]
    if particle_files:
        selected = particle_files[::args.particle_stride]
        sampled_times, enstrophy, circ, wake_lengths, probe_series = [], [], [], [], {"probe_1": [], "probe_2": [], "probe_3": []}
        for pf in selected:
            try:
                root = ET.parse(pf).getroot()
                items = {}
                for a in root.iter():
                    if a.tag.endswith("Attribute"):
                        di = next((n for n in a.iter() if n.tag.endswith("DataItem")), None)
                        if di is not None and di.text and ":" in di.text:
                            hname, ds = di.text.strip().split(":", 1); items[a.attrib.get("Name", "").lower()] = (pf.parent / hname, ds)
                geom = next(n for n in root.iter() if n.tag.endswith("Geometry"))
                di = next(n for n in geom.iter() if n.tag.endswith("DataItem")); hname, ds = di.text.strip().split(":", 1)
                with h5py.File(pf.parent / hname, "r") as h5:
                    xyz = h5[ds][...]
                    def read_field(aliases):
                        for alias in aliases:
                            if alias in items:
                                return h5[items[alias][1]][...]
                        return None
                    vort = read_field(["vorticity"])
                    gam = read_field(["circulation", "gamma"])
                    sig = read_field(["sigma"]); vel = read_field(["velocity"])
                    vol = read_field(["vol"])
                    grad_x = read_field(["velocity gradient x"])
                    grad_y = read_field(["velocity gradient y"])
                    grad_z = read_field(["velocity gradient z"])
                stepm = re.search(r"\.(\d+)\.xmf$", pf.name); step = int(stepm.group(1)) if stepm else len(sampled_times)
                dt = args.dt
                pt = step * dt
                sampled_times.append(pt)
                if vort is not None:
                    omega2 = np.sum(vort * vort, axis=1)
                    weights = np.asarray(vol).reshape(-1) if vol is not None else np.ones(len(omega2))
                    enstrophy.append(0.5 * float(np.nansum(omega2 * weights)))
                else: enstrophy.append(np.nan)
                if gam is not None: circ.append(float(np.nansum(gam)))
                else: circ.append(np.nan)
                wake_lengths.append(float(np.ptp(xyz[:, 0])) if len(xyz) else np.nan)
                qvals = None
                if grad_x is not None and grad_y is not None and grad_z is not None:
                    # Gradient vectors are the x/y/z derivative columns of velocity.
                    grad = np.stack((grad_x, grad_y, grad_z), axis=-1)
                    strain = 0.5 * (grad + np.swapaxes(grad, 1, 2))
                    rotation = 0.5 * (grad - np.swapaxes(grad, 1, 2))
                    qvals = 0.5 * (np.sum(rotation * rotation, axis=(1, 2)) - np.sum(strain * strain, axis=(1, 2)))
                    add_row(rows, run, "flow", "q_criterion_max", np.nanmax(qvals), "1/s²", str(pf), "particle_field_snapshot", pt)
                    add_row(rows, run, "flow", "q_criterion_mean", np.nanmean(qvals), "1/s²", str(pf), "particle_field_snapshot", pt)
                if vort is not None:
                    add_row(rows, run, "flow", "vorticity_magnitude_max", np.nanmax(np.sqrt(np.sum(vort * vort, axis=1))), "1/s", str(pf), "particle_field_snapshot", pt)
                # Probe values via 5 nearest particle velocities; existing XDMF field velocity.
                if vel is not None and len(xyz):
                    for key, target in zip(probe_series, ((1.05, 0, 0), (1.25, 0, 0), (1.5, 0, 0))):
                        k = min(5, len(xyz)); inds = cKDTree(xyz).query(target, k=k)[1]
                        probe_series[key].append(float(np.nanmean(np.linalg.norm(vel[inds], axis=-1))))
                    for key in probe_series:
                        if len(probe_series[key]) < len(sampled_times): probe_series[key].append(np.nan)
                else:
                    for key in probe_series: probe_series[key].append(np.nan)
            except Exception as exc:
                add_row(rows, run, "flow", "particle_snapshot_error", np.nan, "", str(pf), f"error: {exc}")
        for met, vals, unit in (("particle_enstrophy_proxy", enstrophy, "vorticity² sum"), ("total_particle_circulation", circ, "circulation"), ("wake_length_x", wake_lengths, "m")):
            for tt, val in zip(sampled_times, vals): add_row(rows, run, "flow", met, val, unit, "particle XMF/HDF", "sampled_snapshot", tt)
            if len(vals) and np.any(np.isfinite(vals)): plot_line(plots / f"{met}.png", sampled_times, {met: vals}, met.replace("_", " "), unit)
        for key, vals in probe_series.items():
            for tt, val in zip(sampled_times, vals): add_row(rows, run, "flow_probe", key + "_speed_nearest_particles", val, "m/s", "particle velocity; nearest 5 particle values", "proxy_not_fixed_spatial_probe", tt)
            if len(vals) and np.any(np.isfinite(vals)):
                plot_line(plots / f"{key}.png", sampled_times, {key: vals}, f"{key} nearest-particle speed proxy", "m/s")
    else:
        add_row(rows, run, "flow", "particle_snapshots", np.nan, "", "", "missing")

    # Optional requested quantities that require outputs absent from many runs.
    stress = next(iter(run_dir.rglob("membrane_stress.pvd")), None)
    solid_diag = next(iter(run_dir.rglob("solid_v18_diagnostics.csv")), None)
    for name, file in (("membrane_stress", stress), ("solid_newton_diagnostics", solid_diag)):
        add_row(rows, run, "availability", name, 1 if file else np.nan, "present", str(file) if file else "", "available" if file else "missing_from_run")
    if solid_diag:
        try:
            sdf = pd.read_csv(solid_diag)
            for col in sdf.select_dtypes(include=[np.number]):
                add_row(rows, run, "solid_diagnostics", f"{col}_mean", sdf[col].mean(), "", str(solid_diag))
                add_row(rows, run, "solid_diagnostics", f"{col}_max", sdf[col].max(), "", str(solid_diag))
        except Exception as exc: add_row(rows, run, "solid_diagnostics", "read_error", np.nan, "", str(solid_diag), f"error: {exc}")

    # Run parameter metadata for sweep comparisons.
    for key, default, unit in (("vinf", np.nan, "m/s"), ("freestream_frequency_hz", args.gust_frequency, "Hz"), ("gust_amplitude", np.nan, "m/s")):
        found = np.nan
        for cfg in [Path("config/fluid_params.yaml"), Path("config/coupling_params.yaml")]:
            found = config_number(cfg, key, found)
        metric = "gust_frequency" if key == "freestream_frequency_hz" else key
        add_row(rows, run, "run_metadata", metric, found, unit, "config/*.yaml", "sweep_parameter")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("results/rect_nonlinear_regime"), help="single run directory")
    parser.add_argument("--output", type=Path, default=None, help="output CSV (default: each run's coupling/nonlinear_analysis.csv)")
    parser.add_argument("--transient-fraction", type=float, default=.4, help="discard this fraction for nonlinear summary metrics")
    parser.add_argument("--gust-frequency", type=float, default=.7, help="gust frequency (Hz) for Poincare sampling")
    parser.add_argument("--dt", type=float, default=.001, help="particle snapshot time increment (s)")
    parser.add_argument("--particle-stride", type=int, default=10, help="analyze every Nth particle snapshot; 1 reads all")
    parser.add_argument("--sweep-root", type=Path, default=None, help="analyze each child run directory for bifurcation plots")
    args = parser.parse_args()
    if not 0 <= args.transient_fraction < 1: parser.error("--transient-fraction must be in [0,1)")
    if args.particle_stride < 1: parser.error("--particle-stride must be >= 1")
    runs = sorted([p for p in args.sweep_root.iterdir() if p.is_dir()]) if args.sweep_root else [args.run_dir]
    if not runs:
        parser.error("No run directories found")
    all_rows = []
    for run in runs:
        if not run.exists():
            print(f"warning: run directory not found: {run}", file=sys.stderr)
            add_row(all_rows, run.name, "run", "run_directory", np.nan, "", str(run), "missing")
            continue
        print(f"Analyzing {run}")
        analyze_run(run.resolve(), args, all_rows)
    df = pd.DataFrame(all_rows, columns=["run", "category", "metric", "time_s", "frequency_hz", "value", "units", "source", "status"])
    if args.output:
        target = args.output
        target.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(target, index=False)
        print(f"Wrote {target} ({len(df)} rows)")
    else:
        # One combined output for sweep mode; otherwise conventional coupling location.
        if args.sweep_root:
            target = args.sweep_root / "nonlinear_analysis.csv"
            target.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(target, index=False)
            print(f"Wrote {target} ({len(df)} rows)")
        else:
            target = runs[0] / "coupling" / "nonlinear_analysis.csv"
            target.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(target, index=False)
            print(f"Wrote {target} ({len(df)} rows)")
    if args.sweep_root:
        make_sweep_plot(df, args.sweep_root)


def make_sweep_plot(df, root):
    meta = df[df.metric.isin(["vinf", "gust_amplitude"])].pivot_table(index="run", columns="metric", values="value", aggfunc="first")
    summaries = df[df.category.isin(["structure_nonlinear"]) & df.metric.isin(["tip_rms", "tip_peak_to_peak", "dominant_frequency", "largest_lyapunov_rosenstein", "rqa_laminar_fraction"])]
    if summaries.empty or meta.empty: return
    joined = summaries.merge(meta, left_on="run", right_index=True, how="left")
    xcol = "vinf" if "vinf" in joined and joined.vinf.notna().any() else "gust_amplitude"
    if xcol not in joined: return
    fig, axes = plt.subplots(3, 2, figsize=(10, 10)); axes = axes.ravel()
    for ax, metric in zip(axes, ["tip_rms", "tip_peak_to_peak", "dominant_frequency", "largest_lyapunov_rosenstein", "rqa_laminar_fraction"]):
        d = joined[joined.metric == metric].sort_values(xcol)
        ax.plot(d[xcol], d.value, "o-"); ax.set(xlabel=xcol, ylabel=metric); ax.grid(True, alpha=.25)
    axes[-1].axis("off"); fig.suptitle("Bifurcation / run sweep summary (one point per run)"); fig.tight_layout(); fig.savefig(root / "nonlinear_bifurcation_sweep.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
