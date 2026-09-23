#!/usr/bin/env python3
"""
Compare the tip displacement from two different simulation runs:
- Old method: from the original sweep folder (dt_0.001)
- New method: from the reverse‑transfer test folder
Plots both on the same axes.
"""

# Diagnostics file for tip displacement comparison with the sweep plots.

from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml

# ---- matplotlib backend (for headless) ----
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

# ------------------- PATHS -------------------
# Adjust these to your actual folders
OLD_CASE = Path("/media/dysco/New Volume/Neeraj/varflexi solver/publish_result/results_solver/dt/dt_0.001")
NEW_CASE = Path("/home/dysco/FLOWUnsteady/VarFLEXI-rVPM-Fenics/results/dt_0.001_fig_reversetransfer")

# Fallback scales (used if config files are missing)
FALLBACK_CHORD = 0.12       # [m]
FALLBACK_UINF = 8.0         # [m/s]

# Plot style (matches your original)
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 8,
    "axes.labelsize": 9,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

# Output folder
OUTPUT_DIR = Path(__file__).resolve().parent / "plots_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------- HELPER FUNCTIONS -------------------
def load_tip_displacement(case_dir, has_results_subdir=True):
    """
    Load time and tip displacement from CSV.
    - If has_results_subdir=True, look for case_dir/results/solid/.../solid_v18_diagnostics.csv
    - Otherwise, look for case_dir/solid/.../solid_v18_diagnostics.csv
    """
    if has_results_subdir:
        path = case_dir / "results" / "solid" / "v18_reissner_mindlin_plate" / "solid_v18_diagnostics.csv"
        if not path.exists():
            matches = sorted((case_dir / "results").glob("solid/**/solid*_diagnostics.csv"))
            if matches:
                path = matches[0]
    else:
        path = case_dir / "solid" / "v18_reissner_mindlin_plate" / "solid_v18_diagnostics.csv"
        if not path.exists():
            matches = sorted((case_dir / "solid").glob("**/solid*_diagnostics.csv"))
            if matches:
                path = matches[0]

    if not path.exists():
        raise FileNotFoundError(f"Diagnostics CSV not found in {case_dir}")

    df = pd.read_csv(path)

    # Find time column
    time_col = next((col for col in df.columns if 'time' in col.lower()), None)
    if time_col is None:
        raise KeyError(f"No 'time' column. Available: {df.columns.tolist()}")

    # Find tip column
    tip_col = next((col for col in df.columns if 'tip' in col.lower()), None)
    if tip_col is None:
        raise KeyError(f"No 'tip' column. Available: {df.columns.tolist()}")

    return {
        "time": df[time_col].to_numpy(dtype=float),
        "tip":  df[tip_col].to_numpy(dtype=float),
    }


def load_yaml(path):
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def case_scales(case_dir):
    """Return reference chord and free-stream speed from config files (if present)."""
    # Look for config in the standard location
    solid_cfg = load_yaml(case_dir / "config" / "solid_params.yaml")
    fluid_cfg = load_yaml(case_dir / "config" / "fluid_params.yaml")

    root_chord = float(solid_cfg.get("root_chord", fluid_cfg.get("root_chord", FALLBACK_CHORD)))
    tip_chord = float(solid_cfg.get("tip_chord", fluid_cfg.get("tip_chord", root_chord)))
    c_ref = 0.5 * (root_chord + tip_chord)
    u_inf = float(fluid_cfg.get("vinf", FALLBACK_UINF))

    if c_ref <= 0.0:
        raise ValueError(f"Invalid chord scale c_ref={c_ref} for {case_dir}")
    if u_inf <= 0.0:
        raise ValueError(f"Invalid free-stream speed Uinf={u_inf} for {case_dir}")

    return c_ref, u_inf


def nondimensional_tip(case_dir, data):
    c_ref, u_inf = case_scales(case_dir)
    return {
        "t_nd": data["time"] * u_inf / c_ref,
        "tip_nd": data["tip"] / c_ref,
        "c_ref": c_ref,
        "u_inf": u_inf,
    }

# ------------------- LOAD AND COMPARE -------------------
def main():
    # ---- Load old method data ----
    print("Loading old method (dt=0.001)...")
    try:
        data_old = load_tip_displacement(OLD_CASE, has_results_subdir=True)
        nd_old = nondimensional_tip(OLD_CASE, data_old)
        print(f"Old: t_nd from {nd_old['t_nd'][0]:.2f} to {nd_old['t_nd'][-1]:.2f}, length {len(nd_old['t_nd'])}")
    except Exception as e:
        print(f"Error loading old data: {e}")
        nd_old = None

    # ---- Load new method data ----
    print("Loading new method (reverse transfer)...")
    try:
        data_new = load_tip_displacement(NEW_CASE, has_results_subdir=False)
        nd_new = nondimensional_tip(NEW_CASE, data_new)
        print(f"New: t_nd from {nd_new['t_nd'][0]:.2f} to {nd_new['t_nd'][-1]:.2f}, length {len(nd_new['t_nd'])}")
    except Exception as e:
        print(f"Error loading new data: {e}")
        nd_new = None

    if nd_old is None and nd_new is None:
        raise RuntimeError("Could not load any data.")

    # ---- Plot overlay ----
    fig, ax = plt.subplots(figsize=(4.5, 3), dpi=300)

    if nd_old is not None:
        t = nd_old["t_nd"]
        tip = nd_old["tip_nd"]
        step = max(1, len(t)//30)
        ax.plot(t, tip, color='#2C5596', linewidth=0.8,
                marker='o', markevery=step, markersize=2.5,
                markeredgewidth=0.3, markerfacecolor='white',
                label="Previous method (dt=0.001)")

    if nd_new is not None:
        t = nd_new["t_nd"]
        tip = nd_new["tip_nd"]
        step = max(1, len(t)//30)
        ax.plot(t, tip, color='#D62728', linewidth=0.8,
                marker='s', markevery=step, markersize=2.5,
                markeredgewidth=0.3, markerfacecolor='white',
                label="New method (reverse transfer)")

    ax.set_xlabel(r"$tU_\infty/c_{\mathrm{ref}}$")
    ax.set_ylabel(r"$s_{\mathrm{tip}}/c_{\mathrm{ref}}$")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    ax.legend(frameon=False, loc="best")

    fig.tight_layout()
    out_path = OUTPUT_DIR / "comparison_dt_0.001.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved comparison plot to {out_path}")
    plt.show()

if __name__ == "__main__":
    main()