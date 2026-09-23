# Diagnostic file 2 for comparison with work conservation error plot: force + geometry

#!/usr/bin/env python3
"""
Compare relative work-conservation error ε_W(t) between:
- RBF transfer operator (old method)
- New reverse-transfer method

Usage: edit the PATHS dictionary below with absolute paths.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------- USER: SET ABSOLUTE PATHS -------------------
PATHS = {
    "rbf": Path("/media/dysco/New Volume/Neeraj/varflexi solver/publish_result/results_solver/dt/dt_0.001"),  # <-- your RBF folder
    "new": Path("/home/dysco/FLOWUnsteady/VarFLEXI-rVPM-Fenics/results/dt_0.001_fig_reversetransfer"), # <-- your new folder
}
# -----------------------------------------------------------------

# ------------------- PLOT STYLE (same as original) -------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans"],
    "font.size": 7,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

COLORS = {
    "rbf": "#B61717",   # red (from original)
    "new": "#2CA02C",   # green
}

LINESTYLES = {
    "rbf": "--",        # dashed (from original)
    "new": "-.",        # dash-dot
}

# Moving-average window size (same as your original code)
WINDOW = 50

# ------------------- HELPER FUNCTIONS -------------------
def load_solid_diagnostics(base_dir):
    """
    Load solid diagnostics CSV. It expects a file like:
        base_dir/solid/v18_reissner_mindlin_plate/solid_v18_diagnostics.csv
    but falls back to a recursive search.
    """
    path = base_dir / "solid" / "v18_reissner_mindlin_plate" / "solid_v18_diagnostics.csv"
    if not path.exists():
        matches = list(base_dir.glob("**/solid*_diagnostics.csv"))
        if matches:
            path = matches[0]
        else:
            raise FileNotFoundError(f"No solid diagnostics CSV found in {base_dir}")
    df = pd.read_csv(path)
    # Ensure required columns exist
    required = ["time", "work_rel_error"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"{base_dir} CSV missing columns: {missing}. Available: {df.columns.tolist()}")
    return df

def moving_average(data, window_size=50):
    data = np.asarray(data, dtype=float)
    if window_size <= 1 or len(data) <= window_size:
        return data.copy()
    pad_left = window_size // 2
    pad_right = window_size - 1 - pad_left
    padded = np.pad(data, (pad_left, pad_right), mode="edge")
    kernel = np.ones(window_size) / window_size
    return np.convolve(padded, kernel, mode="valid")

# ------------------- LOAD DATA -------------------
data = {}
for method, base in PATHS.items():
    if not base.exists():
        print(f"Warning: path for {method} does not exist: {base}")
        continue
    try:
        df = load_solid_diagnostics(base)
        time = df["time"].to_numpy(dtype=float)
        error_raw = df["work_rel_error"].to_numpy(dtype=float)
        error_ma = moving_average(error_raw, WINDOW)
        data[method] = {
            "time": time,
            "error_raw": error_raw,
            "error_ma": error_ma,
        }
        print(f"Loaded {method}: time range [{time[0]:.2f}, {time[-1]:.2f}], length {len(time)}")
    except Exception as e:
        print(f"Failed to load {method}: {e}")

if not data:
    raise RuntimeError("No data loaded. Check your paths.")

# ------------------- PLOT -------------------
fig, ax = plt.subplots(figsize=(4.0, 2.4), dpi=300)
ax.set_yscale("log")

for method, d in data.items():
    time = d["time"]
    raw = d["error_raw"]
    ma = d["error_ma"]
    color = COLORS.get(method, "black")
    linestyle = LINESTYLES.get(method, "-")
    # Raw data (thin, transparent)
    ax.plot(time, raw,
            color=color, linestyle=linestyle,
            linewidth=0.4, alpha=0.25, zorder=1)
    # Moving average (thick, opaque)
    ax.plot(time, ma,
            color=color, linestyle=linestyle,
            linewidth=1.2, alpha=1.0, zorder=2,
            label=method.upper())

ax.set_xlabel(r"Time $t$ (s)", labelpad=4)
ax.set_ylabel(r"Relative Work Error, $\epsilon_W(t)$", labelpad=4)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(False)
ax.legend(frameon=False, loc="best", handletextpad=0.5, labelspacing=0.2)

fig.tight_layout()
out_file = Path(__file__).resolve().parent / "rbf_vs_new_work_cons.pdf"
fig.savefig(out_file, bbox_inches="tight", transparent=True)
print(f"Figure saved to {out_file}")
plt.show()