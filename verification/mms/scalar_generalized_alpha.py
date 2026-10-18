"""Scalar oscillator verification using the production generalized-alpha updates.

This isolates time integration from FEM space, mesh projection, and nonlinear
plate solves. The forcing is continuous MMS forcing evaluated at the production
force time n+alpha_f; observed rates are measurements, not assumed guarantees.
"""

import csv
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "solid" / "utils"))
from generalized_alpha_coefficients import acceleration, intermediate, parameters, velocity


def exact(t, A=0.7, B=0.2, omega=3.1):
    q = A * math.sin(omega * t) + B * math.cos(2 * omega * t)
    v = A * omega * math.cos(omega * t) - 2 * omega * B * math.sin(2 * omega * t)
    a = -A * omega**2 * math.sin(omega * t) - 4 * omega**2 * B * math.cos(2 * omega * t)
    return q, v, a


def run_scalar_test(output_dir):
    mass, damping, stiffness = 1.7, 0.4, 12.0
    alpha_m, alpha_f = 0.1, 0.2
    gamma, beta = parameters(alpha_m, alpha_f)
    final_time = 0.8
    dts = [0.02 / (2**level) for level in range(6)]
    rows = []
    for dt in dts:
        nsteps = round(final_time / dt)
        q, v, a = exact(0.0)
        errors = []
        for step in range(1, nsteps + 1):
            t_alpha_f = step * dt - alpha_f * dt
            force = sum((mass * exact(t_alpha_f)[2],
                         damping * exact(t_alpha_f)[1],
                         stiffness * exact(t_alpha_f)[0]))

            # The equilibrium is linear in q_(n+1). Build its coefficient by
            # applying the exact same update helpers used by GeneralizedAlpha.
            a_const = acceleration(0.0, q, v, a, dt, beta)
            a_slope = acceleration(1.0, q, v, a, dt, beta) - a_const
            v_const = velocity(a_const, v, a, dt, gamma)
            v_slope = velocity(a_slope, 0.0, 0.0, dt, gamma)
            a_alpha_const = intermediate(a, a_const, alpha_m)
            a_alpha_slope = (1.0 - alpha_m) * a_slope
            v_alpha_const = intermediate(v, v_const, alpha_f)
            v_alpha_slope = (1.0 - alpha_f) * v_slope
            q_alpha_const = alpha_f * q
            q_alpha_slope = 1.0 - alpha_f
            lhs_const = mass*a_alpha_const + damping*v_alpha_const + stiffness*q_alpha_const
            lhs_slope = mass*a_alpha_slope + damping*v_alpha_slope + stiffness*q_alpha_slope
            q_new = (force - lhs_const) / lhs_slope
            a_new = acceleration(q_new, q, v, a, dt, beta)
            v_new = velocity(a_new, v, a, dt, gamma)
            q, v, a = q_new, v_new, a_new
            errors.append(q - exact(step * dt)[0])

        row = {
            "dt_s": dt,
            "steps": nsteps,
            "time_integrated_l2_error": math.sqrt(dt * sum(e*e for e in errors)),
            "final_time_absolute_error": abs(q - exact(final_time)[0]),
            "gamma": gamma,
            "beta": beta,
            "alpha_m": alpha_m,
            "alpha_f": alpha_f,
        }
        rows.append(row)

    for key in ("time_integrated_l2_error", "final_time_absolute_error"):
        for index, row in enumerate(rows):
            if index == 0:
                row[f"{key}_observed_order"] = ""
                continue
            coarse, fine = rows[index - 1][key], row[key]
            row[f"{key}_observed_order"] = (
                math.log(coarse / fine) / math.log(rows[index - 1]["dt_s"] / row["dt_s"])
                if coarse > 0.0 and fine > 0.0 and math.isfinite(coarse) and math.isfinite(fine)
                else ""
            )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "scalar_generalized_alpha.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
    for key, label in (("time_integrated_l2_error", "Time-integrated error"),
                       ("final_time_absolute_error", "Final-time error")):
        ax.loglog([row["dt_s"] for row in rows], [row[key] for row in rows], "o-", label=label)
    ax.set(xlabel="Time step dt [s]", ylabel="Absolute error", title="Scalar generalized-alpha verification")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.savefig(out / "scalar_generalized_alpha.png", dpi=180)
    plt.close(fig)
    return {"status": "PASS", "gamma": gamma, "beta": beta,
            "levels": rows, "csv": str(csv_path)}
