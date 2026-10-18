#!/usr/bin/env python3
"""Audit current solver scales and generate isolated refinement configurations.

No simulations are launched unless --execute is supplied. Generated cases
live below verification/resolution/runs and never edit config/ in place.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import re
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
CONFIG = ROOT / "config"
RESULTS = OUT / "results"
RUNS = OUT / "runs"


class InlineListDumper(yaml.SafeDumper):
    def represent_list(self, data):
        return self.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)


InlineListDumper.add_representer(list, InlineListDumper.represent_list)


def read_configs(config_dir=CONFIG):
    configs = {}
    for name in ("fluid_params.yaml", "solid_params.yaml", "coupling_params.yaml"):
        with open(Path(config_dir) / name) as stream:
            configs[name.split("_")[0]] = yaml.safe_load(stream)
    return configs


def calculate_scales(c):
    f, s, q = c["fluid"], c["solid"], c["coupling"]
    dt = float(q["total_time"]) / int(q["n_steps"])
    U = float(f["vinf"])
    pps = int(f["particles_per_step"])
    chord, span = float(s["root_chord"]), float(s["span"])
    hx, hy = chord / int(s["nx"]), span / int(s["ny"])
    nspan = int(f.get("n_span", q.get("comm_nspan", 1)))
    hvlm = span / nspan
    ncomm = int(q.get("comm_nspan", nspan))
    hc = span / ncomm
    hp = U * dt / pps
    sigma = float(f.get("lambda_vpm", 2.0)) * U * dt / pps
    nu = float(f["nu"])
    gust_frequency = float(f.get("freestream_frequency_hz", 0.0) or 0.0)
    gust_period = 1.0 / gust_frequency if gust_frequency > 0.0 else None
    h = chord * float(s["thickness_ratio"])
    D = float(s["E"]) * h**3 / (12 * (1 - float(s["nu"])**2))
    areal_mass = float(s["rho_s"]) * h
    beta1 = 1.875104068711961
    omega1_beam = beta1**2 * math.sqrt(D / areal_mass) / span**2
    T1_beam = 2 * math.pi / omega1_beam
    manifest = Path("/home/dysco/FLOWUnsteady/Manifest.toml")
    package_versions = {}
    if manifest.is_file():
        manifest_text = manifest.read_text()
        for package in ("FLOWUnsteady", "FLOWVLM", "FLOWVPM"):
            match = re.search(
                rf"(?ms)^\[\[deps\.{re.escape(package)}\]\]\s*(.*?)(?=^\[\[|\Z)",
                manifest_text,
            )
            version = re.search(r'^version\s*=\s*"([^"]+)"', match.group(1), re.M) if match else None
            package_versions[package] = version.group(1) if version else "unknown"
    return {
        "dt_s": dt, "U_inf_m_s": U, "rho_f_kg_m3": float(f["rho"]),
        "aoa_deg": float(f["aoa_deg"]), "nu_m2_s": nu,
        "Re_chord": U * chord / nu,
        "aspect_ratio_span_chord": span / chord,
        "h_s_chord_m": hx, "h_s_span_m": hy,
        "h_vlm_span_m": hvlm, "h_c_span_m": hc,
        "h_particle_release_m": hp, "sigma_particle_initial_m": sigma,
        "plate_thickness_m": h,
        "particles_per_step": pps,
        "n_vlm_span": int(f.get("n_span", 1)), "n_vlm_chord": int(f.get("n_chord", 1)),
        "n_coupling_span": ncomm,
        "C_p_Udt_over_hp": U * dt / hp,
        "Fo_p_nu_dt_over_hp2": nu * dt / hp**2,
        "Fo_sigma_nu_dt_over_sigma2": nu * dt / sigma**2,
        "particle_crossing_time_s": hp / U,
        "particle_diffusion_time_s": hp**2 / nu,
        "gust_period_s": gust_period,
        "beam_approx_fundamental_period_s": T1_beam,
        "beam_approx_frequency_hz": 1.0 / T1_beam,
        "fmm_p": int(f.get("p", 4)), "fmm_ncrit": int(f.get("ncrit", 50)),
        "fmm_theta": float(f.get("theta", 0.4)),
        "fmm_relative_tolerance": float(f.get("relative_tolerance", 1e-3)),
        "fmm_absolute_tolerance": float(f.get("absolute_tolerance", 1e-3)),
        "viscous_scheme": f.get("viscous_scheme", "CoreSpreading"),
        "integration": "FLOWVPM rungekutta3 (FLOWUnsteady default; confirmed in installed run_simulation signature)",
        "formulation": "FLOWVPM ReformulatedVPM (FLOWUnsteady default)",
        "package_versions": package_versions,
        "freestream": {"vinf_m_s": U, "aoa_deg": f.get("aoa_deg"),
                       "gust_amplitude_m_s": f.get("freestream_oscillation"),
                       "gust_wavenumber_rad_m": f.get("freestream_wavenumber"),
                       "gust_frequency_hz": f.get("freestream_frequency_hz")},
        "structural_elements": f"CG2 mixed (u_x,u_y,w,theta_x,theta_y), {s['nx']}x{s['ny']}",
        "material": {"E_Pa": s["E"], "nu": s["nu"], "rho_s_kg_m3": s["rho_s"],
                     "thickness_ratio": s["thickness_ratio"], "thickness_m": h},
        "generalized_alpha": {"alpha_m": s["alpha_m"], "alpha_f": s["alpha_f"],
                              "gamma": 0.5 + s["alpha_f"] - s["alpha_m"],
                              "beta": (0.5 + s["alpha_f"] - s["alpha_m"] + 0.5)**2 / 4},
        "rayleigh_damping": {"eta_m": s["eta_m"], "eta_k": s["eta_k"]},
        "coupling": {"force_transfer_mode": q["force_transfer_mode"],
                     "force_relax": q["force_relax"], "geom_relax": q["geom_relax"],
                     "subiterations": 1, "n_steps": q["n_steps"], "total_time_s": q["total_time"]},
        "wake": {"shed_unsteady": f.get("unsteady"), "shed_boundarylayer": f.get("shed_boundarylayer", False),
                 "start_with_wake": f.get("start_with_wake"), "remove_every_steps": f.get("remove_every"),
                 "sphere_factor": f.get("sphere_factor"), "strength_factors": [f.get("min_factor"), f.get("max_factor")],
                 "core_radius_factors": [f.get("min_factor"), f.get("max_factor")], "core_spread_beta": 1.5},
        "linearization_note": "Beam estimate uses a uniform cantilever strip, beta1=1.8751; not the mixed 2-D FEM eigenfrequency.",
    }


def audit_outputs(scales):
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "current_scales.json", "w") as stream:
        json.dump(scales, stream, indent=2)
    numeric = [(k, v) for k, v in scales.items() if isinstance(v, (int, float))]
    with open(RESULTS / "current_scales.csv", "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("quantity", "value", "units_or_definition"))
        units = {
            "dt_s": "s", "U_inf_m_s": "m/s", "nu_m2_s": "m^2/s",
            "h_s_chord_m": "m", "h_s_span_m": "m", "h_vlm_span_m": "m",
            "h_c_span_m": "m", "h_particle_release_m": "m", "sigma_particle_initial_m": "m",
            "gust_period_s": "s", "beam_approx_fundamental_period_s": "s",
            "beam_approx_frequency_hz": "Hz",
        }
        for key, value in numeric:
            writer.writerow((key, value, units.get(key, "dimensionless diagnostic")))

    report = f"""# Physics-Informed Resolution Audit

Generated from `config/fluid_params.yaml`, `solid_params.yaml`, and `coupling_params.yaml` by `python3 verification/resolution/analyze.py audit`.

## Current configuration inventory

| Quantity | Current value | Meaning |
|---|---:|---|
| Coupled time step | {scales['dt_s']:.6g} s | total time / steps |
| Chord Reynolds number | {scales['Re_chord']:.6g} | U c / nu |
| Fluid density / viscosity | {scales['rho_f_kg_m3']:.6g} kg/m^3 / {scales['nu_m2_s']:.6g} m^2/s | config |
| Angle of attack | {scales['aoa_deg']:.6g} deg | config |
| Prescribed gust | A={scales['freestream']['gust_amplitude_m_s']} m/s, k={scales['freestream']['gust_wavenumber_rad_m']} rad/m, f={scales['freestream']['gust_frequency_hz']} Hz | deterministic transverse sinusoid; not turbulence/LES |
| Plate thickness | {scales['plate_thickness_m']:.6g} m | root chord * thickness ratio |
| Structural element spacing | {scales['h_s_chord_m']:.6g} m chordwise; {scales['h_s_span_m']:.6g} m spanwise | rectangular mesh, mixed CG2 |
| VLM span station pitch | {scales['h_vlm_span_m']:.6g} m | span / n_span; no volumetric fluid mesh |
| Coupling station pitch | {scales['h_c_span_m']:.6g} m | span / comm_nspan |
| VLM discretization | {scales['n_vlm_span']} span x {scales['n_vlm_chord']} chord lattices | lifting-surface mesh; current one chordwise lattice |
| Coupling stations | {scales['n_coupling_span']} | independent `comm_nspan` |
| Wake release spacing estimate | {scales['h_particle_release_m']:.6g} m | U dt / particles_per_step for uniform stream-aligned release |
| Initial wake core radius | {scales['sigma_particle_initial_m']:.6g} m | lambda_vpm U dt / particles_per_step |
| Particle scale diagnostics | U*dt/hp={scales['C_p_Udt_over_hp']:.6g} (=particles_per_step under uniform release); Fo_p={scales['Fo_p_nu_dt_over_hp2']:.6g} | descriptive ratios, not stability limits |
| Synthetic gust period | {scales['gust_period_s']:.6g} s | prescribed config frequency |
| Cantilever-strip frequency estimate | {scales['beam_approx_frequency_hz']:.6g} Hz | preliminary beam approximation, not FEM eigenanalysis |
| Particle integrator / formulation | {scales['integration']} / {scales['formulation']} | FLOWUnsteady defaults at installed version |
| Installed package versions | {scales['package_versions']} | `/home/dysco/FLOWUnsteady/Manifest.toml` |
| Coupling | {scales['coupling']['force_transfer_mode']}, force/geometry relaxation {scales['coupling']['force_relax']}/{scales['coupling']['geom_relax']}, one pass | staggered, no within-step subiterations |

## Interpretation and candidate limits

There is no Eulerian fluid grid spacing in this rVPM calculation. Relevant spatial quantities are structural element sizes, VLM spanwise pitch, wake particle release spacing, particle core radius, and coupling station spacing. In particular, the installed FLOWUnsteady shed routine distributes `p_per_step` particles along the release segment and uses the specified core override; thus `h_p` and `sigma_p` both change with `dt` and `particles_per_step`.

The installed particle integrator is third-order low-storage Runge-Kutta by default, while core spreading updates `sigma^2` according to `d(sigma^2)/dt=2 nu` (with its RK stages), resetting the core when its growth ratio reaches the scheme's default beta=1.5 through its RBF vorticity-remapping procedure. This does not supply a universal CFL stability bound for the complete nonlinear, remapped particle system. `C_p` and `Fo_p` are recorded as scale indicators only. The default Gaussian-erf kernel uses `zeta(r)= (2 pi)^(-3/2) exp(-r^2/2)` in radius normalized by sigma; its radial integral is tested in `verification/mms/fluid_components.jl`.

The exact FEM modal spectrum is not yet measured. A screening time-step estimate is `min(T_gust,T_beam)/20 = {min(scales['gust_period_s'] or math.inf, scales['beam_approx_fundamental_period_s'])/20:.6g} s`, using 20 steps per estimated shortest period as a conservative *starting point for a time-refinement experiment*, not a stability limit or verified maximum. The current step is {scales['beam_approx_fundamental_period_s']/scales['dt_s']:.1f} steps per strip-mode period. The table below separates current scales from bounds needing empirical refinement:

| Parameter | Mechanism | Preliminary upper bound | Status / test required |
|---|---|---|---|
| `h_s,x`, `h_s,y` | Mixed CG2 plate approximation, bending/shear gradients | Not defensibly numeric from config alone; current {scales['h_s_chord_m']:.4g}/{scales['h_s_span_m']:.4g} m | Linearized FEM eigenmodes and spatial MMS/refinement required |
| `h_VLM` | Lifting-surface circulation and spanwise force sampling | No wavelength/benchmark convergence evidence yet; current {scales['h_vlm_span_m']:.4g} m | Refine n_span independently, compare Gamma/load/tip motion |
| `h_p` | Wake shedding and particle representation | Not a stability limit; current estimate {scales['h_particle_release_m']:.4g} m | Particle-resolution study; co-varying sigma must be recorded |
| `sigma_p` | Kernel regularization/core spreading | Initial {scales['sigma_particle_initial_m']:.4g} m | Compare velocity/vorticity and force outputs over sigma study |
| `h_c` | Geometry/force transfer sampling | Current {scales['h_c_span_m']:.4g} m | Transfer resultant/moment/virtual-work errors vs station count |
| `dt` | RK3 particle advance, generalized-alpha plate, staggered FSI | No formal coupled stability bound established; current {scales['dt_s']:.4g} s | Time refinement, with h_p and sigma_p changes explicitly tracked |

The strip estimate is `omega_1 ~= beta_1^2 sqrt(D/(rho_s h))/L^2`, with `D=E h^3/[12(1-nu^2)]`; it is only a preliminary physical period. The configured generalized-alpha parameters produce gamma={scales['generalized_alpha']['gamma']:.4g}, beta={scales['generalized_alpha']['beta']:.4g}; the spectral-radius convention is not specified, so no high-frequency damping claim is inferred. The partitioned algorithm has one force/geometry exchange per step; reducing `dt` is required to assess its temporal lag.

## Refinement

`python3 verification/resolution/analyze.py plan --study time` writes isolated configurations from 0.25x to 20x the current dt at fixed total time. It adjusts `particles_per_step` inversely with dt so `U dt / p_per_step` and the initial `sigma_vpm_overwrite` remain fixed (integer factors are exact for these levels); the manifest verifies this. The `panels` study changes VLM span stations only; the `transfer` study changes `comm_nspan` only. Other independent sweeps are `structural`, `particles`, and `fmm`. Cases are generated but never launched by default. `--execute` runs the entire selected study sequentially via `run.sh` and should be used only in the FEniCS/Julia environment. Afterward, `python3 verification/resolution/collect.py --study time` extracts tip-displacement, lift, transfer-error, and runtime convergence metrics against the finest completed case.

## Evidence classification

- **Scheme/source-derived:** the RK3 default, `p_per_step` particle placement, `sigma_vpm_overwrite`, core-spreading formula, mesh spaces, generalized-alpha formulas, damping form, and one-pass exchange are read directly from the repository and installed package sources.
- **Physics estimates:** release/core sizes, advective/diffusive indicators, and the strip-mode estimate above.
- **Empirical limits:** none yet. The active environment lacks `dolfin`, which prevents coupled runs and numerical FEM convergence. Julia config loading and FLOWVPM/FLOWVLM component checks succeed. No convergence is claimed.

Raw metrics and plot: `results/current_scales.csv`, `current_scales.json`, `resolution_scales.png`.
"""
    (RESULTS / "RESOLUTION_ANALYSIS.md").write_text(report)
    keys = ["h_s_chord_m", "h_s_span_m", "h_vlm_span_m", "h_c_span_m", "h_particle_release_m", "sigma_particle_initial_m"]
    fig, ax = plt.subplots(figsize=(8, 4.6), constrained_layout=True)
    vals = [scales[k] for k in keys]
    bars = ax.bar(range(len(keys)), vals, color=["#2678a5", "#36a87a", "#df9f35", "#875bb5", "#d65a4a", "#55728a"])
    ax.set_yscale("log")
    ax.set_ylabel("Length scale [m], logarithmic")
    ax.set_xticks(range(len(keys)), ["structural\nchord", "structural\nspan", "VLM\nspan", "coupling\nspan", "particle\nrelease", "particle\ncore"])
    ax.set_title("Current spatial scales (not a common fluid grid)")
    ax.grid(axis="y", which="both", alpha=0.25)
    for bar, value in zip(bars, vals):
        ax.annotate(f"{value:.2g}", (bar.get_x() + bar.get_width()/2, value), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8)
    fig.savefig(RESULTS / "resolution_scales.png", dpi=180)
    plt.close(fig)


def mutate_study(c, study, multiplier, tag):
    f, s, q = c["fluid"], c["solid"], c["coupling"]
    if study == "structural":
        s["nx"] = max(2, round(s["nx"] * multiplier))
        s["ny"] = max(2, round(s["ny"] * multiplier))
    elif study == "panels":
        n = max(4, round(f.get("n_span", q.get("comm_nspan", 80)) * multiplier))
        f["n_span"] = n
    elif study == "transfer":
        q["comm_nspan"] = max(4, round(q.get("comm_nspan", f.get("n_span", 80)) * multiplier))
    elif study == "particles":
        f["particles_per_step"] = max(1, round(f["particles_per_step"] * multiplier))
    elif study == "time":
        q["n_steps"] = max(1, round(q["n_steps"] * multiplier))
        f["particles_per_step"] = max(1, round(f["particles_per_step"] / multiplier))
    elif study == "fmm":
        f["relative_tolerance"] = float(multiplier)
        f["absolute_tolerance"] = float(multiplier)
    else:
        raise ValueError(study)
    c["solid"]["results_root"] = f"results/resolution_{study}/{tag}/"
    c["solid"]["output_dir"] = "solid/"
    c["coupling"]["results_root"] = f"results/resolution_{study}/{tag}/"
    f["save_directory"] = f"results/resolution_{study}/{tag}/fluid/"
    f["run_name"] = f"{study}_{tag}"


def plan_study(study, execute=False):
    base = read_configs()
    original = calculate_scales(base)
    if study == "time":
        cases = [(4.0, "dt_quarter"), (2.0, "dt_half"), (1.0, "baseline"),
                 (0.5, "dt2x"), (0.2, "dt5x"), (0.1, "dt10x"), (0.05, "dt20x")]
    elif study == "particles":
        cases = [(0.5, "pps_half"), (1.0, "baseline"), (2.0, "pps_double")]
    elif study == "fmm":
        cases = [(1e-2, "tol_1e-2"), (1e-3, "tol_1e-3"), (1e-4, "tol_1e-4")]
    else:
        cases = [(0.5, "coarse"), (1.0, "baseline"), (2.0, "fine")]
    manifest = []
    for value, tag in cases:
        c = json.loads(json.dumps(base))
        mutate_study(c, study, value, tag)
        derived = calculate_scales(c)
        case_dir = RUNS / study / tag
        config_dir = case_dir / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        for key, filename in (("fluid", "fluid_params.yaml"), ("solid", "solid_params.yaml"), ("coupling", "coupling_params.yaml")):
            with open(config_dir / filename, "w") as stream:
                yaml.dump(c[key], stream, Dumper=InlineListDumper, sort_keys=False)
        config_hash = hashlib.sha256()
        for filename in ("fluid_params.yaml", "solid_params.yaml", "coupling_params.yaml"):
            config_hash.update((config_dir / filename).read_bytes())
        manifest.append({"study": study, "case": tag, "factor_or_value": value,
                         "config_dir": str(config_dir), "results_dir": c["solid"]["results_root"],
                         "dt_s": derived["dt_s"],
                         "total_time_s": float(c["coupling"]["total_time"]),
                         "n_steps": int(c["coupling"]["n_steps"]),
                         "nx_solid": int(c["solid"]["nx"]), "ny_solid": int(c["solid"]["ny"]),
                         "n_vlm_span": int(c["fluid"].get("n_span", 1)),
                         "particles_per_step": int(c["fluid"]["particles_per_step"]),
                         "config_sha256": config_hash.hexdigest(),
                         "h_s_chord_m": derived["h_s_chord_m"], "h_s_span_m": derived["h_s_span_m"],
                         "h_vlm_span_m": derived["h_vlm_span_m"],
                         "h_particle_release_m": derived["h_particle_release_m"],
                         "sigma_particle_initial_m": derived["sigma_particle_initial_m"],
                         "C_p": derived["C_p_Udt_over_hp"], "Fo_p": derived["Fo_p_nu_dt_over_hp2"],
                         "execution": "NOT_RUN"})
        if execute:
            env = dict(os.environ, VARFLEXI_CONFIG_DIR=str(config_dir))
            result = subprocess.run([str(ROOT / "run.sh")], cwd=ROOT, env=env,
                                    check=False, capture_output=True, text=True)
            manifest[-1]["execution"] = "PASS" if result.returncode == 0 else f"FAILED({result.returncode})"
            manifest[-1]["returncode"] = result.returncode
            (case_dir / "run.log").write_text(
                "=== stdout ===\n" + result.stdout + "\n=== stderr ===\n" + result.stderr
            )
    out = RUNS / study
    with open(out / "manifest.csv", "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    with open(out / "manifest.json", "w") as stream:
        json.dump({"baseline_scales": original, "cases": manifest, "executed": bool(execute)}, stream, indent=2)
    print(f"Wrote {len(manifest)} {study} cases under {out}; executed={execute}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("audit")
    plan = sub.add_parser("plan")
    plan.add_argument("--study", choices=("structural", "panels", "particles", "time", "transfer", "fmm"), required=True)
    plan.add_argument("--execute", action="store_true", help="Sequentially launch run.sh for every generated case.")
    args = parser.parse_args()
    if args.command == "audit":
        scales = calculate_scales(read_configs())
        audit_outputs(scales)
        print(f"Wrote audit outputs to {RESULTS}")
    else:
        plan_study(args.study, args.execute)


if __name__ == "__main__":
    main()
