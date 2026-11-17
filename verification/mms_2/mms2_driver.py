"""Entry point for the isolated mms_2 verification suite.

The field fluid/coupled run commands intentionally fail closed until the
continuum-to-rVPM full-state source projection is integrated. Use `preflight`
for analytical/configuration checks and the copied structural mode for a real
standalone structural solve.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

from field_mms import analytical_check, configured_particle_freestream, freestream_match

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent


def production_config(config_dir: Path) -> tuple[dict, dict]:
    fluid = yaml.safe_load((config_dir / "fluid_params.yaml").read_text(encoding="utf-8"))
    solid = yaml.safe_load((config_dir / "solid_params.yaml").read_text(encoding="utf-8"))
    return fluid, solid


def run_preflight(config_dir: Path, out: Path) -> dict:
    fluid, solid = production_config(config_dir)
    field_config = yaml.safe_load((HERE / "field_mms_parameters.yaml").read_text(encoding="utf-8"))
    isolated_fluid_path = HERE / "fluid_mms_config.yaml"
    isolated_fluid = yaml.safe_load(isolated_fluid_path.read_text(encoding="utf-8"))
    expected = field_config["physics"]
    actual_viscous = str(fluid.get("viscous_scheme", "CoreSpreading")).strip().lower()
    actual_sfs = str(fluid.get("vpm_sfs", "none")).strip().lower()
    viscous_off = actual_viscous in ("none", "off", "false", "nothing")
    sfs_off = actual_sfs in ("none", "off", "false")
    report = analytical_check(
        HERE / "field_mms_parameters.yaml", fluid_config=fluid,
        geometry_lengths=(float(solid["root_chord"]), float(solid["span"])))
    final_time = float(report["field_temporal_final_time_s"])
    expected_freestream = configured_particle_freestream(fluid, final_time)
    report["freestream_convention_check"] = freestream_match(
        report["freestream_pfield_origin_m_s"], expected_freestream)
    report["production_config_check"] = {
        "fluid_config": str((config_dir / "fluid_params.yaml").resolve()),
        "solid_geometry_m": {"span": float(solid["span"]),
                             "root_chord": float(solid["root_chord"]),
                             "tip_chord": float(solid["tip_chord"])},
        "configured_viscous_scheme": fluid.get("viscous_scheme", "CoreSpreading"),
        "configured_vpm_sfs": fluid.get("vpm_sfs", "none"),
        "viscosity_disabled_by_config": viscous_off,
        "sfs_disabled_by_config": sfs_off,
        "effective_runtime_types_are_checked_by": "reconstruct_fields.jl",
        "expected_viscous_scheme": expected["viscous_scheme"],
        "expected_vpm_sfs": expected["vpm_sfs"],
        "isolated_mms_fluid_config": str(isolated_fluid_path.resolve()),
        "fixed_particle_lifecycle": {
            "status": "CONFIGURED_NOT_RUNTIME_AUDITED",
            "runtime_mapping": {
                "particles_per_step": "FluidConfig.jl p_per_step -> setup.jl shedding rate",
                "start_with_wake": "FluidConfig.jl shed_starting -> setup.jl starting wake",
                "unsteady": "FluidConfig.jl shed_unsteady -> setup.jl unsteady shedding",
                "wake_coupled": "FluidConfig.jl -> setup.jl wake coupling",
                "shed_boundarylayer": "FluidConfig.jl -> setup.jl boundary-layer shedding",
                "pruning_thresholds": "setup.jl remove_particles_strength/sigma/sphere predicates",
            },
            "configured_controls": field_config["particles"]["fixed_population_controls"],
            "isolated_config_values": {key: isolated_fluid[key] for key in (
                "particles_per_step", "start_with_wake", "unsteady", "wake_coupled",
                "shed_boundarylayer", "remove_every", "sphere_factor",
                "strength_min_factor", "strength_max_factor", "sigma_min_factor",
                "sigma_max_factor")},
            "pruning_behavior": (
                "Very broad finite thresholds make the production removal predicates no-ops "
                "for finite MMS state values. The solver has no explicit pruning disable flag; "
                "this remains subject to runtime population auditing."),
            "separate_fluid_config_supported_by_cli": True,
            "count_and_identity_audit": "single-snapshot counts/roles only; lifecycle event history is unavailable",
            "count_audit_helper": "reconstruct_fields.jl:audit_fixed_population_history",
        },
    }
    fixed_controls_ok = (
        int(isolated_fluid["particles_per_step"]) == 0
        and isolated_fluid["start_with_wake"] is False
        and isolated_fluid["unsteady"] is False
        and isolated_fluid["wake_coupled"] is False
        and isolated_fluid["shed_boundarylayer"] is False
        and float(isolated_fluid["strength_min_factor"]) == 0.0
        and float(isolated_fluid["sigma_min_factor"]) == 0.0
        and float(isolated_fluid["sphere_factor"]) >= 1.0e50
        and float(isolated_fluid["strength_max_factor"]) >= 1.0e50
        and float(isolated_fluid["sigma_max_factor"]) >= 1.0e50
        and str(isolated_fluid.get("viscous_scheme", "none")).strip().lower() == "none"
        and str(isolated_fluid.get("vpm_sfs", "none")).strip().lower() == "none"
    )
    report["production_config_check"]["fixed_particle_lifecycle"]["configuration_check"] = (
        "PASS" if fixed_controls_ok else "FAIL")
    if not (viscous_off and sfs_off and fixed_controls_ok):
        report["status"] = "FAIL"
        report["production_config_check"]["reason"] = (
            "Field MMS requires viscosity/SFS off and all isolated fixed-population controls set")
    out.mkdir(parents=True, exist_ok=True)
    path = out / "analytical_preflight.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(f"Field MMS preflight failed; see {path}")
    return report


def coordinate_reference(request_path: Path) -> dict:
    """Generate Python coordinates and the production freestream at snapshot time."""
    request = json.loads(request_path.read_text(encoding="utf-8"))
    bounds = request["bounds_m"]
    cells = [int(v) for v in request["cell_counts"]]
    fluid = yaml.safe_load(Path(request["fluid_config"]).read_text(encoding="utf-8"))
    gradient = fluid.get("freestream_gradient", [0.0] * 9)
    wavenumber = fluid.get("freestream_wavenumber", [0.0] * 3)
    if len(gradient) != 9 or len(wavenumber) != 3:
        raise ValueError("Freestream gradient/wavenumber must have 9 and 3 entries")
    if any(float(v) != 0.0 for v in gradient) or any(float(v) != 0.0 for v in wavenumber):
        raise ValueError("Snapshot reconstruction requires a spatially uniform freestream")
    from field_mms import coordinate_mapping_samples
    reference = coordinate_mapping_samples(bounds, cells)
    fs = configured_particle_freestream(fluid, float(request["time_s"]))
    return {"coordinate_reference": reference, "freestream_m_s": fs.tolist(),
            "physical_time_s": float(request["time_s"])}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "structural", "fluid", "coupled", "particle-discrete", "coordinate-reference", "reconstruct-snapshot"))
    parser.add_argument("--config-dir", type=Path, default=ROOT / "config")
    parser.add_argument("--output-dir", "--output", dest="output_dir", type=Path, default=HERE / "results")
    parser.add_argument("--request", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--snapshot", type=int)
    parser.add_argument("--grid-config", type=Path, default=HERE / "field_mms_parameters.yaml")
    parser.add_argument("--study", choices=("spatial", "temporal", "both"), default="both")
    parser.add_argument("--levels", default="1,2,4")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.mode == "coordinate-reference":
        if args.request is None:
            parser.error("coordinate-reference requires --request")
        print(json.dumps(coordinate_reference(args.request)))
        return

    if args.mode == "reconstruct-snapshot":
        if args.manifest is None or args.snapshot is None:
            parser.error("reconstruct-snapshot requires --manifest and --snapshot")
        julia = os.environ.get("JULIA_BIN", str(HERE / "julia_with_openssl.sh"))
        project = os.environ.get("FLOWUNSTEADY_PROJECT", str(ROOT.parent))
        env = dict(os.environ, MMS2_PYTHON=sys.executable)
        command = [julia, f"--project={project}", str(HERE / "reconstruct_snapshot.jl"),
                   "--manifest", str(args.manifest.resolve()),
                   "--snapshot", str(args.snapshot),
                   "--grid-config", str(args.grid_config.resolve()),
                   "--output", str(args.output_dir.resolve())]
        raise SystemExit(subprocess.run(command, cwd=ROOT, env=env).returncode)

    if args.mode == "preflight":
        run_preflight(args.config_dir, args.output_dir / "fluid")
        return

    if args.mode == "structural":
        # Reuse the copied, unchanged structural formulation with mms_2 paths.
        from mms_driver import main as legacy_main
        legacy_main(["structural", "--config-dir", str(args.config_dir),
                     "--mms-params", str(HERE / "mms_parameters.yaml"),
                     "--output-dir", str(args.output_dir / "structural"),
                     "--study", args.study] + (["--dry-run"] if args.dry_run else []))
        return

    if args.mode == "particle-discrete":
        from mms_driver import main as legacy_main
        legacy_main(["fluid", "--config-dir", str(args.config_dir),
                     "--mms-params", str(HERE / "mms_parameters.yaml"),
                     "--output-dir", str(args.output_dir / "particle_discrete"),
                     "--dry-run"] if args.dry_run else [
                     "fluid", "--config-dir", str(args.config_dir),
                     "--mms-params", str(HERE / "mms_parameters.yaml"),
                     "--output-dir", str(args.output_dir / "particle_discrete")])
        return

    # The target and complete-pfield reconstruction are implemented. The
    # missing link is a justified projection of continuum forcing onto the
    # production rVPM's full (X, Gamma, sigma) state and stage recurrence.
    # Refuse to label the old particle-state MMS as a continuum field solve.
    run_preflight(args.config_dir, args.output_dir / args.mode)
    projection_note = (
        "Continuum velocity/vorticity target and source are derived, and the "
        "FLOWUnsteady full-particle Eulerian reconstruction is available. "
        "A mathematically justified projection of the continuum source to "
        "the production rVPM full particle state (positions, circulations, "
        "and core sizes) and its RK3 stages is not implemented. The brief "
        "explicitly forbids assuming a circulation-only projection. No "
        "simulation was launched and no field convergence result is claimed."
    )
    print(f"{args.mode} field solve is not enabled: {projection_note}", file=sys.stderr)
    raise SystemExit(2)


if __name__ == "__main__":
    main()
