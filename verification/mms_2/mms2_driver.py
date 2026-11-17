"""Entry point for the isolated mms_2 verification suite.

The field fluid/coupled run commands intentionally fail closed until the
continuum-to-rVPM full-state source projection is integrated. Use `preflight`
for analytical/configuration checks and the copied structural mode for a real
standalone structural solve.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from field_mms import analytical_check

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent


def production_config(config_dir: Path) -> tuple[dict, dict]:
    fluid = yaml.safe_load((config_dir / "fluid_params.yaml").read_text(encoding="utf-8"))
    solid = yaml.safe_load((config_dir / "solid_params.yaml").read_text(encoding="utf-8"))
    return fluid, solid


def run_preflight(config_dir: Path, out: Path) -> dict:
    fluid, solid = production_config(config_dir)
    expected = yaml.safe_load((HERE / "field_mms_parameters.yaml").read_text(encoding="utf-8"))["physics"]
    actual_viscous = str(fluid.get("viscous_scheme", "CoreSpreading")).strip().lower()
    actual_sfs = str(fluid.get("vpm_sfs", "none")).strip().lower()
    viscous_off = actual_viscous in ("none", "off", "false", "nothing")
    sfs_off = actual_sfs in ("none", "off", "false")
    report = analytical_check(HERE / "field_mms_parameters.yaml")
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
    }
    if not (viscous_off and sfs_off):
        report["status"] = "FAIL"
        report["production_config_check"]["reason"] = (
            "Field MMS requires viscous_scheme=none and vpm_sfs=none")
    out.mkdir(parents=True, exist_ok=True)
    path = out / "analytical_preflight.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(f"Field MMS preflight failed; see {path}")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "structural", "fluid", "coupled", "particle-discrete"))
    parser.add_argument("--config-dir", type=Path, default=ROOT / "config")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    parser.add_argument("--study", choices=("spatial", "temporal", "both"), default="both")
    parser.add_argument("--levels", default="1,2,4")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

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
