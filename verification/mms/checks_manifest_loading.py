"""Fixed-population coupled discrete-model MMS. It exercises production
FLOWVLM/rVPM/Reissner–Mindlin operators and the production socket chronology.
It is not continuum Navier–Stokes MMS and does not verify native lifecycle or
CRM/RBF moment conservation. G0 covers Objective B structural coupling and the
fixed-population particle source; see formulation sections "The numerical
staggered socket-coupled system" and "Required output and pass/fail semantics".
"""
import argparse
import json
from pathlib import Path
import tempfile

from coupled_target import build_default_target, write_manifest
from run_coupled_mms import ROOT, SMOKE_DT, run_case


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT / "verification/mms/results/G0"))
    args = parser.parse_args()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    # FLOWUnsteady emits its initial t=0 coupling payload before advancing the
    # VPM field. Two coupling callbacks therefore contain exactly one actual
    # fluid time update, which is the smallest source-active production smoke.
    target_path = write_manifest(build_default_target(ROOT / "config", 2*SMOKE_DT), out / "target.json")
    constant = run_case(None, out / "constant", steps=2, constant_source="0.025,-0.012,0.008")
    target = run_case(target_path, out / "target", steps=2)
    def source_peak(path):
        stage_files = list((Path(path) / "fluid").glob("*_mms_rk3_stage_history.csv"))
        if not stage_files:
            raise AssertionError(f"MMS RK stage trace missing under {path}")
        rows = stage_files[0].read_text().splitlines()[1:]
        vals = [float(row.split(",")[3]) for row in rows if row.strip()]
        return max(vals, default=0.0)
    const_norm, target_norm = source_peak(out / "constant"), source_peak(out / "target")
    if const_norm <= 1e-12 or target_norm <= 1e-12:
        raise AssertionError(f"source path inactive: constant={const_norm}, target={target_norm}")
    if not all("target_force" in row for row in target):
        raise AssertionError("target aerodynamic history is missing")
    errors = [row["particle_state_error"]["position_l2"] for row in target]
    if not any(error > 1e-12 for error in errors):
        raise AssertionError("numerical particle state was replaced by the target")
    if [r["cl"] for r in constant] == [r["cl"] for r in target]:
        raise AssertionError("constant and target source histories are identical")
    result = {"constant_source_rms": const_norm, "target_source_rms": target_norm,
              "target_particle_position_error_l2": errors,
              "constant_vs_target_cl_differ": True}
    (out / "G0.json").write_text(json.dumps(result, indent=2) + "\n")
    print("PASS G0 checks_manifest_loading.py", json.dumps(result))


if __name__ == "__main__":
    main()
