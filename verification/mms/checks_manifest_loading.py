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
    stage_files = list((out / "target" / "fluid").glob("*_mms_rk3_stage_history.csv"))
    if not stage_files:
        raise AssertionError("target source callback history was not emitted")
    stage_rows = [row for row in stage_files[0].read_text().splitlines()[1:] if row.strip()]
    if len(stage_rows) < 3:
        raise AssertionError("target source was not evaluated at all RK3 stages")
    if not {"1", "2", "3"}.issubset({row.split(",", 1)[0] for row in stage_rows}):
        raise AssertionError("target source history is missing one or more production RK3 stages")
    if not all("numerical_particle_state" in row and "target_particle_state" in row for row in target):
        raise AssertionError("independent target and numerical particle histories are missing")
    # Numerical evolution is checked independently of target error: an exact
    # match is allowed. Require finite, changing numerical states and active
    # source rows rather than relying on a nonzero truncation error.
    numeric_states = [row["numerical_particle_state"]["position"] for row in target]
    if len(numeric_states) > 1 and numeric_states[0] == numeric_states[-1]:
        raise AssertionError("numerical particle state did not evolve between physical times")
    if [r["cl"] for r in constant] == [r["cl"] for r in target]:
        raise AssertionError("constant and target source histories are identical")
    result = {"constant_source_rms": const_norm, "target_source_rms": target_norm,
              "target_particle_position_error_l2": [row["particle_state_error"]["position_l2"] for row in target],
              "target_source_stage_rows": len(stage_rows),
              "numerical_particle_evolution_observed": True,
              "constant_vs_target_cl_differ": True}
    (out / "G0.json").write_text(json.dumps(result, indent=2) + "\n")
    print("PASS G0 checks_manifest_loading.py", json.dumps(result))


if __name__ == "__main__":
    main()
