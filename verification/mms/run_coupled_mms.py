"""Fixed-population coupled discrete-model MMS. It exercises the installed
FLOWVLM/rVPM/Reissner–Mindlin operators and the production socket chronology.
It is not a continuum Navier–Stokes MMS, does not verify native particle
lifecycle, and does not treat CRM/RBF moment conservation as verified.

Implements formulation sections "The numerical staggered socket-coupled
system", "Required output and pass/fail semantics", and "Error metrics".
This is Objective B (end-to-end transfer and FSI accuracy) for the structural
source and fixed-population scope for the particle source. The target
population is seeded inside the production integrator immediately before its
first RK stage, allowing a one-step source activation gate.
"""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time

import yaml

ROOT = Path(__file__).resolve().parents[2]
SMOKE_DT = 1.0e-2


class _FlowListDumper(yaml.SafeDumper):
    pass


_FlowListDumper.add_representer(
    list, lambda dumper, value: dumper.represent_sequence("tag:yaml.org,2002:seq", value, flow_style=True)
)


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def write_config(source, destination, updates):
    with open(source, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    data.update(updates)
    with open(destination, "w", encoding="utf-8") as f:
        yaml.dump(data, f, Dumper=_FlowListDumper, sort_keys=False)


def run_case(manifest, output, steps=3, constant_source=None, config_dir=ROOT / "config"):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    work = output / "configs"
    work.mkdir(exist_ok=True)
    port = free_port()
    dt = SMOKE_DT
    config_dir = Path(config_dir).resolve()
    write_config(config_dir / "fluid_params.yaml", work / "fluid_params.yaml", {
        "n_span": 16, "save_directory": str(output / "fluid"), "run_name": "coupled_mms",
        "prompt": False, "save_horseshoes": False, "save_geometry_csv": False,
        "save_every": 1,
    })
    write_config(config_dir / "solid_params.yaml", work / "solid_params.yaml", {
        "nx": 8, "ny": 16, "output_dir": str(output / "solid"),
        "force_ramp_iters": 1,
    })
    write_config(config_dir / "coupling_params.yaml", work / "coupling_params.yaml", {
        "n_steps": int(steps), "total_time": dt * int(steps), "port": port,
        "comm_nspan": 16, "results_root": str(output / "coupling"),
    })
    procs = []
    log_path = output / "process.log"
    with open(log_path, "w", encoding="utf-8") as log:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        if constant_source is not None:
            env["VARFLEXI_MMS_POSITION_SOURCE"] = constant_source
        cmd_coupler = [sys.executable, str(ROOT / "coupling/varflexi.py"), "--config_path", str(work)]
        cmd_solid = [sys.executable, str(ROOT / "solid/structural_solver.py"), "--config_path", str(work),
                     "--mms-output", str(output)]
        if manifest:
            cmd_solid += ["--mms-target", str(Path(manifest).resolve())]
        procs.append(subprocess.Popen(cmd_coupler, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT))
        time.sleep(0.25)
        procs.append(subprocess.Popen(cmd_solid, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT))
        time.sleep(0.25)
        julia_project = Path(os.environ.get("FLOWUNSTEADY_PROJECT", ROOT.parent))
        cmd_fluid = ["julia", f"--project={julia_project}", str(ROOT / "fluid/fluid.jl"),
                     "--fluid", str(work / "fluid_params.yaml"), "--solid", str(work / "solid_params.yaml"),
                     "--coupling", str(work / "coupling_params.yaml"), "--mms-output", str(output / "fluid")]
        if manifest:
            cmd_fluid += ["--mms-target", str(Path(manifest).resolve())]
        procs.append(subprocess.Popen(cmd_fluid, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT))
        deadline = time.monotonic() + 300
        while any(p.poll() is None for p in procs) and time.monotonic() < deadline:
            failed = [p for p in procs if p.poll() not in (None, 0)]
            if failed:
                break
            time.sleep(0.25)
        for p in procs:
            if p.poll() is None:
                p.terminate()
        for p in procs:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
        codes = [p.returncode for p in procs]
    if any(code != 0 for code in codes):
        raise RuntimeError(f"coupled processes failed ({codes}); see {log_path}")
    traces = list((output / "fluid").glob("*_force_payload_history.jsonl"))
    if not traces:
        raise RuntimeError(f"fluid force history missing under {output / 'fluid'}")
    rows = [json.loads(line) for line in traces[0].read_text().splitlines() if line.strip()]
    if len(rows) != int(steps):
        raise RuntimeError(f"expected {steps} fluid history rows, got {len(rows)}")
    solid_history = output / "solid_mms_history.csv"
    if manifest and (not solid_history.is_file() or len(solid_history.read_text().splitlines()) != int(steps) + 1):
        raise RuntimeError(f"target/numerical solid history missing or incomplete: {solid_history}")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", default=str(ROOT / "config"))
    parser.add_argument("--manifest")
    parser.add_argument("--output", default=str(ROOT / "verification/mms/results/coupled"))
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()
    from coupled_target import build_default_target, write_manifest
    manifest = Path(args.manifest) if args.manifest else Path(args.output) / "target.json"
    if not args.manifest:
        write_manifest(build_default_target(args.config_dir, args.steps * SMOKE_DT), manifest)
    rows = run_case(manifest, args.output, args.steps, config_dir=args.config_dir)
    solid_rows = []
    import csv
    with open(Path(args.output) / "solid_mms_history.csv", newline="", encoding="utf-8") as stream:
        solid_rows = list(csv.DictReader(stream))
    coupler_history = Path(args.output) / "coupling" / "coupling" / "forces_sent_received_history.jsonl"
    coupling_rows = [json.loads(line) for line in coupler_history.read_text().splitlines() if line.strip()]
    summary = {
        "scope": "fixed-population coupled discrete-model MMS; no lifecycle or convergence claim",
        "steps": len(rows),
        "target_force_history_present": all("target_force" in r for r in rows),
        "target_circulation_history_present": all("target_bound_circulation" in r for r in rows),
        "target_circulation_residual_max": max(r["target_circulation_residual"] for r in rows),
        "particle_position_error_l2": [r["particle_state_error"]["position_l2"] for r in rows],
        "particle_circulation_error_l2": [r["particle_state_error"]["circulation_l2"] for r in rows],
        "bound_circulation_error_l2": [sum((float(a)-float(b))**2 for a,b in zip(
            r["bound_circulation"], r["target_bound_circulation"]))**0.5 for r in rows],
        "panel_force_error_l2": [sum((float(a)-float(b))**2
            for ar,br in zip(r["force"],r["target_force"]) for a,b in zip(ar,br))**0.5 for r in rows],
        "fixed_grid_velocity_error_l2": [r["fixed_grid_velocity_error_l2"] for r in rows],
        "geometry_cp_error_l2": [r["geometry_cp_error_l2"] for r in rows],
        "structural_state_error_l2": [float(r["state_error_l2"]) for r in solid_rows],
        "structural_source_l2": [float(r["source_l2"]) for r in solid_rows],
        "force_transfer_relative_error": [r.get("force_transfer_relative_error") for r in coupling_rows],
        "geometry_transfer_relative_error": [r.get("geometry_transfer_relative_error") for r in coupling_rows],
        "fluid_solid_stage_times_s": [[r.get("fluid_time_s"), r.get("solid_time_s")] for r in coupling_rows],
        "physical_times_s": [r["fluid_time_s"] for r in rows],
        "fluid_history": str(next((Path(args.output) / "fluid").glob("*_force_payload_history.jsonl"))),
    }
    Path(args.output, "mms_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
