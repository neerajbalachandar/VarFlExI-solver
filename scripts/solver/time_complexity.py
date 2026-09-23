#!/usr/bin/env python3
"""Reproducible time-complexity benchmark sweeps for VarFlExI.

This runner keeps the computational environment fixed, launches run.sh, samples
process-tree RSS, and aggregates warm-up-excluded component timings from the
solver diagnostics CSV files.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"
CONFIG_FILES = {
    "fluid": "fluid_params.yaml",
    "solid": "solid_params.yaml",
    "coupling": "coupling_params.yaml",
}
THREAD_ENV_KEYS = [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "JULIA_NUM_THREADS",
]
COMPONENT_COLUMNS = {
    "fluid": "fluid_step_time",
    "solid": "solid_solve_time",
    "f_to_s": "force_transfer_apply_time",
    "s_to_f": "geometry_transfer_apply_time",
    "io": "solid_io_time",
}


@dataclass(frozen=True)
class BenchCase:
    sweep: str
    parameter: str
    value: Any
    name: str
    fluid_updates: dict[str, Any]
    solid_updates: dict[str, Any]
    coupling_updates: dict[str, Any]


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def dump_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def sanitize(value: Any) -> str:
    out = str(value).strip().replace(" ", "_")
    for ch in "/\\:=,":
        out = out.replace(ch, "_")
    return out


def read_configs(config_dir: Path) -> dict[str, dict[str, Any]]:
    return {k: load_yaml(config_dir / v) for k, v in CONFIG_FILES.items()}


def write_configs(config_dir: Path, configs: dict[str, dict[str, Any]]) -> None:
    for key, filename in CONFIG_FILES.items():
        dump_yaml(config_dir / filename, configs[key])


def snapshot_configs(config_dir: Path) -> dict[Path, bytes]:
    return {config_dir / f: (config_dir / f).read_bytes() for f in CONFIG_FILES.values()}


def restore_configs(snapshot: dict[Path, bytes]) -> None:
    for path, payload in snapshot.items():
        path.write_bytes(payload)


def apply_updates(config: dict[str, Any], updates: dict[str, Any]) -> None:
    for key, value in updates.items():
        cursor = config
        parts = key.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value


def proc_children_map() -> dict[int, list[int]]:
    children: dict[int, list[int]] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text().split()
            pid = int(stat[0])
            ppid = int(stat[3])
        except Exception:
            continue
        children.setdefault(ppid, []).append(pid)
    return children


def descendants(root_pid: int) -> set[int]:
    children = proc_children_map()
    todo = [root_pid]
    seen: set[int] = set()
    while todo:
        pid = todo.pop()
        if pid in seen:
            continue
        seen.add(pid)
        todo.extend(children.get(pid, []))
    return seen


def rss_bytes_for_pid(pid: int) -> int:
    try:
        fields = Path(f"/proc/{pid}/statm").read_text().split()
        return int(fields[1]) * os.sysconf("SC_PAGE_SIZE")
    except Exception:
        return 0


def process_tree_rss(root_pid: int) -> tuple[int, int]:
    pids = descendants(root_pid)
    return sum(rss_bytes_for_pid(pid) for pid in pids), len(pids)


def read_last_step(coupling_csv: Path) -> int:
    if not coupling_csv.exists():
        return -1
    try:
        with coupling_csv.open(newline="", encoding="utf-8") as f:
            rows = csv.DictReader(f)
            last = -1
            for row in rows:
                last = int(float(row.get("step", -1)))
            return last
    except Exception:
        return -1


def fixed_env(threads: int) -> dict[str, str]:
    env = os.environ.copy()
    for key in THREAD_ENV_KEYS:
        env[key] = str(threads)
    env["PYTHONPATH"] = f"{REPO_ROOT}:{env.get('PYTHONPATH', '')}"
    env["VARFLEXI_BENCHMARK"] = "1"
    return env


def run_monitored(command: list[str], cwd: Path, log_path: Path, sample_csv: Path,
                  coupling_csv: Path, threads: int, cpu_list: str | None,
                  sample_interval: float) -> tuple[float, dict[str, Any]]:
    env = fixed_env(threads)
    launch = list(command)
    if cpu_list:
        launch = ["taskset", "-c", cpu_list, *launch]

    peak = {"rss_bytes": 0, "elapsed_s": 0.0, "step": -1, "phase": "simulation"}
    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log, sample_csv.open("w", newline="", encoding="utf-8") as sf:
        writer = csv.DictWriter(sf, fieldnames=["elapsed_s", "rss_bytes", "rss_mib", "n_processes", "step", "phase"])
        writer.writeheader()
        proc = subprocess.Popen(launch, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert proc.stdout is not None
        next_sample = start
        while True:
            line = proc.stdout.readline()
            if line:
                print(line, end="")
                log.write(line)
            now = time.perf_counter()
            if now >= next_sample:
                rss, nproc = process_tree_rss(proc.pid)
                elapsed = now - start
                step = read_last_step(coupling_csv)
                row = {
                    "elapsed_s": f"{elapsed:.9e}",
                    "rss_bytes": rss,
                    "rss_mib": f"{rss / 1048576.0:.9e}",
                    "n_processes": nproc,
                    "step": step,
                    "phase": "simulation",
                }
                writer.writerow(row)
                sf.flush()
                if rss > int(peak["rss_bytes"]):
                    peak = {"rss_bytes": rss, "elapsed_s": elapsed, "step": step, "phase": "simulation"}
                next_sample = now + sample_interval
            if proc.poll() is not None:
                for rest in proc.stdout:
                    print(rest, end="")
                    log.write(rest)
                break
            if not line:
                time.sleep(min(0.05, sample_interval))
        rc = proc.wait()
    wall = time.perf_counter() - start
    if rc != 0:
        raise subprocess.CalledProcessError(rc, launch)
    return wall, peak


def numeric(value: Any, default: float = math.nan) -> float:
    try:
        return float(value)
    except Exception:
        return default


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def sum_column(rows: list[dict[str, str]], column: str, warm: int) -> float:
    total = 0.0
    for row in rows:
        if int(float(row.get("step", 0))) <= warm:
            continue
        v = numeric(row.get(column))
        if math.isfinite(v):
            total += v
    return total


def problem_size(configs: dict[str, dict[str, Any]], rows: list[dict[str, str]]) -> dict[str, Any]:
    fluid = configs["fluid"]
    solid = configs["solid"]
    coupling = configs["coupling"]
    n_span = int(fluid.get("n_span", coupling.get("comm_nspan", 0)))
    n_chord = int(fluid.get("n_chord", 1))
    nx = int(solid.get("nx", 0))
    ny = int(solid.get("ny", 0))
    n_particles = math.nan
    for row in reversed(rows):
        if "particles" in row:
            n_particles = numeric(row.get("particles"))
            break
    return {
        "dt": float(coupling.get("total_time", 0.0)) / max(int(coupling.get("n_steps", 1)), 1),
        "n_steps": int(coupling.get("n_steps", 0)),
        "n_surface": n_span * n_chord,
        "n_structural_dof": 3 * (nx + 1) * (ny + 1),
        "n_span": n_span,
        "n_chord": n_chord,
        "nx": nx,
        "ny": ny,
        "n_pperstep": int(fluid.get("particles_per_step", 0)),
        "n_particles": n_particles,
    }


def aggregate_run(run_dir: Path, configs: dict[str, dict[str, Any]], warm: int, wall_s: float, peak: dict[str, Any],
                  sweep: str, parameter: str, value: Any, run_id: int, threads: int, cpu_list: str | None) -> dict[str, Any]:
    coupling_rows = read_csv(run_dir / "results" / "coupling" / "coupling_history.csv")
    fluid_rows = read_csv(run_dir / "results" / "fluid" / "static_airfoil_spanwise_coupling_diagnostics.csv")
    rows = coupling_rows
    bench_rows = [r for r in rows if int(float(r.get("step", 0))) > warm]
    n_bench = len(bench_rows)
    components = {name: sum_column(rows, col, warm) for name, col in COMPONENT_COLUMNS.items()}
    setup_crm = max((numeric(r.get("crm_setup_time"), 0.0) for r in rows), default=0.0)
    setup_rbf = max((numeric(r.get("rbf_setup_time"), 0.0) for r in rows), default=0.0)
    setup_geom = max((numeric(r.get("geometry_rbf_setup_time"), 0.0) for r in rows), default=0.0)
    t_benchmark = components["fluid"] + sum_column(rows, "solid_step_time", warm)
    transfer = components["f_to_s"] + components["s_to_f"]
    known = components["fluid"] + components["solid"] + transfer + components["io"]
    coupling_other = max(t_benchmark - known, 0.0)
    components["transfer"] = transfer
    components["coupling"] = coupling_other
    components["other"] = coupling_other
    size = problem_size(configs, fluid_rows)
    out = {
        "sweep": sweep,
        "parameter": parameter,
        "parameter_value": value,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "n_warm": warm,
        "n_benchmark_steps": n_bench,
        "wall_total_s": wall_s,
        "t_benchmark_s": t_benchmark,
        "mean_step_s": t_benchmark / n_bench if n_bench else math.nan,
        "setup_crm_s": setup_crm,
        "setup_rbf_s": setup_rbf,
        "setup_geometry_rbf_s": setup_geom,
        "m_peak_bytes": peak["rss_bytes"],
        "m_peak_mib": peak["rss_bytes"] / 1048576.0,
        "t_peak_s": peak["elapsed_s"],
        "n_step_peak": peak["step"],
        "phase_peak": peak["phase"],
        "threads": threads,
        "cpu_list": cpu_list or "",
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    out.update(size)
    for key, value2 in components.items():
        out[f"t_{key}_s"] = value2
        out[f"mean_{key}_step_s"] = value2 / n_bench if n_bench else math.nan
        out[f"frac_{key}"] = value2 / t_benchmark if t_benchmark > 0 else math.nan
    return out


def make_cases(base: dict[str, dict[str, Any]], total_time: float, dt_values: list[float], spatial_values: list[int], p_values: list[int]) -> list[BenchCase]:
    cases: list[BenchCase] = []
    for dt in dt_values:
        cases.append(BenchCase("dt", "dt", dt, f"dt_{sanitize(dt)}", {}, {}, {"total_time": total_time, "n_steps": int(round(total_time / dt))}))
    for n in spatial_values:
        cases.append(BenchCase("spatial", "n_span", n, f"n_span_{n}", {"n_span": n}, {"ny": max(n, int(base["solid"].get("ny", n)))}, {"comm_nspan": n}))
    for pper in p_values:
        cases.append(BenchCase("pperstep", "particles_per_step", pper, f"pperstep_{pper}", {"particles_per_step": pper}, {}, {}))
    return cases


def stats(values: list[float]) -> tuple[float, float, float]:
    vals = [v for v in values if math.isfinite(v)]
    if not vals:
        return math.nan, math.nan, math.nan
    mean = statistics.mean(vals)
    var = statistics.variance(vals) if len(vals) > 1 else 0.0
    return mean, var, math.sqrt(var)


def write_aggregate(raw_rows: list[dict[str, Any]], path: Path) -> None:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in raw_rows:
        groups.setdefault((row["sweep"], row["parameter"], str(row["parameter_value"])), []).append(row)
    metric_cols = [c for c in raw_rows[0] if c.startswith("t_") or c.startswith("mean_") or c.startswith("setup_") or c.startswith("m_peak") or c.startswith("frac_")]
    fieldnames = ["sweep", "parameter", "parameter_value", "n_runs", "run_dirs"]
    for col in metric_cols:
        fieldnames.extend([f"{col}_mean", f"{col}_variance", f"{col}_std"])
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for key, rows in sorted(groups.items()):
            out = {"sweep": key[0], "parameter": key[1], "parameter_value": key[2], "n_runs": len(rows), "run_dirs": json.dumps([r["run_dir"] for r in rows])}
            for col in metric_cols:
                mean, var, std = stats([numeric(r.get(col)) for r in rows])
                out[f"{col}_mean"] = mean
                out[f"{col}_variance"] = var
                out[f"{col}_std"] = std
            w.writerow(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run VarFlExI time-complexity benchmarks")
    parser.add_argument("--results-root", default=str(REPO_ROOT / "results" / "time_complexity"))
    parser.add_argument("--total-time", type=float, default=0.20)
    parser.add_argument("--dt-values", default="0.004,0.002,0.001")
    parser.add_argument("--spatial-values", default="20,40,80")
    parser.add_argument("--pperstep-values", default="1,2,4")
    parser.add_argument("--n-runs", type=int, default=3)
    parser.add_argument("--n-warm", type=int, default=60)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--cpu-list", default=None)
    parser.add_argument("--sample-interval", type=float, default=0.25)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    results_root = Path(args.results_root).expanduser().resolve()
    raw_csv = results_root / "benchmark_runs.csv"
    agg_csv = results_root / "benchmark_summary.csv"
    results_root.mkdir(parents=True, exist_ok=True)

    baseline = read_configs(CONFIG_DIR)
    snapshot = snapshot_configs(CONFIG_DIR)
    cases = make_cases(
        baseline,
        args.total_time,
        [float(v) for v in args.dt_values.split(",") if v.strip()],
        [int(v) for v in args.spatial_values.split(",") if v.strip()],
        [int(v) for v in args.pperstep_values.split(",") if v.strip()],
    )
    raw_rows: list[dict[str, Any]] = []
    try:
        for case in cases:
            for run_id in range(1, args.n_runs + 1):
                configs = copy.deepcopy(baseline)
                apply_updates(configs["fluid"], case.fluid_updates)
                apply_updates(configs["solid"], case.solid_updates)
                apply_updates(configs["coupling"], case.coupling_updates)
                run_dir = results_root / case.sweep / case.name / f"run_{run_id:02d}"
                run_results = run_dir / "results"
                configs["fluid"]["save_directory"] = str(run_results / "fluid")
                configs["solid"]["results_root"] = str(run_results)
                configs["coupling"]["results_root"] = str(run_results)
                run_dir.mkdir(parents=True, exist_ok=True)
                write_configs(CONFIG_DIR, configs)
                config_copy = run_dir / "config"
                config_copy.mkdir(parents=True, exist_ok=True)
                for filename in CONFIG_FILES.values():
                    shutil.copy2(CONFIG_DIR / filename, config_copy / filename)
                metadata = {
                    "case": case.__dict__,
                    "n_warm": args.n_warm,
                    "threads": args.threads,
                    "cpu_list": args.cpu_list,
                    "sample_interval": args.sample_interval,
                    "thread_env": {k: str(args.threads) for k in THREAD_ENV_KEYS},
                }
                (run_dir / "benchmark_metadata.yaml").write_text(yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8")
                print(f"[time_complexity] prepared {case.sweep}/{case.name} run {run_id}: {run_dir}")
                if args.dry_run:
                    continue
                coupling_csv = run_results / "coupling" / "coupling_history.csv"
                wall, peak = run_monitored(["bash", "run.sh"], REPO_ROOT, run_dir / "run.log", run_dir / "memory_samples.csv", coupling_csv, args.threads, args.cpu_list, args.sample_interval)
                row = aggregate_run(run_dir, configs, args.n_warm, wall, peak, case.sweep, case.parameter, case.value, run_id, args.threads, args.cpu_list)
                raw_rows.append(row)
                fieldnames = sorted({k for r in raw_rows for k in r})
                with raw_csv.open("w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=fieldnames)
                    w.writeheader(); w.writerows(raw_rows)
                write_aggregate(raw_rows, agg_csv)
                print(f"[time_complexity] completed {case.name} run {run_id}")
    finally:
        restore_configs(snapshot)

    if raw_rows:
        print(f"Raw benchmark data: {raw_csv}")
        print(f"Aggregated summary: {agg_csv}")


if __name__ == "__main__":
    main()
