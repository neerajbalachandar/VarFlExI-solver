"""Run coupled fixed-population MMS time/spatial refinements.

Each case is compared to its own analytic target histories. The finest
computed solution is never used as a reference. This driver launches the
production socket-coupled solvers only when explicitly run by the user.

Scope: fixed particle population; bound circulation is checked against the
target VLM residual, not forced; the CRM transfer operator is machine-precision
on the target; the coupled error therefore includes the VLM solution error
and the RBF interpolation floor where RBF is used. The coupled error is not a
continuum Navier-Stokes error.
"""
import argparse
import csv
import json
import math
from numbers import Real
from pathlib import Path
import sys

from coupled_target import build_default_target, write_manifest
from run_coupled_mms import ROOT, SMOKE_DT, run_case


def configure_csv_field_limit():
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


configure_csv_field_limit()


def flatten(value):
    """Flatten scalar or nested JSON arrays without assuming a fixed rank."""
    if isinstance(value, (list, tuple)):
        return [component for item in value for component in flatten(item)]
    if isinstance(value, Real):
        return [float(value)]
    raise TypeError(f"expected a numeric scalar or nested sequence, got {type(value).__name__}")


def relative_space_time_l2(rows, numerical_key, target_key):
    errors, references = [], []
    for row in rows:
        numerical, target = row[numerical_key], row[target_key]
        target, numerical = flatten(target), flatten(numerical)
        if len(numerical) != len(target):
            raise ValueError(
                f"{numerical_key}/{target_key} shape mismatch at row: "
                f"{len(numerical)} numerical values vs {len(target)} target values"
            )
        errors.extend((float(a)-float(b))**2 for a,b in zip(numerical,target))
        references.extend(float(v)**2 for v in target)
    denominator = math.sqrt(sum(references))
    return math.sqrt(sum(errors)) / max(denominator, 1e-30)


def load_completed_case(case, steps):
    """Reuse a finished case when a prior run stopped during report assembly."""
    fluid_dir = case / "fluid"
    traces = list(fluid_dir.glob("*_force_payload_history.jsonl"))
    solid_path = case / "solid_mms_history.csv"
    coupling_path = case / "coupling" / "coupling" / "forces_sent_received_history.jsonl"
    stage_files = list(fluid_dir.glob("*_mms_rk3_stage_history.csv"))
    if (len(traces) != 1 or not solid_path.is_file() or not coupling_path.is_file()
            or not coupling_path.read_text().strip() or len(stage_files) != 1):
        return None
    try:
        fluid_rows = [json.loads(line) for line in traces[0].read_text().splitlines() if line.strip()]
        with solid_path.open(newline="", encoding="utf-8") as stream:
            solid_rows = list(csv.DictReader(stream))
        if len(fluid_rows) != steps or len(solid_rows) != steps:
            return None
        fluid_required = (
            "numerical_particle_state", "target_particle_state", "bound_circulation",
            "target_bound_circulation", "force", "target_force",
        )
        if any(key not in row for row in fluid_rows for key in fluid_required):
            return None
        for row in solid_rows:
            numerical = json.loads(row["numerical_state"])
            target = json.loads(row["target_state"])
            error = float(row["state_error_l2"])
            if (not isinstance(numerical, list) or not isinstance(target, list)
                    or len(numerical) != len(target) or not math.isfinite(error) or error < 0):
                return None
        with stage_files[0].open(newline="", encoding="utf-8") as stream:
            stage_rows = list(csv.DictReader(stream))
        if not {"1", "2", "3"}.issubset({row.get("stage") for row in stage_rows}):
            return None
    except (OSError, ValueError, KeyError, json.JSONDecodeError, csv.Error):
        return None
    return fluid_rows, solid_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", choices=("time", "spatial", "both"), default="both")
    parser.add_argument("--levels", default="1,2,4")
    parser.add_argument("--duration", type=float, default=0.03)
    parser.add_argument("--dt", type=float, default=SMOKE_DT)
    parser.add_argument("--n-span", type=int, default=8)
    parser.add_argument("--nx", type=int, default=4)
    parser.add_argument("--ny", type=int, default=8)
    parser.add_argument("--output", default=str(ROOT / "verification/mms/results/coupled_refinement"))
    parser.add_argument("--config-dir", default=str(ROOT / "config"))
    args = parser.parse_args()
    levels = [int(v) for v in args.levels.split(",") if v.strip()]
    if not levels or any(v < 1 for v in levels):
        parser.error("--levels must be a comma-separated list of positive integers")
    output, config = Path(args.output).resolve(), Path(args.config_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    studies = ("time", "spatial") if args.study == "both" else (args.study,)
    report = []
    for study in studies:
        for level in levels:
            if study == "time":
                dt = args.dt / level
                steps_float = args.duration / dt
                steps = round(steps_float)
                if not math.isclose(steps_float, steps, rel_tol=0.0, abs_tol=1e-9):
                    parser.error("duration/dt must be integral at every temporal refinement level")
                nspan, nx, ny = args.n_span, args.nx, args.ny
            else:
                dt = args.dt
                steps_float = args.duration / dt
                steps = round(steps_float)
                if not math.isclose(steps_float, steps, rel_tol=0.0, abs_tol=1e-9):
                    parser.error("duration/dt must be integral for spatial refinement")
                nspan, nx, ny = args.n_span*level, args.nx*level, args.ny*level
            case = output / f"{study}_level_{level}"
            manifest = write_manifest(build_default_target(config, args.duration, nspan), case / "target.json")
            cached = load_completed_case(case, steps)
            reused = cached is not None
            if cached is None:
                fluid = run_case(manifest, case, steps, config_dir=config, dt=dt,
                                 n_span=nspan, nx=nx, ny=ny)
                with (case / "solid_mms_history.csv").open(newline="", encoding="utf-8") as stream:
                    solid_rows = list(csv.DictReader(stream))
            else:
                fluid, solid_rows = cached
                print(f"Reusing completed histories in {case}")
            measures = {}
            solid_pairs = [{"numerical": json.loads(r["numerical_state"]),
                            "target": json.loads(r["target_state"])} for r in solid_rows]
            measures["structural_state_relative_space_time_l2"] = relative_space_time_l2(
                solid_pairs, "numerical", "target")
            for field in ("position", "circulation", "core"):
                paired = [{"numerical": r["numerical_particle_state"][field],
                           "target": r["target_particle_state"][field]} for r in fluid]
                measures[f"particle_{field}_relative_space_time_l2"] = relative_space_time_l2(paired, "numerical", "target")
            for key, nkey, tkey in (
                ("bound_circulation_relative_space_time_l2", "bound_circulation", "target_bound_circulation"),
                ("force_relative_space_time_l2", "force", "target_force"),
                ("probe_velocity_relative_space_time_l2", "numerical_probe_velocity_m_s", "target_probe_velocity_m_s"),
                ("probe_vorticity_relative_space_time_l2", "numerical_probe_vorticity_s_inv", "target_probe_vorticity_s_inv"),
            ):
                measures[key] = relative_space_time_l2(fluid, nkey, tkey)
            report.append({"study":study,"level":level,"dt_s":dt,"steps":steps,
                           "n_span":nspan,"solid_nx":nx,"solid_ny":ny,
                           "physical_final_time_s":steps*dt,
                           "reused":reused,
                           "structural_error_with_target_force":measures["structural_state_relative_space_time_l2"],
                           "target_force_source":"analytic panel force from BoundCirculationTarget",
                           **measures,"case_dir":str(case)})
        series = [row for row in report if row["study"] == study]
        series.sort(key=lambda row: row["level"])
        metric_names = [key for key in series[0] if key.endswith("relative_space_time_l2")]
        for index, row in enumerate(series):
            orders = {}
            if index:
                previous = series[index-1]
                h_now = row["dt_s"] if study == "time" else 1.0 / row["level"]
                h_prev = previous["dt_s"] if study == "time" else 1.0 / previous["level"]
                for metric in metric_names:
                    e0, e1 = previous[metric], row[metric]
                    orders[metric] = (math.log(e0/e1) / math.log(h_prev/h_now)
                                      if e0 > 0 and e1 > 0 and h_prev > h_now else None)
            row["observed_order_from_previous_level"] = orders
    json_path = output / "coupled_refinement_report.json"
    csv_path = output / "coupled_refinement_report.csv"
    json_path.write_text(json.dumps({"scope":"analytic-target coupled discrete-model MMS; no finest-run reference", "cases":report},indent=2)+"\n")
    with csv_path.open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(report[0]))
        writer.writeheader(); writer.writerows(report)
    print(f"Wrote {json_path} and {csv_path}")


if __name__ == "__main__":
    main()
