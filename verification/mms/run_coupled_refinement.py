"""Run coupled fixed-population MMS time/spatial refinements.

Each case is compared to its own analytic target histories. The finest
computed solution is never used as a reference. This driver launches the
production socket-coupled solvers only when explicitly run by the user.
"""
import argparse
import csv
import json
import math
from pathlib import Path

from coupled_target import build_default_target, write_manifest
from run_coupled_mms import ROOT, SMOKE_DT, run_case


def flatten(value):
    if value and isinstance(value[0], list):
        return [float(v) for item in value for v in item]
    return [float(v) for v in value]


def relative_space_time_l2(rows, numerical_key, target_key):
    errors, references = [], []
    for row in rows:
        numerical, target = row[numerical_key], row[target_key]
        target, numerical = flatten(target), flatten(numerical)
        errors.extend((float(a)-float(b))**2 for a,b in zip(numerical,target))
        references.extend(float(v)**2 for v in target)
    denominator = math.sqrt(sum(references))
    return math.sqrt(sum(errors)) / max(denominator, 1e-30)


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
            fluid = run_case(manifest, case, steps, config_dir=config, dt=dt,
                             n_span=nspan, nx=nx, ny=ny)
            measures = {}
            with (case / "solid_mms_history.csv").open(newline="", encoding="utf-8") as stream:
                solid_rows = list(csv.DictReader(stream))
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
                           "physical_final_time_s":steps*dt,**measures,"case_dir":str(case)})
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
