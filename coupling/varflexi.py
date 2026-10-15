import argparse
import csv
import errno
import json
import os
import socket
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml


def load_yaml_config(config_dir, *candidate_names):
    for name in candidate_names:
        cfg_path = os.path.join(config_dir, name)
        if os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as stream:
                return yaml.safe_load(stream)
    raise FileNotFoundError(
        f"could not find this config file in {config_dir}: {', '.join(candidate_names)}"
    )

def cfg_get(cfg, *keys, default=None):
    cur = cfg
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def as_bool(value, default=False):
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in ("0", "false", "no", "off", "")


def vector_rows(raw: Any, key: str) -> List[List[float]]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise RuntimeError(f"Payload key '{key}' must be a list")

    rows: List[List[float]] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            raise RuntimeError(f"Payload key '{key}' entry {idx} is not a 3-vector")
        rows.append([float(item[0]), float(item[1]), float(item[2])])
    return rows


def vector_norm(rows: Sequence[Sequence[float]]) -> float:
    total = 0.0
    for row in rows:
        for value in row[:3]:
            total += float(value) ** 2
    return total ** 0.5


def residual_metrics(
    current: Sequence[Sequence[float]],
    previous: Optional[Sequence[Sequence[float]]],
    epsilon_reg: float,
) -> Tuple[float, float, float]:
    reference_norm = vector_norm(current)
    if previous is None or len(current) == 0 or len(current) != len(previous):
        return 0.0, reference_norm, 0.0

    residual_sq = 0.0
    for a, b in zip(current, previous):
        dx = float(a[0]) - float(b[0])
        dy = float(a[1]) - float(b[1])
        dz = float(a[2]) - float(b[2])
        residual_sq += dx * dx + dy * dy + dz * dz

    residual = residual_sq ** 0.5
    relative_error = residual / max(reference_norm + epsilon_reg, 1.0e-300)
    return residual, reference_norm, relative_error


def transfer_metrics(
    structural_common: Sequence[Sequence[float]],
    fluid_common: Sequence[Sequence[float]],
    epsilon_reg: float,
) -> Tuple[float, float]:
    if len(structural_common) == 0 or len(structural_common) != len(fluid_common):
        return float("nan"), float("nan")

    residual_sq = 0.0
    reference_sq = 0.0
    for structural_row, fluid_row in zip(structural_common, fluid_common):
        for structural_value, fluid_value in zip(structural_row[:3], fluid_row[:3]):
            structural_value = float(structural_value)
            fluid_value = float(fluid_value)
            diff = structural_value - fluid_value
            residual_sq += diff * diff
            reference_sq += fluid_value * fluid_value

    residual = residual_sq ** 0.5
    reference_norm = reference_sq ** 0.5
    relative_error = residual / max(reference_norm + epsilon_reg, 1.0e-300)
    return residual, relative_error


def force_components(forces: Sequence[Sequence[float]]) -> Tuple[float, float]:
    lift = 0.0
    drag = 0.0
    for row in forces:
        if len(row) >= 3:
            drag -= float(row[0])
            lift += float(row[2])
    return lift, drag


def force_coefficients(
    forces: Sequence[Sequence[float]],
    q_inf: Optional[float],
    ref_area: Optional[float],
) -> Tuple[float, float]:
    lift, drag = force_components(forces)
    if q_inf is None or ref_area is None:
        return float("nan"), float("nan")
    denom = max(float(q_inf) * float(ref_area), 1.0e-16)
    return lift / denom, drag / denom


def optional_float(data: Dict[str, Any], key: str) -> float:
    value = data.get(key, float("nan"))
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


@dataclass
class CouplingRuntimeState:
    prev_geometry: Optional[List[List[float]]] = None
    prev_forces: Optional[List[List[float]]] = None
    aitken_relax: float = 1.0


class VarFlExICoupler:
    def __init__(self, config_path: str):
        self.repo_root = os.path.abspath(os.path.dirname(__file__))
        self.config_path = config_path
        self.coupling_params = load_yaml_config(config_path, "coupling_params.yaml")

        self.host=str(cfg_get(self.coupling_params, "host", default="127.0.0.1"))
        self.port=int(cfg_get(self.coupling_params, "port", default=9000))
        self.total_time = float(cfg_get(self.coupling_params, "total_time", default=0.0))
        self.nsteps=int(cfg_get(self.coupling_params,"n_steps",default=cfg_get(self.coupling_params, "nsteps", default=1000)))
        self.dt = self.total_time / self.nsteps if self.total_time > 0.0 and self.nsteps > 0 else float("nan")
        self.force_relax = float(cfg_get(self.coupling_params, "force_relax", default=1.0))
        self.geom_relax = float(cfg_get(self.coupling_params, "geom_relax", default=1.0))
        self.epsilon_reg = float(cfg_get(self.coupling_params, "epsilon_reg", default=1.0e-16))
        self.use_aitken = as_bool(cfg_get(self.coupling_params, "use_aitken", default=False))
        self.debug_io = as_bool(cfg_get(self.coupling_params, "debug_io", default=False))

        results_root = cfg_get(self.coupling_params, "results_root", default=os.path.join(self.repo_root, "results"))
        if os.path.isabs(results_root):
            base_results_root = results_root
        else:
            base_results_root = os.path.abspath(os.path.join(self.repo_root, "..", results_root))
        self.results_dir = os.path.join(base_results_root, "coupling")
        os.makedirs(self.results_dir, exist_ok=True)
        self.log_csv_path = os.path.join(self.results_dir, "coupling_history.csv")
        self.history_jsonl_path = os.path.join(self.results_dir, "forces_sent_received_history.jsonl")

        self.state = CouplingRuntimeState(aitken_relax=self.force_relax)

    def port_owner_hint(self) -> str:
        return f"Port {self.port} is already in use. Stop the old process or choose another port."

    def read_json_line(self, stream, name):
        line = stream.readline()
        if line == "":
            raise RuntimeError(f"{name} disconnected or sent empty line")
        return json.loads(line)

    def accept_role(self, server, fluid_conn, fluid_file, solid_conn, solid_file):
        conn, addr = server.accept()
        stream = conn.makefile("r", encoding="utf-8")
        hello = self.read_json_line(stream, f"Client@{addr}")
        role = str(hello.get("role", "")).strip().lower()

        if role == "fluid":
            if fluid_conn is not None:
                raise RuntimeError("Duplicate fluid connection")
            print(f"Fluid connected from {addr}.")
            return conn, stream, solid_conn, solid_file
        if role == "solid":
            if solid_conn is not None:
                raise RuntimeError("Duplicate solid connection")
            print(f"Solid connected from {addr}.")
            return fluid_conn, fluid_file, conn, stream

        raise RuntimeError(f"Client {addr} must send role handshake {{\"role\":\"fluid\"|\"solid\"}}")

    def bind_server(self):
        print("Starting coupling server...")
        print(
            "Coupling config: "
            f"NSTEPS={self.nsteps}, FORCE_RELAX={self.force_relax}, "
            f"GEOM_RELAX={self.geom_relax}, AITKEN={self.use_aitken}"
        )

        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            server.bind((self.host, self.port))
        except OSError as err:
            if err.errno == errno.EADDRINUSE:
                raise RuntimeError(
                    f"Port {self.port} is already in use on {self.host}.\n"
                    f"{self.port_owner_hint()}\n"
                    f"Stop the old process or run with another port."
                ) from err
            raise
        server.listen(2)
        return server

    def wait_for_clients(self, server):
        fluid_conn, fluid_file = None, None
        solid_conn, solid_file = None, None
        print("Waiting for fluid and solid role handshakes...")
        while fluid_conn is None or solid_conn is None:
            fluid_conn, fluid_file, solid_conn, solid_file = self.accept_role(
                server, fluid_conn, fluid_file, solid_conn, solid_file
            )
        print("Both participants connected.")
        return fluid_conn, fluid_file, solid_conn, solid_file

    def validate_geometry_payload(self, geo_data: Dict[str, Any], geometry: List[List[float]], step: int):
        if not isinstance(geometry, list) or len(geometry) == 0:
            raise RuntimeError(f"Solid sent invalid geometry payload at step {step}")

        n_span = int(geo_data.get("n_span", 0)) if "n_span" in geo_data else 0
        n_chord = int(geo_data.get("n_chord", 0)) if "n_chord" in geo_data else 0
        if n_span > 0 and n_chord > 0:
            expected = n_span * n_chord
            if len(geometry) != expected:
                raise RuntimeError(
                    f"Solid geometry length mismatch at step {step}: "
                    f"received {len(geometry)}, expected {expected} ({n_span}x{n_chord})"
                )

            rot = geo_data.get("rotation", [])
            if isinstance(rot, list) and len(rot) not in (0, expected):
                raise RuntimeError(
                    f"Solid rotation length mismatch at step {step}: "
                    f"received {len(rot)}, expected {expected} ({n_span}x{n_chord})"
                )

            idxing = str(geo_data.get("indexing", "span-major"))
            if idxing not in ("span-major", "chord-major"):
                raise RuntimeError(f"Unsupported indexing '{idxing}' at step {step}")

    def validate_force_payload(self, force_data: Dict[str, Any], forces: List[List[float]], step: int):
        if not isinstance(forces, list):
            raise RuntimeError(f"Fluid sent invalid force payload at step {step}")
        if len(forces) == 0:
            raise RuntimeError(f"Fluid sent empty force payload at step {step}")

        n_span = int(force_data.get("n_span", 0)) if "n_span" in force_data else 0
        n_chord = int(force_data.get("n_chord", 0)) if "n_chord" in force_data else 0
        if n_span > 0 and n_chord > 0:
            expected = n_span * n_chord
            if len(forces) != expected:
                raise RuntimeError(
                    f"Fluid force length mismatch at step {step}: "
                    f"received {len(forces)}, expected {expected} ({n_span}x{n_chord})"
                )

    def relax_forces(self, forces: List[List[float]]) -> Tuple[List[List[float]], float, float, float, float]:
        force_residual, force_reference_norm, force_relative_error = residual_metrics(
            forces, self.state.prev_forces, self.epsilon_reg
        )
        relax_used = self.force_relax

        if self.use_aitken and self.state.prev_forces is not None:
            if force_relative_error > 1.0e-2:
                self.state.aitken_relax = max(0.25, 0.9 * self.state.aitken_relax)
            else:
                self.state.aitken_relax = min(1.0, 1.05 * self.state.aitken_relax)
            relax_used = max(0.2, min(1.0, self.state.aitken_relax))

        if self.state.prev_forces is None or relax_used >= 0.999:
            relaxed = [row[:] for row in forces]
        else:
            relaxed = []
            for f_new, f_old in zip(forces, self.state.prev_forces):
                relaxed.append(
                    [
                        relax_used * float(f_new[0]) + (1.0 - relax_used) * float(f_old[0]),
                        relax_used * float(f_new[1]) + (1.0 - relax_used) * float(f_old[1]),
                        relax_used * float(f_new[2]) + (1.0 - relax_used) * float(f_old[2]),
                    ]
                )

        self.state.prev_forces = [row[:] for row in relaxed]
        return relaxed, relax_used, force_residual, force_reference_norm, force_relative_error

    def send_json(self, conn, payload: Dict[str, Any]):
        conn.sendall((json.dumps(payload) + "\n").encode("utf-8"))

    def run(self):
        server = self.bind_server()
        log_fp = open(self.log_csv_path, "w", newline="", encoding="utf-8")
        history_fp = open(self.history_jsonl_path, "w", encoding="utf-8")
        fluid_conn = fluid_file = solid_conn = solid_file = None

        try:
            writer = csv.writer(log_fp)
            writer.writerow(
                [
                    "step",
                    "time",
                    "n_forces",
                    "force_relax_used",
                    "force_residual",
                    "force_reference_norm",
                    "force_relative_error",
                    "force_transfer_residual",
                    "force_transfer_relative_error",
                    "force_transfer_apply_time",
                    "fluid_force_round_trip_residual",
                    "fluid_force_round_trip_relative_error",
                    "solid_displacement_round_trip_residual",
                    "solid_displacement_round_trip_relative_error",
                    "geometry_transfer_apply_time",
                    "solid_solve_time",
                    "solid_io_time",
                    "crm_setup_time",
                    "rbf_setup_time",
                    "geometry_rbf_setup_time",
                    "geometry_work_Wf",
                    "geometry_work_Ws",
                    "geometry_work_rel_error",
                    "geometry_residual",
                    "geometry_reference_norm",
                    "geometry_relative_error",
                    "geometry_transfer_residual",
                    "geometry_transfer_relative_error",
                    "sample_fx",
                    "sample_fy",
                    "sample_fz",
                    "lift",
                    "drag",
                    "cl",
                    "cd",
                    "solid_step_time",
                    "fluid_step_time",
                    "lift_kj",
                    "drag_kj",
                    "lift_unsteady_circulation",
                    "drag_unsteady_circulation",
                    "lift_parasitic_polar",
                    "drag_parasitic_polar",
                ]
            )

            fluid_conn, fluid_file, solid_conn, solid_file = self.wait_for_clients(server)

            for step in range(1, self.nsteps + 1):
                print(f"\n--- Step {step} ---")
                print("Waiting for geometry from solid...")
                geo_data = self.read_json_line(solid_file, "Solid")
                geometry = vector_rows(geo_data.get("geometry", []), "geometry")
                self.validate_geometry_payload(geo_data, geometry, step)
                print("Geometry received.")

                time_value = step * self.dt if self.dt == self.dt else float("nan")
                (
                    geometry_residual,
                    geometry_reference_norm,
                    geometry_relative_error,
                ) = residual_metrics(geometry, self.state.prev_geometry, self.epsilon_reg)
                self.state.prev_geometry = [row[:] for row in geometry]
                solid_step_time = optional_float(geo_data, "solid_step_time")

                if self.debug_io and geometry:
                    geom_mag = [(g[0] ** 2 + g[1] ** 2 + g[2] ** 2) ** 0.5 for g in geometry]
                    print(f"Geometry max={max(geom_mag):.6e}, mean={sum(geom_mag)/len(geom_mag):.6e}")

                print("Sending geometry to fluid...")
                geo_data["geometry_residual"] = geometry_residual
                geo_data["geometry_reference_norm"] = geometry_reference_norm
                geo_data["geometry_relative_error"] = geometry_relative_error
                self.send_json(fluid_conn, geo_data)
                print(
                    f"Geometry residual={geometry_residual:.3e}, "
                    f"relative={geometry_relative_error:.3e}"
                )

                print("Waiting for forces from fluid...")
                force_data = self.read_json_line(fluid_file, "Fluid")
                forces = vector_rows(force_data.get("force", []), "force")
                self.validate_force_payload(force_data, forces, step)
                print("Forces received.")

                forces_received_raw = [row[:] for row in forces]
                (
                    relaxed_forces,
                    relax_used,
                    force_residual,
                    force_reference_norm,
                    force_relative_error,
                ) = self.relax_forces(forces)
                force_data["force"] = relaxed_forces

                q_inf = force_data.get("q_inf")
                ref_area = force_data.get("ref_area")
                cl, cd = force_coefficients(relaxed_forces, q_inf, ref_area)
                lift, drag = force_components(relaxed_forces)
                fluid_step_time = optional_float(force_data, "fluid_step_time")
                aero_force_components = {
                    key: optional_float(force_data, key)
                    for key in (
                        "lift_kj", "drag_kj", "lift_unsteady_circulation",
                        "drag_unsteady_circulation", "lift_parasitic_polar",
                        "drag_parasitic_polar",
                    )
                }
                force_data["lift"] = lift
                force_data["drag"] = drag
                force_data["cl"] = cl
                force_data["cd"] = cd
                force_data["fluid_step_time"] = fluid_step_time
                force_data["force_residual"] = force_residual
                force_data["force_reference_norm"] = force_reference_norm
                force_data["force_relative_error"] = force_relative_error

                solid_geometry_absolute = (
                    vector_rows(geo_data.get("geometry_le_absolute", []), "geometry_le_absolute")
                    + vector_rows(geo_data.get("geometry_te_absolute", []), "geometry_te_absolute")
                )
                fluid_geometry_absolute = (
                    vector_rows(force_data.get("geometry_le_absolute", []), "geometry_le_absolute")
                    + vector_rows(force_data.get("geometry_te_absolute", []), "geometry_te_absolute")
                )
                (
                    geometry_transfer_residual,
                    geometry_transfer_relative_error,
                ) = transfer_metrics(
                    solid_geometry_absolute,
                    fluid_geometry_absolute,
                    self.epsilon_reg,
                )

                if self.debug_io and relaxed_forces:
                    print(
                        f"Sample force[0] = {relaxed_forces[0]} "
                        f"(relax={relax_used:.3f}, residual={force_residual:.3e}, "
                        f"relative={force_relative_error:.3e})"
                    )

                print("Sending forces to solid...")
                self.send_json(solid_conn, force_data)
                force_transfer_data = self.read_json_line(
                    solid_file, "Solid force-transfer diagnostic"
                )
                if force_transfer_data.get("type") != "force_transfer_diagnostics":
                    raise RuntimeError(
                        "Solid sent an unexpected message while coupling expected "
                        f"force-transfer diagnostics at step {step}: {force_transfer_data}"
                    )
                force_transfer_step = int(force_transfer_data.get("step", -1))
                if force_transfer_step != step:
                    raise RuntimeError(
                        f"Solid force-transfer diagnostic step mismatch: "
                        f"got {force_transfer_step}, expected {step}"
                    )
                force_transfer_residual = optional_float(
                    force_transfer_data, "force_transfer_residual"
                )
                force_transfer_relative_error = optional_float(
                    force_transfer_data, "force_transfer_relative_error"
                )
                force_transfer_apply_time = optional_float(force_transfer_data, "force_transfer_apply_time")
                fluid_force_round_trip_residual = optional_float(
                    force_transfer_data, "fluid_force_round_trip_residual"
                )
                fluid_force_round_trip_relative_error = optional_float(
                    force_transfer_data, "fluid_force_round_trip_relative_error"
                )
                solid_displacement_round_trip_residual = optional_float(
                    force_transfer_data, "solid_displacement_round_trip_residual"
                )
                solid_displacement_round_trip_relative_error = optional_float(
                    force_transfer_data, "solid_displacement_round_trip_relative_error"
                )
                geometry_transfer_apply_time = optional_float(geo_data, "geometry_transfer_apply_time")
                solid_solve_time = optional_float(geo_data, "solid_solve_time")
                solid_io_time = optional_float(geo_data, "solid_io_time")
                crm_setup_time = optional_float(force_transfer_data, "crm_setup_time")
                rbf_setup_time = optional_float(force_transfer_data, "rbf_setup_time")
                geometry_rbf_setup_time = optional_float(force_transfer_data, "geometry_rbf_setup_time")
                geometry_work_Wf = optional_float(force_transfer_data, "geometry_work_Wf")
                geometry_work_Ws = optional_float(force_transfer_data, "geometry_work_Ws")
                geometry_work_rel_error = optional_float(
                    force_transfer_data, "geometry_work_rel_error"
                )

                json.dump(
                    {
                        "step": step,
                        "time": time_value,
                        "n_span": force_data.get("n_span"),
                        "n_chord": force_data.get("n_chord"),
                        "indexing": force_data.get("indexing", "span-major"),
                        "force_relax_used": relax_used,
                        "force_residual": force_residual,
                        "force_reference_norm": force_reference_norm,
                        "force_relative_error": force_relative_error,
                        "force_transfer_residual": force_transfer_residual,
                        "force_transfer_relative_error": force_transfer_relative_error,
                        "force_transfer_apply_time": force_transfer_apply_time,
                        "fluid_force_round_trip_residual": fluid_force_round_trip_residual,
                        "fluid_force_round_trip_relative_error": fluid_force_round_trip_relative_error,
                        "solid_displacement_round_trip_residual": solid_displacement_round_trip_residual,
                        "solid_displacement_round_trip_relative_error": solid_displacement_round_trip_relative_error,
                        "geometry_transfer_apply_time": geometry_transfer_apply_time,
                        "solid_solve_time": solid_solve_time,
                        "solid_io_time": solid_io_time,
                        "crm_setup_time": crm_setup_time,
                        "rbf_setup_time": rbf_setup_time,
                        "geometry_rbf_setup_time": geometry_rbf_setup_time,
                        "geometry_work_Wf": geometry_work_Wf,
                        "geometry_work_Ws": geometry_work_Ws,
                        "geometry_work_rel_error": geometry_work_rel_error,
                        "geometry_residual": geometry_residual,
                        "geometry_reference_norm": geometry_reference_norm,
                        "geometry_relative_error": geometry_relative_error,
                        "geometry_transfer_residual": geometry_transfer_residual,
                        "geometry_transfer_relative_error": geometry_transfer_relative_error,
                        "force_received": forces_received_raw,
                        "force_sent": relaxed_forces,
                        "lift": lift,
                        "drag": drag,
                        "cl": cl,
                        "cd": cd,
                        "solid_step_time": solid_step_time,
                        "fluid_step_time": fluid_step_time,
                        "force_components": aero_force_components,
                    },
                    history_fp,
                )
                history_fp.write("\n")

                sample = relaxed_forces[0]
                writer.writerow(
                    [
                        step,
                        f"{time_value:.6e}" if time_value == time_value else "nan",
                        len(relaxed_forces),
                        f"{relax_used:.6f}",
                        f"{force_residual:.6e}",
                        f"{force_reference_norm:.6e}",
                        f"{force_relative_error:.6e}",
                        f"{force_transfer_residual:.6e}" if force_transfer_residual == force_transfer_residual else "nan",
                        f"{force_transfer_relative_error:.6e}" if force_transfer_relative_error == force_transfer_relative_error else "nan",
                        f"{force_transfer_apply_time:.6e}" if force_transfer_apply_time == force_transfer_apply_time else "nan",
                        f"{fluid_force_round_trip_residual:.6e}" if fluid_force_round_trip_residual == fluid_force_round_trip_residual else "nan",
                        f"{fluid_force_round_trip_relative_error:.6e}" if fluid_force_round_trip_relative_error == fluid_force_round_trip_relative_error else "nan",
                        f"{solid_displacement_round_trip_residual:.6e}" if solid_displacement_round_trip_residual == solid_displacement_round_trip_residual else "nan",
                        f"{solid_displacement_round_trip_relative_error:.6e}" if solid_displacement_round_trip_relative_error == solid_displacement_round_trip_relative_error else "nan",
                        f"{geometry_transfer_apply_time:.6e}" if geometry_transfer_apply_time == geometry_transfer_apply_time else "nan",
                        f"{solid_solve_time:.6e}" if solid_solve_time == solid_solve_time else "nan",
                        f"{solid_io_time:.6e}" if solid_io_time == solid_io_time else "nan",
                        f"{crm_setup_time:.6e}" if crm_setup_time == crm_setup_time else "nan",
                        f"{rbf_setup_time:.6e}" if rbf_setup_time == rbf_setup_time else "nan",
                        f"{geometry_rbf_setup_time:.6e}" if geometry_rbf_setup_time == geometry_rbf_setup_time else "nan",
                        f"{geometry_work_Wf:.6e}" if geometry_work_Wf == geometry_work_Wf else "nan",
                        f"{geometry_work_Ws:.6e}" if geometry_work_Ws == geometry_work_Ws else "nan",
                        f"{geometry_work_rel_error:.6e}" if geometry_work_rel_error == geometry_work_rel_error else "nan",
                        f"{geometry_residual:.6e}",
                        f"{geometry_reference_norm:.6e}",
                        f"{geometry_relative_error:.6e}",
                        f"{geometry_transfer_residual:.6e}" if geometry_transfer_residual == geometry_transfer_residual else "nan",
                        f"{geometry_transfer_relative_error:.6e}" if geometry_transfer_relative_error == geometry_transfer_relative_error else "nan",
                        f"{float(sample[0]):.6e}",
                        f"{float(sample[1]):.6e}",
                        f"{float(sample[2]):.6e}",
                        f"{float(lift):.6e}",
                        f"{float(drag):.6e}",
                        f"{float(cl):.6e}" if cl == cl else "nan",
                        f"{float(cd):.6e}" if cd == cd else "nan",
                        f"{solid_step_time:.6e}" if solid_step_time == solid_step_time else "nan",
                        f"{fluid_step_time:.6e}" if fluid_step_time == fluid_step_time else "nan",
                        *[
                            f"{value:.6e}" if value == value else "nan"
                            for value in aero_force_components.values()
                        ],
                    ]
                )

            print("Coupling finished.")
            print(f"Coupling diagnostics saved at: {self.log_csv_path}")
            print(f"Force transfer history saved at: {self.history_jsonl_path}")
        finally:
            try:
                if fluid_file is not None:
                    fluid_file.close()
            finally:
                try:
                    if solid_file is not None:
                        solid_file.close()
                finally:
                    try:
                        if fluid_conn is not None:
                            fluid_conn.close()
                    finally:
                        try:
                            if solid_conn is not None:
                                solid_conn.close()
                        finally:
                            server.close()
                            log_fp.close()
                            history_fp.close()


def main():
    parser=argparse.ArgumentParser(description="coupling server params path")
    default_config_path=os.path.join(os.path.abspath(os.path.dirname(__file__)), "config")
    parser.add_argument("--config_path", type=str, default=default_config_path)
    args=parser.parse_args()

    coupler=VarFlExICoupler(args.config_path)
    coupler.run()


if __name__ == "__main__":
    main()
