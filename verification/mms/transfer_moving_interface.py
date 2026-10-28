#!/usr/bin/env python3
"""Exercise the production CRM/RBF transfer implementations on moving fields.

Run in the legacy DOLFIN environment. This is a transfer component test, not a
coupled FSI MMS. Results go to a caller-selected CSV and JSON file.
"""

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "solid"))

from dolfin import Constant, Expression, Function, FunctionAssigner, FunctionSpace, interpolate
from utils import ReissnerMindlinModel, SolidDomain
from coupling.utils.coupling_transfer import CouplingTransfer


def parse_levels(value):
    levels = []
    for item in value.split(","):
        nx, ny = (int(part) for part in item.lower().split("x"))
        if nx < 2 or ny < 2:
            raise ValueError(f"invalid mesh level {item!r}")
        levels.append((nx, ny))
    if len(levels) < 2:
        raise ValueError("at least two mesh levels are required")
    return levels


def make_model(domain, cfg, dt):
    return ReissnerMindlinModel(
        domain=domain,
        E=cfg["E"], nu=cfg["nu"], rho_s=cfg["rho_s"],
        eta_m=cfg["eta_m"], eta_k=cfg["eta_k"],
        kappa_shear=cfg["kappa_shear"], dt=Constant(dt),
        alpha_m=cfg.get("alpha_m", 0.1), alpha_f=cfg.get("alpha_f", 0.2),
        geometric_nonlinearity=cfg.get("geometric_nonlinearity", True),
        include_quadratic_inplane_strain=cfg.get("include_quadratic_inplane_strain", True),
    )


def exact_displacement(points, amplitude, chord, span, phase):
    points = np.asarray(points, dtype=float)
    x = points[:, 0] / chord
    eta = points[:, 1] / span
    return np.column_stack((
        amplitude * x**2 * (1.0 + 0.3 * np.sin(2.0 * np.pi * eta)) * np.cos(phase),
        amplitude * 0.2 * x * np.sin(np.pi * eta) * np.sin(phase),
        amplitude * (0.3 + x) * np.sin(np.pi * eta) * np.cos(phase),
    ))


def set_moving_state(model, chord, span, amplitude, phase):
    spaces = [model.V.sub(i).collapse() for i in range(3)]
    assigner = FunctionAssigner(model.V, spaces)
    fields = [
        interpolate(Expression((
            "A*pow(x[0]/Lc,2)*(1.0+0.3*sin(2*pi*x[1]/Ly))*cos(ph)",
            "A*0.2*(x[0]/Lc)*sin(pi*x[1]/Ly)*sin(ph)"),
            degree=8, A=amplitude, Lc=chord, Ly=span, ph=phase), spaces[0]),
        interpolate(Expression(
            "A*(0.3+x[0]/Lc)*sin(pi*x[1]/Ly)*cos(ph)",
            degree=8, A=amplitude, Lc=chord, Ly=span, ph=phase), spaces[1]),
        interpolate(Expression(("0.0", "0.0"), degree=1), spaces[2]),
    ]
    q = Function(model.V)
    assigner.assign(q, fields)
    return q


def build_transfer(domain, model, eta, mode, coupling_cfg):
    return CouplingTransfer(
        domain=domain, model=model, eta_span_comm=eta, eta_cp=0.75,
        force_transfer_mode=mode,
        rbf_radius=float(coupling_cfg.get("rbf_radius", 0.08)),
        rbf_neighbors=int(coupling_cfg.get("rbf_neighbors", 24)),
        max_abs_force_component=None,
        edge_eval_xi_eps=1.0e-6,
        work_conservative_mode=True,
        enforce_chord_projection=False, enforce_span_projection=False,
    )


def run(config_dir, levels, csv_path, json_path):
    with open(Path(config_dir) / "solid_params.yaml") as stream:
        solid = yaml.safe_load(stream)
    with open(Path(config_dir) / "coupling_params.yaml") as stream:
        coupling = yaml.safe_load(stream)

    rows = []
    for nx, ny in levels:
        nstations = max(4, ny // 2 + 1)
        domain = SolidDomain(
            span=solid["span"], root_chord=solid["root_chord"],
            tip_chord=solid["tip_chord"], thickness_ratio=solid["thickness_ratio"],
            leading_edge_sweep=solid.get("leading_edge_sweep", 0.0),
            nx=nx, ny=ny, n_span_comm=nstations, span_sampling_mode="midpoint",
        )
        model = make_model(domain, solid, dt=1.0e-3)
        for mode in ("crm", "rbf"):
            transfer = build_transfer(domain, model, domain.eta_span_comm, mode, coupling)
            _, crm_areas, _ = transfer.crm_backend.build_common_refinement_operator()
            geom_errors = []
            geometry_states = (
                ("undeformed", 0.0, 0.0),
                ("deformed_a", 0.01 * domain.root_chord, 0.37),
                ("deformed_b", 0.01 * domain.root_chord, 1.13),
            )
            for _state_name, amplitude, phase in geometry_states:
                q = set_moving_state(
                    model, domain.root_chord, domain.span,
                    amplitude, phase,
                )
                u_le, u_te, _, _ = transfer.apply_geometry_transfer(q)
                le_exact = exact_displacement(
                    transfer.le_targets, amplitude,
                    domain.root_chord, domain.span, phase,
                )
                te_exact = exact_displacement(
                    transfer.te_targets, amplitude,
                    domain.root_chord, domain.span, phase,
                )
                geom_errors.append(np.concatenate((u_le - le_exact, u_te - te_exact)))

            cp = transfer.cp_targets
            eta = cp[:, 1] / domain.span
            force_cases = {
                "constant": np.tile(np.array([1.0, -0.4, 0.7]), (len(cp), 1)),
                "varying": np.column_stack((
                    1.0 + 0.4 * np.sin(2.0 * np.pi * eta),
                    -0.4 + 0.2 * np.cos(np.pi * eta),
                    0.7 * (1.0 - eta) + 0.1 * np.sin(np.pi * eta),
                )),
            }
            for force_name, forces in force_cases.items():
                coefficients, nodal_forces = transfer.apply_force_transfer(forces)
                if coefficients is None:
                    coefficients = nodal_forces
                solid_coords_2d = transfer.interface_coords
                if len(solid_coords_2d) != len(nodal_forces) and mode == "crm":
                    solid_coords_2d = FunctionSpace(domain.mesh, "CG", 1).tabulate_dof_coordinates().reshape(-1, 2)
                if len(solid_coords_2d) != len(nodal_forces):
                    raise RuntimeError(
                        f"{mode} force/coordinate layout mismatch: forces={nodal_forces.shape}, "
                        f"interface_coords={transfer.interface_coords.shape}, "
                        f"CG1_coords={solid_coords_2d.shape}"
                    )
                solid_coords = np.column_stack((solid_coords_2d, np.zeros(len(solid_coords_2d))))
                solid_displacement = transfer.sample_vector_field_at_targets(q, solid_coords)
                solid_coords_moved = solid_coords + solid_displacement
                cp_coords_moved = cp + exact_displacement(
                    cp, 0.01 * domain.root_chord, domain.root_chord, domain.span, 1.13,
                )
                displacement = np.column_stack((
                    np.sin(2.0 * np.pi * solid_coords[:, 1] / domain.span),
                    solid_coords[:, 0] / domain.root_chord,
                    np.cos(np.pi * solid_coords[:, 1] / domain.span),
                ))
                work = transfer.incremental_work_metrics(
                    displacement, forces, nodal_forces,
                )
                solid_weights = (
                    np.ones(len(nodal_forces)) if mode == "crm" else transfer.S_lumped
                )
                coefficient_work_solid = float(np.sum(
                    displacement * coefficients * solid_weights[:, None]
                ))
                rng = np.random.default_rng(73021 + nx + len(forces))
                random_displacement = rng.standard_normal(displacement.shape)
                random_work = transfer.incremental_work_metrics(
                    random_displacement, forces, nodal_forces,
                )
                random_coefficient_work_solid = float(np.sum(
                    random_displacement * coefficients * solid_weights[:, None]
                ))
                wrench = transfer.wrench_transfer_metrics(
                    cp_coords_moved, forces, solid_coords_moved, nodal_forces,
                )
                coefficient_resultant = (
                    np.sum(coefficients, axis=0) if mode == "crm"
                    else np.sum(coefficients * transfer.S_lumped[:, None], axis=0)
                )
                target_resultant = np.sum(forces * transfer.A_diag[:, None], axis=0)
                generalized_resultant = np.sum(nodal_forces, axis=0)
                rows.append({
                    "nx": nx, "ny": ny, "stations": nstations, "backend": mode,
                    "force_case": force_name,
                    "geometry_states": ";".join(name for name, _, _ in geometry_states),
                    "geometry_rms_error_m": float(np.sqrt(np.mean(np.concatenate(geom_errors) ** 2))),
                    "geometry_max_error_m": float(np.max(np.linalg.norm(np.concatenate(geom_errors).reshape(-1, 3), axis=1))),
                    "force_resultant_relative_error": float(
                        np.linalg.norm(generalized_resultant - target_resultant)
                        / max(np.linalg.norm(target_resultant), 1.0e-16)
                    ),
                    "coefficient_weighted_resultant_relative_error": float(
                        np.linalg.norm(coefficient_resultant - target_resultant)
                        / max(np.linalg.norm(target_resultant), 1.0e-16)
                    ),
                    "force_resultant_relative_error_diagnostic": wrench[6],
                    "moment_relative_error_diagnostic": wrench[7],
                    "wrench_coordinates": "deformed reference coordinates; operator weights remain reference-parameterized",
                    "virtual_work_fluid": work[0], "virtual_work_solid": work[1],
                    "virtual_work_relative_error_diagnostic": work[3],
                    "coefficient_virtual_work_relative_error": float(
                        abs(work[0] - coefficient_work_solid)
                        / max(abs(work[0]), abs(coefficient_work_solid), 1.0e-16)
                    ),
                    "random_virtual_work_relative_error_diagnostic": random_work[3],
                    "random_coefficient_virtual_work_relative_error": float(
                        abs(random_work[0] - random_coefficient_work_solid)
                        / max(abs(random_work[0]), abs(random_coefficient_work_solid), 1.0e-16)
                    ),
                    "lumped_weight_min": float(np.min(solid_weights)),
                    "lumped_weight_max": float(np.max(solid_weights)),
                    "crm_area_sum": float(np.sum(crm_areas)),
                    "covered_reference_area": float(domain.span * domain.root_chord),
                    "rbf_radius_m": transfer.rbf_radius,
                    "rbf_neighbors": transfer.rbf_neighbors,
                })

    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "status": "COMPONENT_ONLY",
        "claim": "Exercises production CouplingTransfer/CRM/RBF methods; not coupled FSI verification.",
        "mesh_levels": [list(level) for level in levels],
        "solid_config": str((Path(config_dir) / "solid_params.yaml").resolve()),
        "coupling_config": str((Path(config_dir) / "coupling_params.yaml").resolve()),
        "metrics": rows,
    }
    Path(json_path).write_text(json.dumps(payload, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", default=str(ROOT / "config"))
    parser.add_argument("--levels", default="8x16,16x32")
    parser.add_argument("--csv", default=str(Path(__file__).parent / "results" / "transfer_moving_interface.csv"))
    parser.add_argument("--json", default=str(Path(__file__).parent / "results" / "transfer_moving_interface.json"))
    args = parser.parse_args()
    run(args.config_dir, parse_levels(args.levels), args.csv, args.json)
    print(f"Wrote production transfer audit: {args.csv} and {args.json}")


if __name__ == "__main__":
    main()
