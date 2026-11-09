#!/usr/bin/env python3
"""Check production CRM/RBF transfer operators on the coupled MMS target.

The geometry comparison uses the analytical target at the production leading-
and trailing-edge stations.  The load comparison uses the exposed Galerkin
operator for CRM and the net-force plus virtual-work invariants for RBF.
"""
import json
from pathlib import Path
import sys

import numpy as np
import yaml
from dolfin import Constant

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "solid"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from coupled_target import build_default_target
from coupled_solid_source import _target_function
from coupling.utils.coupling_transfer import CouplingTransfer
from utils import ReissnerMindlinModel, SolidDomain


def relative_norm(actual, expected):
    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)
    return float(np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1.0e-14))


def make_model(domain, cfg, dt):
    return ReissnerMindlinModel(
        domain, cfg["E"], cfg["nu"], cfg["rho_s"], cfg["eta_m"], cfg["eta_k"],
        cfg["kappa_shear"], Constant(dt), cfg.get("alpha_m", 0.1), cfg.get("alpha_f", 0.2),
        geometric_nonlinearity=cfg.get("geometric_nonlinearity", True),
        include_quadratic_inplane_strain=cfg.get("include_quadratic_inplane_strain", True),
    )


def target_displacement(target, points, time):
    """Production interface displacement map at physical edge stations."""
    out = np.empty((len(points), 3), dtype=float)
    for index, point in enumerate(np.asarray(points, dtype=float)):
        out[index] = target.geometry.point(point[0], point[1], time) - point
    return out


def main():
    config = ROOT / "config"
    with (config / "solid_params.yaml").open(encoding="utf-8") as stream:
        solid_cfg = yaml.safe_load(stream)
    with (config / "coupling_params.yaml").open(encoding="utf-8") as stream:
        coupling_cfg = yaml.safe_load(stream)

    time = 0.005
    target = build_default_target(config, 0.01)
    domain = SolidDomain(
        solid_cfg["span"], solid_cfg["root_chord"], solid_cfg["tip_chord"],
        solid_cfg["thickness_ratio"], solid_cfg.get("leading_edge_sweep", 0.0),
        8, 16, target.n_span, "midpoint",
    )
    model = make_model(domain, solid_cfg, dt=0.01)
    spaces = [model.V.sub(index).collapse() for index in range(3)]
    q_star = _target_function(model, target, t=time, order=0, spaces=spaces)
    panel_forces = target.panel_force(time)
    rows = []

    for mode in ("crm", "rbf"):
        transfer = CouplingTransfer(
            domain, model, eta_span_comm=target.y_stations / target.structural.Ly,
            eta_cp=0.75, force_transfer_mode=mode, displacement_transfer_mode=mode,
            rbf_radius=float(coupling_cfg.get("rbf_radius", 0.08)),
            rbf_neighbors=int(coupling_cfg.get("rbf_neighbors", 24)),
            max_abs_force_component=None, edge_eval_xi_eps=1.0e-6,
            work_conservative_mode=True,
            enforce_chord_projection=False, enforce_span_projection=False,
        )
        u_le, u_te, _, _ = transfer.apply_geometry_transfer(q_star)
        u_le_expected = target_displacement(target, transfer.le_targets, time)
        u_te_expected = target_displacement(target, transfer.te_targets, time)
        geometry_actual = np.vstack((u_le, u_te))
        geometry_expected = np.vstack((u_le_expected, u_te_expected))
        geom_rel = relative_norm(geometry_actual, geometry_expected)

        coefficients, nodal = transfer.apply_force_transfer(panel_forces)
        nodal = np.asarray(nodal, dtype=float)
        if mode == "crm":
            expected_nodal = transfer.crm_transfer_matrix @ (
                panel_forces / transfer.crm_panel_areas[:, None]
            )
            force_rel = relative_norm(nodal, expected_nodal)
        else:
            # The RBF backend exposes a lumped dual operator. Its nodal RHS
            # preserves total panel load because the interpolation weights
            # form a partition of unity.
            force_rel = relative_norm(np.sum(nodal, axis=0), np.sum(panel_forces, axis=0))

        index = np.arange(len(nodal), dtype=float)
        delta_nodal = np.column_stack((1.0e-6*(1.0 + index),
                                       -0.5e-6*(1.0 + index),
                                       0.25e-6*(1.0 + index)))
        _wf, _ws, _defect, work_rel = transfer.incremental_work_metrics(
            delta_nodal, panel_forces, nodal
        )
        threshold = 1.0e-10 if mode == "crm" else 1.0e-3
        work_threshold = 1.0e-12 if mode == "crm" else 1.0e-4
        row = {
            "mode": mode, "geom_rel": geom_rel, "force_rel": force_rel,
            "work_rel": float(work_rel), "threshold_geom": threshold,
            "threshold_force": threshold,
        }
        rows.append(row)
        if not (np.isfinite(geom_rel) and np.isfinite(force_rel) and np.isfinite(work_rel)):
            raise AssertionError(f"{mode}: transfer check produced a non-finite metric")
        if geom_rel > threshold or force_rel > threshold or work_rel > work_threshold:
            raise AssertionError(
                f"{mode}: geom_rel={geom_rel:.3e}, force_rel={force_rel:.3e}, "
                f"work_rel={work_rel:.3e}; thresholds={threshold:.3e}, {work_threshold:.3e}"
            )
        print(f"PASS checks_transfer_mms mode={mode}")

    output = ROOT / "verification/mms/results/pipeline/checks_transfer_mms.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
