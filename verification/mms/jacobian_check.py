#!/usr/bin/env python3
"""Finite-difference check of the production nonlinear plate residual tangent.

This checks derivative consistency only; it is not an independent check that
the residual represents the intended PDE. The latter is ``strong_weak_check``.
Requires legacy DOLFIN and uses the production residual/Jacobian forms.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "solid"))

from plate import make_plate_mms

try:
    from dolfin import Constant, FunctionAssigner, assemble
except ImportError as exc:
    raise SystemExit(f"Plate Jacobian check requires legacy dolfin: {exc}") from exc

from utils import ReissnerMindlinModel, SolidDomain
from fenics_plate import make_mixed_function


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", default=str(ROOT / "config"))
    parser.add_argument("--output", default=str(HERE / "results" / "plate_jacobian_check.csv"))
    args = parser.parse_args()
    with open(Path(args.config_dir) / "solid_params.yaml") as stream:
        cfg = yaml.safe_load(stream)
    domain = SolidDomain(cfg["span"], cfg["root_chord"], cfg["tip_chord"], cfg["thickness_ratio"],
                         cfg.get("leading_edge_sweep", 0.0), 8, 16, 16, "midpoint")
    model = ReissnerMindlinModel(
        domain, cfg["E"], cfg["nu"], cfg["rho_s"], cfg["eta_m"], cfg["eta_k"],
        cfg["kappa_shear"], Constant(0.001), cfg.get("alpha_m", 0.1), cfg.get("alpha_f", 0.2),
        geometric_nonlinearity=True,
        include_quadratic_inplane_strain=cfg.get("include_quadratic_inplane_strain", True),
    )
    mms = make_plate_mms(cfg, True, cfg.get("include_quadratic_inplane_strain", True))
    spaces = [model.V.sub(i).collapse() for i in range(3)]
    assigner = FunctionAssigner(model.V, spaces)
    base = make_mixed_function(model, mms.fields, 0.37, spaces, assigner)
    direction = make_mixed_function(model, mms.fields, 0.61, spaces, assigner)
    model.q.assign(base)
    base_values = model.q.vector().get_local().copy()
    direction_values = direction.vector().get_local().copy()
    tangent = assemble(model.jacobian)
    tangent_direction = direction.vector().copy()
    tangent.mult(direction.vector(), tangent_direction)
    tangent_values = tangent_direction.get_local()
    rows = []
    for epsilon in (1.0e-2, 3.0e-3, 1.0e-3, 3.0e-4, 1.0e-4, 3.0e-5):
        model.q.vector().set_local(base_values + epsilon * direction_values)
        model.q.vector().apply("insert")
        plus = assemble(model.residual).get_local()
        model.q.vector().set_local(base_values - epsilon * direction_values)
        model.q.vector().apply("insert")
        minus = assemble(model.residual).get_local()
        model.q.vector().set_local(base_values)
        finite_difference = (plus - minus) / (2.0 * epsilon)
        error = np.linalg.norm(finite_difference - tangent_values)
        scale = max(np.linalg.norm(finite_difference), np.linalg.norm(tangent_values), 1.0e-30)
        rows.append({"epsilon": epsilon, "relative_jacobian_direction_error": error / scale})
    model.q.vector().apply("insert")
    best = min(row["relative_jacobian_direction_error"] for row in rows)
    if not np.isfinite(best) or best > 2.0e-5:
        raise AssertionError(f"Production Jacobian directional check failed: best relative error={best:.4e}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Production nonlinear residual/Jacobian check passed; best relative discrepancy={best:.3e}")


if __name__ == "__main__":
    main()
