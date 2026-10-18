#!/usr/bin/env python3
"""Compare independently derived plate strong loads to production weak forms.

The strong differential operators and natural boundary quantities are built
symbolically in ``plate.py``. The other side calls the production model's mass,
damping, and internal virtual-work forms. Multiple admissible tests are used.
Both sides are evaluated at the same physical time (0.37 s). The strong load
is independently derived in plate.py; the production weak form is not used to
construct that load. This is a weak/strong identity test, not a discrete
solution or convergence test. Requires the legacy DOLFIN environment.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "solid"))

from plate import make_plate_mms

try:
    from dolfin import (Constant, Expression, FunctionSpace, VectorFunctionSpace,
                        assemble, dot, dx, interpolate)
except ImportError as exc:
    raise SystemExit(f"Strong/weak plate check requires legacy dolfin: {exc}") from exc

from utils import ReissnerMindlinModel, SolidDomain
from fenics_plate import scalar_expr, vector_expr


def run(config_dir, output):
    with open(Path(config_dir) / "solid_params.yaml") as stream:
        cfg = yaml.safe_load(stream)
    domain = SolidDomain(cfg["span"], cfg["root_chord"], cfg["tip_chord"], cfg["thickness_ratio"],
                         cfg.get("leading_edge_sweep", 0.0), 12, 24, 24, "midpoint")
    result_rows = []
    for label, nonlinear in (("linear", False), ("nonlinear", True)):
        model = ReissnerMindlinModel(
            domain, cfg["E"], cfg["nu"], cfg["rho_s"], cfg["eta_m"], cfg["eta_k"],
            cfg["kappa_shear"], Constant(0.001), cfg.get("alpha_m", 0.1), cfg.get("alpha_f", 0.2),
            geometric_nonlinearity=nonlinear,
            include_quadratic_inplane_strain=cfg.get("include_quadratic_inplane_strain", True),
        )
        mms = make_plate_mms(cfg, nonlinear, cfg.get("include_quadratic_inplane_strain", True))
        boundary_values = [value for group in mms.boundary_checks.values()
                           for value in (group if isinstance(group, list) else [group])]
        if any(float(value) != 0.0 for value in boundary_values):
            raise AssertionError(f"Manufactured boundary conditions are nonzero: {boundary_values}")
        fields = mms.fields
        velocity = tuple(v.diff(mms.t) for v in fields)
        acceleration = tuple(v.diff(mms.t, 2) for v in fields)
        # The production UFL operators need mesh-attached fields for grad().
        # The production forms are exercised on mesh-attached CG2 fields.
        # This keeps the verification in the solver's actual approximation
        # space; the independent strong load is integrated against the same
        # test functions, so interpolation error is part of the reported gap.
        ref_spaces = (VectorFunctionSpace(domain.mesh, "CG", 2, dim=2),
                      FunctionSpace(domain.mesh, "CG", 2),
                      VectorFunctionSpace(domain.mesh, "CG", 2, dim=2))
        test_spaces = [model.V.sub(i).collapse() for i in range(3)]

        def projected_state(source_fields):
            return (
                interpolate(vector_expr(source_fields[0], 0.37), ref_spaces[0]),
                interpolate(scalar_expr(source_fields[1], 0.37), ref_spaces[1]),
                interpolate(vector_expr(source_fields[2], 0.37), ref_spaces[2]),
            )

        exact_state = projected_state(fields)
        velocity_state = projected_state(velocity)
        acceleration_state = projected_state(acceleration)
        tests = (
            ("test_a", "(1+x[0]/Lx)*sin(pi*x[1]/(2*Ly))", "(1-x[0]/Lx)*sin(pi*x[1]/(2*Ly))"),
            ("test_b", "sin(2*pi*x[0]/Lx)*sin(pi*x[1]/Ly)", "cos(pi*x[0]/Lx)*sin(pi*x[1]/Ly)"),
            ("test_c", "pow(x[0]/Lx,2)*sin(3*pi*x[1]/(2*Ly))", "sin(pi*x[0]/Lx)*sin(3*pi*x[1]/(2*Ly))"),
        )
        for name, shape_x, shape_y in tests:
            test_u = Expression((f"1e-4*({shape_x})", f"1e-4*({shape_y})"), degree=8,
                                Lx=cfg["root_chord"], Ly=cfg["span"])
            test_w = Expression(f"1e-3*({shape_x})", degree=8,
                                Lx=cfg["root_chord"], Ly=cfg["span"])
            test_theta = Expression((f"1e-3*({shape_y})", f"-1e-3*({shape_x})"), degree=8,
                                    Lx=cfg["root_chord"], Ly=cfg["span"])
            test = (interpolate(test_u, test_spaces[0]), interpolate(test_w, test_spaces[1]),
                    interpolate(test_theta, test_spaces[2]))
            weak = model.manufactured_weak_load(
                exact_state, velocity_state, acceleration_state, test
            )
            load = (
                dot(vector_expr(mms.loads[0], 0.37), test[0])
                + scalar_expr(mms.loads[1], 0.37) * test[1]
                + dot(vector_expr(mms.loads[2], 0.37), test[2])
            ) * dx
            weak_value, load_value = float(assemble(weak)), float(assemble(load))
            discrepancy = abs(weak_value - load_value)
            scale = max(abs(weak_value), abs(load_value), 1.0e-20)
            row = {"formulation": label, "test": name, "production_weak_value": weak_value,
                   "independent_strong_load_value": load_value, "absolute_discrepancy": discrepancy,
                   "relative_discrepancy": discrepancy / scale,
                   "natural_boundary_conditions": "symbolically zero on all free edges"}
            result_rows.append(row)
            if row["relative_discrepancy"] > 2.0e-5:
                raise AssertionError(f"Strong/weak identity failed: {row}")

    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result_rows[0]))
        writer.writeheader()
        writer.writerows(result_rows)
    path.with_suffix(".json").write_text(json.dumps({"status": "PASS", "checks": result_rows}, indent=2))
    print(f"Independent plate strong/weak checks passed: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", default=str(ROOT / "config"))
    parser.add_argument("--output", default=str(HERE / "results" / "plate_strong_weak.csv"))
    args = parser.parse_args()
    run(args.config_dir, args.output)


if __name__ == "__main__":
    main()
