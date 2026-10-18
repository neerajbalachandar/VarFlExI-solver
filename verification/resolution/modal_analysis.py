#!/usr/bin/env python3
"""Compute low plate modes from the production linearized FEM operators.

Requires legacy FEniCS with SLEPc. Optionally evaluates the nonlinear tangent
at a prescribed transverse displacement amplitude.
"""

import argparse
import csv
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", default=str(ROOT / "config"))
    parser.add_argument("--nx", type=int)
    parser.add_argument("--ny", type=int)
    parser.add_argument("--modes", type=int, default=12)
    parser.add_argument("--transverse-amplitude", type=float, default=0.0, help="Evaluate nonlinear tangent at this imposed w amplitude [m].")
    parser.add_argument("--output", default=str(Path(__file__).parent / "results" / "plate_modes.csv"))
    args = parser.parse_args()
    try:
        from dolfin import (Constant, Expression, Function, FunctionAssigner,
                            SLEPcEigenSolver, assemble, derivative, parameters,
                            TrialFunction, TestFunction)
    except ImportError as exc:
        raise SystemExit(f"Modal analysis requires legacy dolfin/SLEPc: {exc}") from exc
    sys.path.insert(0, str(ROOT / "solid"))
    from utils import ReissnerMindlinModel, SolidDomain
    with open(Path(args.config_dir) / "solid_params.yaml") as stream:
        cfg = yaml.safe_load(stream)
    nx, ny = args.nx or int(cfg["nx"]), args.ny or int(cfg["ny"])
    domain = SolidDomain(cfg["span"], cfg["root_chord"], cfg["tip_chord"], cfg["thickness_ratio"],
                         cfg.get("leading_edge_sweep", 0.0), nx, ny, min(ny, 40), "midpoint")
    model = ReissnerMindlinModel(domain, cfg["E"], cfg["nu"], cfg["rho_s"], cfg["eta_m"], cfg["eta_k"],
                                 cfg["kappa_shear"], Constant(1e-3), cfg.get("alpha_m", 0.1),
                                 cfg.get("alpha_f", 0.2), geometric_nonlinearity=cfg.get("geometric_nonlinearity", True),
                                 include_quadratic_inplane_strain=cfg.get("include_quadratic_inplane_strain", True))
    if args.transverse_amplitude:
        spaces = [model.V.sub(i).collapse() for i in range(3)]
        qparts = [Function(spaces[0]), Function(spaces[1]), Function(spaces[2])]
        w_expr = Expression(
            "amp*pow(sin(pi*x[0]/Lx),4)*pow(sin(pi*x[1]/Ly),4)", degree=6,
            amp=args.transverse_amplitude, Lx=cfg["root_chord"], Ly=cfg["span"],
        )
        qparts[1].interpolate(w_expr)
        FunctionAssigner(model.V, spaces).assign(model.q, qparts)
    if model.geometric_nonlinearity:
        internal = model.internal_work(model.q, model.q_test)
        k_form = derivative(internal, model.q, model.dq_trial)
    else:
        k_form = model.stiffness_form(model.dq_trial, model.q_test)
    m_form = model.mass_form(model.dq_trial, model.q_test)
    K, M = assemble(k_form), assemble(m_form)
    for bc in model.bcs:
        bc.apply(K)
        bc.zero(M)
    eigensolver = SLEPcEigenSolver(K, M)
    eigensolver.parameters["problem_type"] = "gen_hermitian"
    eigensolver.parameters["spectrum"] = "smallest real"
    eigensolver.solve(args.modes)
    rows = []
    for i in range(eigensolver.get_number_converged()):
        value, _, _, _ = eigensolver.get_eigenpair(i)
        if value <= 0:
            continue
        rows.append({"mode": i + 1, "eigenvalue_rad2_s2": value,
                     "frequency_hz": value**0.5/(2*3.141592653589793),
                     "nx": nx, "ny": ny, "transverse_tangent_amplitude_m": args.transverse_amplitude,
                     "tangent": "nonlinear" if model.geometric_nonlinearity else "linear"})
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} modal frequencies to {out}")


if __name__ == "__main__":
    main()
