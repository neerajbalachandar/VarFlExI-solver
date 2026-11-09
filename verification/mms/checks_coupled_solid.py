"""Checks that the analytic target satisfies the production structural
residual under the target force transferred from the analytic panel force.
This replaces the earlier tautological residual-identity check.
"""
import sys
from pathlib import Path

import numpy as np
import yaml
from dolfin import Constant, parameters, assemble

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "solid"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import ReissnerMindlinModel, SolidDomain
from coupled_target import build_default_target
from coupled_solid_source import ReferenceForceFunctional, _target_function


def main():
    parameters["form_compiler"]["cpp_optimize"] = True
    with (ROOT / "config/solid_params.yaml").open(encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    dt = 0.01
    domain = SolidDomain(cfg["span"], cfg["root_chord"], cfg["tip_chord"],
                         cfg["thickness_ratio"], cfg.get("leading_edge_sweep", 0.0),
                         8, 16, 8, "midpoint")
    model = ReissnerMindlinModel(
        domain, cfg["E"], cfg["nu"], cfg["rho_s"], cfg["eta_m"], cfg["eta_k"],
        cfg["kappa_shear"], Constant(dt), cfg.get("alpha_m", 0.1), cfg.get("alpha_f", 0.2),
        geometric_nonlinearity=cfg.get("geometric_nonlinearity", True),
        include_quadratic_inplane_strain=cfg.get("include_quadratic_inplane_strain", True))
    spaces = [model.V.sub(i).collapse() for i in range(3)]
    target = build_default_target(ROOT / "config", dt)
    model.q_old.assign(_target_function(model, target, 0.0, 0, spaces))
    model.v_old.assign(_target_function(model, target, 0.0, 1, spaces))
    model.a_old.assign(_target_function(model, target, 0.0, 2, spaces))
    model.q.assign(_target_function(model, target, dt, 0, spaces))

    force = ReferenceForceFunctional(model, target)
    force.set_panel_forces(target.panel_force(dt))
    rhs = assemble(model.residual)
    rhs *= -1.0
    rhs.axpy(1.0, force.vector(dt))
    for bc in model.bcs:
        bc.apply(rhs)
    norm = rhs.norm("l2")
    reference = force.vector(dt).norm("l2")
    if not np.isfinite(norm) or norm > 1.0e-10 * max(reference, 1.0):
        raise AssertionError(f"analytic target residual under target panel force is {norm:.6e}; reference={reference:.6e}")
    print("PASS checks_coupled_solid")


if __name__ == "__main__":
    main()
