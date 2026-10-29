"""Fixed-population coupled discrete-model MMS. It exercises the installed
FLOWVLM/rVPM/Reissner–Mindlin operators and the production socket chronology.
It is not a continuum Navier–Stokes MMS, does not verify native particle
lifecycle, and does not treat CRM/RBF moment conservation as verified.

G4 implements formulation sections "Weak form", "Generalized-α stage
values", and "Structural manufactured source coupled to the target force".
This is Objective B (end-to-end transfer and FSI accuracy) with the explicitly
documented reference-pressure approximation.
"""
import sys
from pathlib import Path

import numpy as np
import yaml
from dolfin import Constant, Function, assemble, parameters

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "solid"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import ReissnerMindlinModel, SolidDomain
from coupled_target import build_default_target
from coupled_solid_source import ReferenceForceFunctional, _target_function, add_coupled_mms_source


def main():
    parameters["form_compiler"]["cpp_optimize"] = True
    with open(ROOT / "config/solid_params.yaml", encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    dt = 0.005
    domain = SolidDomain(cfg["span"], cfg["root_chord"], cfg["tip_chord"],
                         cfg["thickness_ratio"], cfg.get("leading_edge_sweep", 0.0),
                         8, 16, 8, "midpoint")
    model = ReissnerMindlinModel(
        domain, cfg["E"], cfg["nu"], cfg["rho_s"], cfg["eta_m"], cfg["eta_k"],
        cfg["kappa_shear"], Constant(dt), cfg.get("alpha_m", 0.1), cfg.get("alpha_f", 0.2),
        geometric_nonlinearity=cfg.get("geometric_nonlinearity", True),
        include_quadratic_inplane_strain=cfg.get("include_quadratic_inplane_strain", True))
    spaces = [model.V.sub(i).collapse() for i in range(3)]
    target = build_default_target(ROOT / "config", 0.01)
    t = dt
    model.q_old.assign(_target_function(model, target, 0.0, 0, spaces))
    model.v_old.assign(_target_function(model, target, 0.0, 1, spaces))
    model.a_old.assign(_target_function(model, target, 0.0, 2, spaces))
    model.q.assign(_target_function(model, target, t, 0, spaces))
    force = ReferenceForceFunctional(model, target)
    source = add_coupled_mms_source(model, target, t, force, spaces)
    source_norm = source.norm("l2")
    residual = assemble(model.residual)
    rhs = residual.copy()
    rhs *= -1.0
    rhs.axpy(1.0, force.vector(t))
    rhs.axpy(1.0, source)
    for bc in model.bcs:
        bc.apply(rhs)
    defect = rhs.norm("l2")
    if not np.isfinite(source_norm) or source_norm <= 1e-12:
        raise AssertionError(f"manufactured structural source is not active: {source_norm}")
    if defect > 1e-8 * max(source_norm, 1.0):
        raise AssertionError(f"target structural residual/source identity failed: {defect=}, {source_norm=}")
    print("PASS G4 checks_coupled_solid")


if __name__ == "__main__":
    main()
