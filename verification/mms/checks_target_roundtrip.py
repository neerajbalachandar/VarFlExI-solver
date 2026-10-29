"""Fixed-population coupled discrete-model MMS. It exercises the installed
FLOWVLM/rVPM/Reissner–Mindlin operators and the production socket chronology.
It is not a continuum Navier–Stokes MMS, does not verify native particle
lifecycle, and does not treat CRM/RBF moment conservation as verified.

G1 implements formulation sections "Target structural field", "Target
aerodynamic geometry", and "Fixed-population target". The source goal is
Objective B for structure and fixed-population for particles.
"""
import json
from pathlib import Path

import numpy as np

from coupled_target import build_default_target, CoupledMMSTarget


ROOT = Path(__file__).resolve().parents[2]


def main():
    target = build_default_target(ROOT / "config", 0.01)
    manifest = target.manifest()
    restored = CoupledMMSTarget.from_manifest(json.loads(json.dumps(manifest)))
    assert restored.manifest() == manifest, "manifest did not round-trip exactly"
    for x in np.linspace(0.05*target.structural.Lx, 0.95*target.structural.Lx, 5):
        for y in np.linspace(0.05*target.structural.Ly, 0.95*target.structural.Ly, 5):
            for t in (0.0, 0.003, 0.01):
                spatial = target.structural.spatial(x, y)
                assert all(np.isfinite(v) for vals in spatial.values() for v in vals)
                for values in target.structural.q_and_derivatives(x, y, t):
                    assert np.isfinite(values).all()
                assert np.isfinite(target.geometry.point(0.7*target.structural.Lx, y, t)).all()
                state = target.particles.state(t)
                deriv = target.particles.derivatives(t)
                assert all(np.isfinite(v).all() for v in (*state, *deriv))
                assert np.all(state[2] > 0.0)
    print("PASS G1 checks_target_roundtrip")


if __name__ == "__main__":
    main()
