"""Fixed-population coupled discrete-model MMS. It exercises the installed
FLOWVLM/rVPM/Reissner–Mindlin operators and the production socket chronology.
It is not a continuum Navier–Stokes MMS, does not verify native particle
lifecycle, and does not treat CRM/RBF moment conservation as verified.

G2 implements formulation sections "Target structural field" and
"Fixed-population target" with analytic derivative comparisons. It is
Objective B for structure and fixed-population for particles.
"""
from pathlib import Path

import numpy as np

from coupled_target import build_default_target, envelope


ROOT = Path(__file__).resolve().parents[2]


def relerr(a, b):
    return abs(float(a)-float(b)) / max(abs(float(a)), abs(float(b)), 1e-14)


def main():
    target = build_default_target(ROOT / "config", 0.04)
    x, y, t = 0.43*target.structural.Lx, 0.57*target.structural.Ly, 0.013
    h_x, h_y = 2e-4*target.structural.Lx, 2e-4*target.structural.Ly
    h_t = 2e-4*target.T
    b = envelope(x, y, target.structural.Lx, target.structural.Ly)
    assert relerr(b["B_x"], (envelope(x+h_x,y,target.structural.Lx,target.structural.Ly)["B"]-
                             envelope(x-h_x,y,target.structural.Lx,target.structural.Ly)["B"])/(2*h_x)) < 1e-6
    assert relerr(b["B_y"], (envelope(x,y+h_y,target.structural.Lx,target.structural.Ly)["B"]-
                             envelope(x,y-h_y,target.structural.Lx,target.structural.Ly)["B"])/(2*h_y)) < 1e-6
    q, qt, _ = target.structural.q_and_derivatives(x, y, t)
    q_plus = target.structural.q_and_derivatives(x,y,t+h_t)[0]
    q_minus = target.structural.q_and_derivatives(x,y,t-h_t)[0]
    assert max(relerr(a,b) for a,b in zip(qt,(q_plus-q_minus)/(2*h_t))) < 1e-6
    Xp, Gp, Sp = target.particles.state(t+h_t)
    Xm, Gm, Sm = target.particles.state(t-h_t)
    Xd, Gd, Sd = target.particles.derivatives(t)
    for exact, fd in ((Xd,(Xp-Xm)/(2*h_t)), (Gd,(Gp-Gm)/(2*h_t)), (Sd,(Sp-Sm)/(2*h_t))):
        assert np.max(np.abs(exact-fd)/np.maximum(np.maximum(np.abs(exact),np.abs(fd)),1e-14)) < 1e-6
    assert np.isfinite(q).all()
    print("PASS G2 checks_target_derivatives")


if __name__ == "__main__":
    main()
