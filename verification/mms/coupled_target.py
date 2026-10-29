"""Fixed-population coupled discrete-model MMS. It exercises the installed
FLOWVLM/rVPM/Reissner–Mindlin operators and the production socket chronology.
It is not a continuum Navier–Stokes MMS, does not verify native particle
lifecycle, and does not treat CRM/RBF moment conservation as verified.

Implements formulation sections "Target structural field", "Target
aerodynamic geometry", and "Fixed-population target" under "Particle target
and rVPM manufactured source". This is Objective B (end-to-end transfer and
FSI accuracy) for the structural source and fixed-population scope for the
particle source. The module contains analytic target definitions only; it does
not import solver-specific packages or replace numerical solver states.
"""

from dataclasses import dataclass
from pathlib import Path
import json
import math

import numpy as np
import yaml


SCOPE = (
    "Fixed-population coupled discrete-model MMS. It exercises the installed "
    "FLOWVLM/rVPM/Reissner–Mindlin operators and the production socket chronology. "
    "It is not a continuum Navier–Stokes MMS, does not verify native particle "
    "lifecycle, and does not treat CRM/RBF moment conservation as verified."
)


def envelope(x, y, Lx, Ly):
    """Return B and its exact first/second spatial derivatives.

    Formulation section: "Target structural field".
    """
    kx, ky = math.pi / float(Lx), math.pi / float(Ly)
    sx, cx = math.sin(kx * x), math.cos(kx * x)
    sy, cy = math.sin(ky * y), math.cos(ky * y)
    fx, fy = sx**4, sy**4
    fx1, fy1 = 4 * kx * sx**3 * cx, 4 * ky * sy**3 * cy
    fx2 = 4 * kx**2 * (3 * sx**2 * cx**2 - sx**4)
    fy2 = 4 * ky**2 * (3 * sy**2 * cy**2 - sy**4)
    return {
        "B": fx * fy,
        "B_x": fx1 * fy,
        "B_y": fx * fy1,
        "B_xx": fx2 * fy,
        "B_xy": fx1 * fy1,
        "B_yy": fx * fy2,
    }


@dataclass(frozen=True)
class StructuralTarget:
    Lx: float
    Ly: float
    components: dict

    @staticmethod
    def temporal(amplitude, omega, mode, t, order=0):
        """Exact g, g', or g'' for cos, sin, or 1-cos temporal modes."""
        phase = float(omega) * float(t)
        mode = str(mode)
        if mode == "cos":
            return (float(amplitude) * math.cos(phase) if order == 0 else
                    -float(amplitude) * omega * math.sin(phase) if order == 1 else
                    -float(amplitude) * omega**2 * math.cos(phase) if order == 2 else
                    (_ for _ in ()).throw(ValueError("time derivative order must be 0, 1, or 2")))
        if mode == "sin":
            return (float(amplitude) * math.sin(phase) if order == 0 else
                    float(amplitude) * omega * math.cos(phase) if order == 1 else
                    -float(amplitude) * omega**2 * math.sin(phase) if order == 2 else
                    (_ for _ in ()).throw(ValueError("time derivative order must be 0, 1, or 2")))
        if mode == "ramp":
            return (float(amplitude) * (1.0 - math.cos(phase)) if order == 0 else
                    float(amplitude) * omega * math.sin(phase) if order == 1 else
                    float(amplitude) * omega**2 * math.cos(phase) if order == 2 else
                    (_ for _ in ()).throw(ValueError("time derivative order must be 0, 1, or 2")))
        raise ValueError(f"unsupported temporal mode {mode!r}")

    def spatial(self, x, y):
        b = envelope(float(x), float(y), self.Lx, self.Ly)
        out = {}
        for name, spec in self.components.items():
            out[name] = tuple(float(spec["amplitude"]) * b[key] for key in
                              ("B", "B_x", "B_y", "B_xx", "B_xy", "B_yy"))
        return out

    def q_and_derivatives(self, x, y, t):
        b = envelope(float(x), float(y), self.Lx, self.Ly)["B"]
        q, qt, qtt = [], [], []
        for name in ("u", "v", "w", "theta_x", "theta_y"):
            spec = self.components[name]
            q.append(b * self.temporal(**spec, t=t, order=0))
            qt.append(b * self.temporal(**spec, t=t, order=1))
            qtt.append(b * self.temporal(**spec, t=t, order=2))
        return np.asarray(q), np.asarray(qt), np.asarray(qtt)

    def manifest(self):
        return {"Lx": float(self.Lx), "Ly": float(self.Ly), "components": self.components}


@dataclass(frozen=True)
class GeometryTarget:
    structural: StructuralTarget
    sweep: float
    x_bound: float = None

    def point(self, x, y, t):
        """Map to the target surface using the linearized production interface.

        Formulation section: "Target aerodynamic geometry". The lifting
        displacement is (u, v, w + theta_x*(x-x_bound)); the repository's
        production transfer sends spanwise displacement and adds edge rotation
        through omega cross r, whose requested reduced-map convention is this
        chordwise linearization about the bound-vortex station.
        """
        q, _, _ = self.structural.q_and_derivatives(x, y, t)
        xb = 0.25 * self.structural.Lx if self.x_bound is None else self.x_bound
        return np.asarray((x + q[0], y + q[1],
                           q[2] + q[3] * (x - xb)), dtype=float)


@dataclass(frozen=True)
class ParticleTarget:
    ids: np.ndarray
    x0: np.ndarray
    gamma0: np.ndarray
    sigma0: np.ndarray
    a: np.ndarray
    b: np.ndarray
    c: np.ndarray
    d: np.ndarray
    e: np.ndarray
    omega_x: np.ndarray
    omega_gamma: np.ndarray
    omega_sigma: np.ndarray

    def state(self, t):
        tx = self.omega_x[:, None] * float(t)
        tg = self.omega_gamma[:, None] * float(t)
        ts = self.omega_sigma * float(t)
        X = self.x0 + self.a * np.sin(tx) + self.b * (1.0 - np.cos(tx))
        Gamma = self.gamma0 + self.c * np.sin(tg) + self.d * (1.0 - np.cos(tg))
        sigma = self.sigma0 + self.e * (1.0 - np.cos(ts))
        return X, Gamma, sigma

    def derivatives(self, t):
        tx = self.omega_x[:, None] * float(t)
        tg = self.omega_gamma[:, None] * float(t)
        ts = self.omega_sigma * float(t)
        X_t = self.omega_x[:, None] * (self.a * np.cos(tx) + self.b * np.sin(tx))
        Gamma_t = self.omega_gamma[:, None] * (self.c * np.cos(tg) + self.d * np.sin(tg))
        sigma_t = self.e * self.omega_sigma * np.sin(ts)
        return X_t, Gamma_t, sigma_t

    def manifest(self):
        return {name: np.asarray(getattr(self, name)).tolist() for name in (
            "ids",
            "x0", "gamma0", "sigma0", "a", "b", "c", "d", "e",
            "omega_x", "omega_gamma", "omega_sigma")}

    @classmethod
    def from_manifest(cls, data):
        arrays = {k: np.asarray(data[k], dtype=int if k == "ids" else float) for k in (
            "ids",
            "x0", "gamma0", "sigma0", "a", "b", "c", "d", "e",
            "omega_x", "omega_gamma", "omega_sigma")}
        n = arrays["x0"].shape[0]
        if arrays["x0"].shape != (n, 3) or arrays["gamma0"].shape != (n, 3):
            raise ValueError("particle x0 and gamma0 must have shape (N, 3)")
        for k in ("a", "b", "c", "d"):
            if arrays[k].shape != (n, 3):
                raise ValueError(f"particle {k} must have shape (N, 3)")
        for k in ("sigma0", "e", "omega_x", "omega_gamma", "omega_sigma"):
            if arrays[k].shape != (n,):
                raise ValueError(f"particle {k} must have shape (N,)")
        if np.any(arrays["sigma0"] <= 0) or np.any(arrays["e"] < 0):
            raise ValueError("sigma0 must be positive and sigma modulation nonnegative")
        if arrays["ids"].shape != (n,) or len(set(arrays["ids"].tolist())) != n:
            raise ValueError("particle ids must be unique immutable identifiers, one per particle")
        return cls(**arrays)


@dataclass(frozen=True)
class CoupledMMSTarget:
    structural: StructuralTarget
    geometry: GeometryTarget
    particles: ParticleTarget
    n_span: int
    y_stations: np.ndarray
    chord: float
    freestream: dict
    T: float

    def manifest(self):
        """Serialize all analytic data consumed by both solver processes."""
        initial_q, initial_qt, initial_qtt = self.structural.q_and_derivatives(
            0.5 * self.structural.Lx, 0.5 * self.structural.Ly, 0.0
        )
        return {
            "scope": SCOPE,
            "Lx": float(self.structural.Lx), "Ly": float(self.structural.Ly),
            "n_span": int(self.n_span), "y_stations": self.y_stations.tolist(),
            "chord": float(self.chord), "sweep": float(self.geometry.sweep),
            "x_bound": float(self.geometry.x_bound if self.geometry.x_bound is not None else 0.25*self.structural.Lx),
            "freestream": self.freestream, "T": float(self.T),
            "structural": {
                **self.structural.manifest(),
                "initial_values_at_center": {
                    "q": initial_q.tolist(), "q_t": initial_qt.tolist(), "q_tt": initial_qtt.tolist(),
                },
            },
            "particles": self.particles.manifest(),
            "particle_formula": {
                "position": "x0 + a*sin(omega_x*t) + b*(1-cos(omega_x*t))",
                "circulation": "gamma0 + c*sin(omega_gamma*t) + d*(1-cos(omega_gamma*t))",
                "core": "sigma0 + e*(1-cos(omega_sigma*t))",
            },
        }

    @classmethod
    def from_manifest(cls, data):
        structural = StructuralTarget(float(data["Lx"]), float(data["Ly"]), data["structural"]["components"])
        geometry = GeometryTarget(structural, float(data["sweep"]), float(data["x_bound"]))
        return cls(structural, geometry, ParticleTarget.from_manifest(data["particles"]),
                   int(data["n_span"]), np.asarray(data["y_stations"], dtype=float),
                   float(data["chord"]), data["freestream"], float(data["T"]))


def _read_yaml(path):
    with open(path, encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def build_default_target(config, T, n_span_override=None):
    """Build one deterministic dynamic target from a config directory or map."""
    if isinstance(config, (str, Path)):
        config = Path(config)
        solid = _read_yaml(config / "solid_params.yaml")
        fluid = _read_yaml(config / "fluid_params.yaml")
        coupling = _read_yaml(config / "coupling_params.yaml")
    else:
        solid, fluid, coupling = config["solid"], config["fluid"], config["coupling"]
    Lx, Ly = float(solid["root_chord"]), float(solid["span"])
    omega = 2.3
    components = {
        "u": {"amplitude": 1.0e-4, "omega": omega, "mode": "ramp"},
        "v": {"amplitude": 1.0e-4, "omega": omega, "mode": "ramp"},
        "w": {"amplitude": 1.0e-3, "omega": omega, "mode": "ramp"},
        "theta_x": {"amplitude": 1.0e-3, "omega": omega, "mode": "ramp"},
        "theta_y": {"amplitude": -1.0e-3, "omega": omega, "mode": "ramp"},
    }
    structural = StructuralTarget(Lx, Ly, components)
    n_span = int(n_span_override or coupling.get("comm_nspan", fluid.get("n_span", 80)))
    eta = (np.arange(n_span) + 0.5) / n_span
    y_stations = eta * Ly
    # A small, deterministic, fixed population remains in the downstream wake.
    n = 4
    x0 = np.column_stack((np.full(n, 1.5 * Lx), np.linspace(0.2*Ly, 0.8*Ly, n), np.full(n, 0.1*Lx)))
    gamma0 = np.tile(np.asarray((0.0, 0.0, 1.0e-4)), (n, 1))
    particle = ParticleTarget(
        ids=np.arange(1, n+1, dtype=int),
        x0=x0, gamma0=gamma0, sigma0=np.full(n, max(0.5*float(fluid.get("sigma_surface", 0.025))*Ly, 1e-5)),
        a=np.column_stack((np.full(n, 0.01*Lx), np.zeros(n), np.full(n, 0.005*Lx))),
        b=np.column_stack((np.zeros(n), np.full(n, 0.004*Ly), np.zeros(n))),
        c=np.zeros((n, 3)), d=np.tile(np.asarray((0.0, 0.0, 1.0e-5)), (n, 1)),
        e=np.full(n, 1.0e-5), omega_x=np.full(n, 2.0), omega_gamma=np.full(n, 1.7),
        omega_sigma=np.full(n, 1.3),
    )
    freestream = {
        "vinf": float(fluid.get("vinf", 15.0)),
        "aoa_deg": float(fluid.get("aoa_deg", 0.0)),
        "rho": float(fluid.get("rho", 1.225)),
        "nu": float(fluid.get("nu", 1.5e-5)),
        "oscillation": fluid.get("freestream_oscillation", [0.0, 0.0, 0.0]),
        "frequency_hz": float(fluid.get("freestream_frequency_hz", 0.0)),
    }
    return CoupledMMSTarget(
        structural, GeometryTarget(structural, float(solid.get("leading_edge_sweep", 0.0))),
        particle, n_span, y_stations, Lx, freestream, float(T)
    )


def write_manifest(target, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(target.manifest(), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path
