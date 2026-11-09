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

    def velocity(self, x, y, t):
        """Exact time derivative of :meth:`point` at a fixed material point."""
        _, q_t, _ = self.structural.q_and_derivatives(x, y, t)
        xb = 0.25 * self.structural.Lx if self.x_bound is None else self.x_bound
        return np.asarray((q_t[0], q_t[1], q_t[2] + q_t[3] * (x - xb)), dtype=float)


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
class BoundCirculationTarget:
    """Analytic spanwise bound-circulation magnitude at VLM stations."""
    gamma0: np.ndarray
    a: np.ndarray
    b: np.ndarray
    omega: float

    def state(self, t):
        gamma0 = np.asarray(self.gamma0, dtype=float)
        a = np.asarray(self.a, dtype=float)
        b = np.asarray(self.b, dtype=float)
        if gamma0.ndim != 1 or a.shape != gamma0.shape or b.shape != gamma0.shape:
            raise ValueError("bound circulation gamma0, a, and b must be matching 1D arrays")
        if not np.isfinite(gamma0).all() or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("bound circulation coefficients must be finite")
        if not math.isfinite(float(self.omega)) or not math.isfinite(float(t)):
            raise ValueError("bound circulation frequency and time must be finite")
        phase = float(self.omega) * float(t)
        result = gamma0 + a * math.sin(phase) + b * (1.0 - math.cos(phase))
        if not np.isfinite(result).all():
            raise ValueError("bound circulation state is non-finite")
        return result

    def manifest(self):
        return {"gamma0": np.asarray(self.gamma0).tolist(),
                "a": np.asarray(self.a).tolist(), "b": np.asarray(self.b).tolist(),
                "omega": float(self.omega)}

    @classmethod
    def from_manifest(cls, data):
        values = {key: np.asarray(data[key], dtype=float) for key in ("gamma0", "a", "b")}
        if any(value.ndim != 1 for value in values.values()):
            raise ValueError("bound circulation manifest coefficients must be 1D")
        if len({value.shape for value in values.values()}) != 1:
            raise ValueError("bound circulation manifest coefficients must have equal lengths")
        if not all(np.isfinite(value).all() for value in values.values()):
            raise ValueError("bound circulation manifest coefficients must be finite")
        omega = float(data["omega"])
        if not math.isfinite(omega):
            raise ValueError("bound circulation manifest frequency must be finite")
        return cls(**values, omega=omega)


@dataclass(frozen=True)
class CoupledMMSTarget:
    structural: StructuralTarget
    geometry: GeometryTarget
    particles: ParticleTarget
    bound_circulation: BoundCirculationTarget
    n_span: int
    y_stations: np.ndarray
    chord: float
    tip_chord: float
    freestream: dict
    T: float

    def panel_force(self, t):
        """Return a span-panel Kutta–Joukowski target force in global xyz.

        Circulation is oriented along +y. The panel's chordwise length is the
        scalar spanwise-varying chord; the directed circulation line is +y, so
        +x flow produces positive z lift. Surface velocity is subtracted from
        local freestream at the target bound-vortex stations.
        """
        if (int(self.n_span) < 1 or self.y_stations.shape != (int(self.n_span),)
                or len(self.bound_circulation.gamma0) != int(self.n_span)):
            raise ValueError("bound-circulation station count does not match n_span")
        eta = (np.arange(self.n_span, dtype=float) + 0.5) / self.n_span
        gamma = self.bound_circulation.state(t)
        rho = float(self.freestream["rho"])
        speed = float(self.freestream["vinf"])
        aoa = math.radians(float(self.freestream.get("aoa_deg", 0.0)))
        gradient = np.asarray(self.freestream.get("gradient", np.zeros(9)), dtype=float).reshape(3, 3).T
        amplitudes = np.asarray(self.freestream.get("oscillation", np.zeros(3)), dtype=float)
        wave = np.asarray(self.freestream.get("wavenumber", np.zeros(3)), dtype=float)
        reference = np.asarray(self.freestream.get("reference", np.zeros(3)), dtype=float)
        frequency = float(self.freestream.get("frequency_hz", 0.0))
        phase0 = math.radians(float(self.freestream.get("phase_deg", 0.0)))
        if amplitudes.shape != (3,) or wave.shape != (3,) or reference.shape != (3,):
            raise ValueError("freestream oscillation, wavenumber, and reference must be 3-vectors")
        if not all(np.isfinite(x).all() if isinstance(x, np.ndarray) else math.isfinite(float(x))
                   for x in (gradient, amplitudes, wave, reference, rho, speed, frequency, phase0)):
            raise ValueError("target freestream contains non-finite values")

        force = np.zeros((self.n_span, 3), dtype=float)
        dt = float(t)
        xb = float(self.geometry.x_bound if self.geometry.x_bound is not None else 0.25*self.structural.Lx)
        for i, eta_i in enumerate(eta):
            y = float(self.y_stations[i])
            x = xb
            surface_velocity = self.geometry.velocity(x, y, dt)
            point = self.geometry.point(x, y, dt)
            xp = point - reference
            base = speed * np.asarray((math.cos(aoa), 0.0, math.sin(aoa)))
            gust_phase = 2.0 * math.pi * frequency * dt + phase0
            gust = amplitudes * np.sin(gust_phase + np.asarray((wave[1]*xp[1], wave[2]*xp[2], wave[0]*xp[0])))
            local_velocity = base + gradient @ xp + gust - surface_velocity
            chord_i = self.tip_chord * (1.0 - eta_i) + self.chord * eta_i
            # The structural/fluid interface uses y as the span direction.
            # Thus the directed circulation line is +y and +x flow gives +z lift.
            circulation_line = np.asarray((0.0, chord_i, 0.0))
            force[i] = rho * gamma[i] * np.cross(local_velocity, circulation_line)
        if force.shape != (self.n_span, 3) or not np.isfinite(force).all():
            raise ValueError("target panel force is non-finite or has an invalid shape")
        return force

    def manifest(self):
        """Serialize all analytic data consumed by both solver processes."""
        initial_q, initial_qt, initial_qtt = self.structural.q_and_derivatives(
            0.5 * self.structural.Lx, 0.5 * self.structural.Ly, 0.0
        )
        return {
            "scope": SCOPE,
            "Lx": float(self.structural.Lx), "Ly": float(self.structural.Ly),
            "n_span": int(self.n_span), "y_stations": self.y_stations.tolist(),
            "chord": float(self.chord), "tip_chord": float(self.tip_chord),
            "sweep": float(self.geometry.sweep),
            "x_bound": float(self.geometry.x_bound if self.geometry.x_bound is not None else 0.25*self.structural.Lx),
            "freestream": self.freestream, "T": float(self.T),
            "structural": {
                **self.structural.manifest(),
                "initial_values_at_center": {
                    "q": initial_q.tolist(), "q_t": initial_qt.tolist(), "q_tt": initial_qtt.tolist(),
                },
            },
            "particles": self.particles.manifest(),
            "bound_circulation": self.bound_circulation.manifest(),
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
        n_span = int(data["n_span"])
        y_stations = np.asarray(data["y_stations"], dtype=float)
        if n_span < 1 or y_stations.shape != (n_span,) or not np.isfinite(y_stations).all():
            raise ValueError("manifest y_stations must be a finite vector with n_span entries")
        bound_circulation = BoundCirculationTarget.from_manifest(data["bound_circulation"])
        if bound_circulation.gamma0.shape != (n_span,):
            raise ValueError("bound-circulation manifest length must equal n_span")
        return cls(structural, geometry, ParticleTarget.from_manifest(data["particles"]),
                   bound_circulation, n_span, y_stations,
                   float(data["chord"]), float(data.get("tip_chord", data["chord"])),
                   data["freestream"], float(data["T"]))


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
        "gradient": fluid.get("freestream_gradient", [0.0] * 9),
        "wavenumber": fluid.get("freestream_wavenumber", [0.0] * 3),
        "phase_deg": float(fluid.get("freestream_phase_deg", 0.0)),
        "reference": fluid.get("freestream_reference", [0.0] * 3),
    }
    bound_circulation = BoundCirculationTarget(
        gamma0=np.full(n_span, 1.0e-4), a=np.full(n_span, 1.0e-5),
        b=np.full(n_span, 5.0e-6), omega=omega,
    )
    return CoupledMMSTarget(
        structural, GeometryTarget(structural, float(solid.get("leading_edge_sweep", 0.0))),
        particle, bound_circulation, n_span, y_stations, Lx,
        float(solid.get("tip_chord", Lx)), freestream, float(T)
    )


def write_manifest(target, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(target.manifest(), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path
