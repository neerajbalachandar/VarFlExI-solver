"""Analytical continuum velocity-vorticity MMS and field-error utilities.

This module deliberately does not equate a continuum source with a particle
circulation source. It derives the continuum forcing independently and offers
common grid metrics; a production particle source projection remains a
separate requirement (see README.md and IMPLEMENTATION_REPORT.md).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import sympy as sp


@dataclass
class FieldTarget:
    """Symbolically derived, divergence-free analytical target fields."""

    symbols: tuple
    velocity_expr: tuple
    vorticity_expr: tuple
    source_expr: tuple
    velocity_fn: Any
    vorticity_fn: Any
    source_fn: Any
    divergence_expr: Any

    @classmethod
    def build(cls, *, length: float = 1.0, amplitude: float = 1.0,
              frequency: float = 2.0 * math.pi, phase: float = 0.37):
        if length <= 0 or amplitude <= 0 or frequency <= 0:
            raise ValueError("length, amplitude, and frequency must be positive")
        x, y, z, t = sp.symbols("x y z t", real=True)
        k = 2 * sp.pi / sp.Float(length)
        a = sp.Float(amplitude)
        w = sp.Float(frequency)
        phi = sp.Float(phase)
        # A smooth, time-dependent vector potential. Distinct components and
        # phase shifts excite all velocity/vorticity components.
        potential = (
            a * sp.sin(k*y) * sp.sin(k*z) * sp.cos(w*t),
            a * sp.sin(k*z) * sp.sin(k*x) * sp.sin(w*t + phi),
            a * sp.sin(k*x) * sp.sin(k*y) * sp.cos(w*t + phi),
        )
        curl = lambda q: (
            sp.diff(q[2], y) - sp.diff(q[1], z),
            sp.diff(q[0], z) - sp.diff(q[2], x),
            sp.diff(q[1], x) - sp.diff(q[0], y),
        )
        u = curl(potential)
        omega = curl(u)
        div_u = sp.simplify(sum(sp.diff(u[i], (x, y, z)[i]) for i in range(3)))
        grad_omega = [[sp.diff(omega[i], (x, y, z)[j]) for j in range(3)] for i in range(3)]
        grad_u = [[sp.diff(u[i], (x, y, z)[j]) for j in range(3)] for i in range(3)]
        source = tuple(sp.simplify(
            sp.diff(omega[i], t)
            + sum(u[j] * grad_omega[i][j] for j in range(3))
            - sum(omega[j] * grad_u[i][j] for j in range(3))
        ) for i in range(3))
        args = (x, y, z, t)
        return cls(
            args, tuple(map(sp.simplify, u)), tuple(map(sp.simplify, omega)),
            source,
            sp.lambdify(args, u, modules="numpy"),
            sp.lambdify(args, omega, modules="numpy"),
            sp.lambdify(args, source, modules="numpy"), div_u,
        )

    def evaluate(self, field: str, points: np.ndarray, time: float) -> np.ndarray:
        """Evaluate a vector field on (...,3) points; return (...,3)."""
        points = np.asarray(points, dtype=float)
        if points.shape[-1] != 3:
            raise ValueError("points must have final dimension 3")
        fn = {"u": self.velocity_fn, "omega": self.vorticity_fn,
              "source": self.source_fn}.get(field)
        if fn is None:
            raise ValueError(f"unknown field {field!r}")
        vals = np.asarray(fn(points[..., 0], points[..., 1], points[..., 2], time), dtype=float)
        return np.moveaxis(vals, 0, -1)


def cartesian_grid(bounds: list[list[float]], cells: list[int]):
    """Return node coordinates and tensor-product trapezoid weights."""
    box = np.asarray(bounds, dtype=float)
    n = np.asarray(cells, dtype=int)
    if box.shape != (3, 2) or np.any(box[:, 1] <= box[:, 0]):
        raise ValueError("bounds must be [[xmin,xmax],[ymin,ymax],[zmin,zmax]]")
    if n.shape != (3,) or np.any(n < 1):
        raise ValueError("cells must contain three positive cell counts")
    axes = [np.linspace(box[d, 0], box[d, 1], n[d] + 1) for d in range(3)]
    mesh = np.meshgrid(*axes, indexing="ij")
    points = np.stack(mesh, axis=-1)
    spacing = (box[:, 1] - box[:, 0]) / n
    one_d_weights = []
    for axis in axes:
        weights = np.ones(axis.size)
        weights[[0, -1]] = 0.5
        one_d_weights.append(weights)
    wmesh = np.meshgrid(*one_d_weights, indexing="ij")
    weights = np.prod(np.stack(wmesh, axis=0), axis=0) * float(np.prod(spacing))
    return points, weights, spacing


def field_error_metrics(numerical: np.ndarray, target: np.ndarray,
                        weights: np.ndarray, *, prefix: str,
                        norm_floor: float = 1.0e-13) -> dict[str, Any]:
    """Absolute/relative weighted norms, components, max error and norms."""
    numerical = np.asarray(numerical, dtype=float)
    target = np.asarray(target, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if numerical.shape != target.shape or numerical.shape[:-1] != weights.shape or numerical.shape[-1] != 3:
        raise ValueError("field arrays must be matching (...,3) with weights matching (...)")
    if not (np.all(np.isfinite(numerical)) and np.all(np.isfinite(target)) and np.all(np.isfinite(weights))):
        raise ValueError("field or quadrature weights contain non-finite values")
    if np.any(weights <= 0):
        raise ValueError("quadrature weights must be positive")
    err = numerical - target
    target_norm = math.sqrt(float(np.sum(weights * np.sum(target * target, axis=-1))))
    numeric_norm = math.sqrt(float(np.sum(weights * np.sum(numerical * numerical, axis=-1))))
    abs_l2 = math.sqrt(float(np.sum(weights * np.sum(err * err, axis=-1))))
    if target_norm <= norm_floor:
        raise ValueError(f"{prefix} target L2 norm {target_norm:.4e} is too small for relative normalization")
    components = {}
    for i, name in enumerate("xyz"):
        target_c = math.sqrt(float(np.sum(weights * target[..., i] ** 2)))
        error_c = math.sqrt(float(np.sum(weights * err[..., i] ** 2)))
        components[name] = {
            "target_l2": target_c,
            "absolute_l2": error_c,
            "relative_l2": error_c / target_c if target_c > norm_floor else None,
            "max_absolute": float(np.max(np.abs(err[..., i]))),
        }
    return {
        "absolute_l2": abs_l2,
        "relative_l2": abs_l2 / target_norm,
        "target_l2": target_norm,
        "numerical_l2": numeric_norm,
        "max_absolute": float(np.max(np.abs(err))),
        "component_errors": components,
        "grid_points": int(weights.size),
    }


def mesh_spacing(lengths: list[float], nx: int, ny: int) -> float:
    if min(lengths) <= 0 or nx < 1 or ny < 1:
        raise ValueError("geometry lengths and element counts must be positive")
    return max(lengths[0] / nx, lengths[1] / ny)


def observed_orders(errors: list[float], scales: list[float]) -> list[float | None]:
    if len(errors) != len(scales):
        raise ValueError("errors and refinement scales must have equal lengths")
    result: list[float | None] = [None]
    for i in range(1, len(errors)):
        if errors[i-1] <= 0 or errors[i] <= 0 or scales[i-1] <= scales[i] or scales[i] <= 0:
            result.append(None)
        elif errors[i] >= errors[i-1]:
            result.append(None)
        else:
            result.append(math.log(errors[i-1] / errors[i]) / math.log(scales[i-1] / scales[i]))
    return result


def analytical_check(config_path: Path, output_path: Path | None = None,
                     geometry_lengths: tuple[float, float] | None = None) -> dict:
    import yaml
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    target_cfg = config["target"]
    target = FieldTarget.build(**target_cfg)
    if sp.simplify(target.divergence_expr) != 0:
        raise RuntimeError(f"target velocity is not divergence-free: {target.divergence_expr}")
    bounds = config["reconstruction"]["bounds"]
    cells = config["reconstruction"]["preflight_cells"]
    points, weights, spacing = cartesian_grid(bounds, cells)
    t_end = float(config["temporal_study"]["final_time"])
    fields = {name: target.evaluate(name, points, t_end) for name in ("u", "omega", "source")}
    norms = {name: math.sqrt(float(np.sum(weights * np.sum(value**2, axis=-1))))
             for name, value in fields.items()}
    if min(norms["u"], norms["omega"]) <= 1.0e-13:
        raise RuntimeError("target velocity/vorticity norm is zero or too small")
    if geometry_lengths is None:
        production_solid = config_path.resolve().parents[2] / "config" / "solid_params.yaml"
        if production_solid.exists():
            solid_cfg = yaml.safe_load(production_solid.read_text(encoding="utf-8"))
            geometry_lengths = (float(solid_cfg["root_chord"]), float(solid_cfg["span"]))
        else:
            raise FileNotFoundError("Pass production geometry lengths or provide config/solid_params.yaml")
    mesh = config["spatial_study"]
    levels = []
    for case in mesh["cases"]:
        h = mesh_spacing(list(geometry_lengths), int(case["nx"]), int(case["ny"]))
        levels.append({**case, "h_mesh": h})
    result = {
        "status": "PASS",
        "claim": "analytical/configuration checks only; no particle solver run",
        "divergence_u_symbolic": str(target.divergence_expr),
        "target_norms_at_final_time": norms,
        "reconstruction": {"bounds": bounds,
                           "preflight_grid_cells": cells,
                           "preflight_grid_points": int(weights.size),
                           "preflight_spacing": spacing.tolist(),
                           "grid_resolution_rule": config["reconstruction"]["grid_resolution"],
                           "core_intervals_minimum": config["particles"]["core_grid_intervals_minimum"],
                           "target_intervals_per_wavelength": config["reconstruction"]["target_intervals_per_wavelength"],
                           "sensitivity_level": config["reconstruction"]["sensitivity_level"],
                           "sensitivity_factor": config["reconstruction"]["sensitivity_factor"]},
        "spatial_cases_with_h_mesh": levels,
        "structural_characteristic_lengths_m": {
            "Lx_root_chord": float(geometry_lengths[0]),
            "Ly_span": float(geometry_lengths[1]),
        },
        "temporal_dts_s": config["temporal_study"]["dt_s"],
        "temporal_final_time_s": t_end,
        "expected_viscous_scheme": config["physics"]["viscous_scheme"],
        "expected_vpm_sfs": config["physics"]["vpm_sfs"],
    }
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def check_reconstruction_resolution(spacing: np.ndarray, sigma_min: float,
                                    target_wavelength: float, *,
                                    core_intervals: int = 3,
                                    target_intervals: int = 8) -> dict:
    """Fail if a Cartesian grid under-resolves either core or target scale."""
    h = np.asarray(spacing, dtype=float)
    if h.shape != (3,) or np.any(h <= 0) or sigma_min <= 0 or target_wavelength <= 0:
        raise ValueError("spacing, minimum core, and target wavelength must be positive")
    core_limit = sigma_min / core_intervals
    target_limit = target_wavelength / target_intervals
    allowed = min(core_limit, target_limit)
    if np.max(h) > allowed * (1 + 1.0e-12):
        raise ValueError(
            f"Eulerian grid spacing {h.tolist()} exceeds resolution limit {allowed:.6e} m "
            f"(sigma_min={sigma_min:.6e} m, wavelength={target_wavelength:.6e} m)")
    return {"spacing_m": h.tolist(), "sigma_min_m": float(sigma_min),
            "target_wavelength_m": float(target_wavelength),
            "core_intervals": int(core_intervals), "target_intervals": int(target_intervals),
            "maximum_allowed_spacing_m": allowed}


def derive_reconstruction_cells(bounds: list[list[float]], sigma_min: float,
                                target_wavelength: float, *,
                                core_intervals: int = 3,
                                target_intervals: int = 8) -> tuple[list[int], float]:
    """Plan cell counts from measured core scale and target wavelength."""
    box = np.asarray(bounds, dtype=float)
    if box.shape != (3, 2) or np.any(box[:, 1] <= box[:, 0]):
        raise ValueError("bounds must be three increasing coordinate intervals")
    if sigma_min <= 0 or target_wavelength <= 0:
        raise ValueError("smallest core and target wavelength must be positive")
    dx = min(sigma_min / core_intervals, target_wavelength / target_intervals)
    cells = np.ceil((box[:, 1] - box[:, 0]) / dx).astype(int).tolist()
    return cells, dx


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("field_mms_parameters.yaml"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    report = analytical_check(args.config, args.output)
    print(json.dumps(report, indent=2))
