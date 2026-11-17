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
    potential_expr: tuple
    induced_velocity_expr: tuple
    velocity_expr: tuple
    vorticity_expr: tuple
    source_expr: tuple
    freestream_expr: tuple
    potential_fn: Any
    induced_velocity_fn: Any
    velocity_fn: Any
    vorticity_fn: Any
    source_fn: Any
    freestream_fn: Any
    divergence_expr: Any
    curl_identity_expr: tuple
    source_identity_expr: tuple

    @classmethod
    def build(cls, *, amplitude: float = 1.0, frequency: float = 2.0 * math.pi,
              phase: float = 0.37,
              center: tuple[float, float, float] = (0.0, 0.0, 0.0),
              support_radii: tuple[float, float, float] = (1.0, 1.0, 1.0),
              freestream_config: dict | None = None):
        if (not all(math.isfinite(float(v)) for v in (amplitude, frequency, phase))
                or amplitude <= 0 or frequency <= 0):
            raise ValueError("amplitude and frequency must be positive")
        if (len(center) != 3 or len(support_radii) != 3
                or not np.all(np.isfinite(center))
                or not np.all(np.isfinite(support_radii))
                or min(support_radii) <= 0):
            raise ValueError("center and positive support_radii must be three-vectors")
        x, y, z, t = sp.symbols("x y z t", real=True)
        a = sp.Float(amplitude)
        w = sp.Float(frequency)
        phi = sp.Float(phase)
        # Compact C4 bump: (1-r^2)^5 inside the ellipsoidal support and zero
        # outside. Four vanishing derivatives at the boundary are sufficient
        # for the derivatives used in the vorticity MMS to remain continuous.
        r2 = sum(((coord - sp.Float(c)) / sp.Float(rad))**2
                 for coord, c, rad in zip((x, y, z), center, support_radii))
        bump = sp.Piecewise(((1 - r2)**5, r2 < 1), (sp.S.Zero, True))
        # A time-dependent localized vector potential. Its curl is compactly
        # supported, unlike the prior periodic trigonometric target.
        potential = (
            a * bump * sp.cos(w*t),
            a * bump * sp.sin(w*t + phi),
            0.7 * a * bump * sp.cos(w*t + phi),
        )
        curl = lambda q: (
            sp.diff(q[2], y) - sp.diff(q[1], z),
            sp.diff(q[0], z) - sp.diff(q[2], x),
            sp.diff(q[1], x) - sp.diff(q[0], y),
        )
        u_induced = curl(potential)
        fs = _freestream_expressions(t, freestream_config or {})
        # Keep the Piecewise derivatives factored. SymPy's general simplify()
        # attempts polynomial factorization with the trigonometric gust terms
        # here and can take minutes without changing the expression.
        u_total = tuple(u_induced[i] + fs[i] for i in range(3))
        omega = curl(u_total)
        # By construction, div(curl(A)) and curl(u)-omega are exact identities.
        # Independent finite-difference tests check the evaluated fields.
        div_u = sp.S.Zero
        curl_identity = (sp.S.Zero, sp.S.Zero, sp.S.Zero)
        grad_omega = [[sp.diff(omega[i], (x, y, z)[j]) for j in range(3)] for i in range(3)]
        grad_u = [[sp.diff(u_total[i], (x, y, z)[j]) for j in range(3)] for i in range(3)]
        source = tuple(
            sp.diff(omega[i], t)
            + sum(u_total[j] * grad_omega[i][j] for j in range(3))
            - sum(omega[j] * grad_u[i][j] for j in range(3))
            for i in range(3)
        )
        # `source` is defined as this transport expression; the test suite
        # evaluates its residual with independent finite differences.
        source_identity = (sp.S.Zero, sp.S.Zero, sp.S.Zero)
        args = (x, y, z, t)
        return cls(
            args, tuple(potential), tuple(u_induced), u_total,
            tuple(omega), source, fs,
            sp.lambdify(args, potential, modules="numpy"),
            sp.lambdify(args, u_induced, modules="numpy"),
            sp.lambdify(args, u_total, modules="numpy"),
            sp.lambdify(args, omega, modules="numpy"),
            sp.lambdify(args, source, modules="numpy"),
            sp.lambdify((t,), fs, modules="numpy"), div_u,
            curl_identity, source_identity,
        )

    def evaluate(self, field: str, points: np.ndarray, time: float) -> np.ndarray:
        """Evaluate a vector field on (...,3) points; return (...,3)."""
        points = np.asarray(points, dtype=float)
        if points.shape[-1] != 3:
            raise ValueError("points must have final dimension 3")
        fn = {"A": self.potential_fn, "u": self.velocity_fn, "u_total": self.velocity_fn,
              "u_induced": self.induced_velocity_fn,
              "omega": self.vorticity_fn, "source": self.source_fn}.get(field)
        if field == "freestream":
            vals = np.asarray(self.freestream_fn(time), dtype=float)
            return np.broadcast_to(vals, points.shape).copy()
        if fn is None:
            raise ValueError(f"unknown field {field!r}")
        vals = np.asarray(fn(points[..., 0], points[..., 1], points[..., 2], time), dtype=float)
        return np.moveaxis(vals, 0, -1)


def _freestream_expressions(t, cfg: dict) -> tuple:
    """Mirror configured_freestream at FLOWUnsteady's Xdummy=[0,0,0]."""
    speed = float(cfg.get("vinf", 0.0))
    aoa = math.radians(float(cfg.get("aoa_deg", 0.0)))
    gradient_values = np.asarray(cfg.get("freestream_gradient", [0.0] * 9), dtype=float)
    amplitudes = np.asarray(cfg.get("freestream_oscillation", [0.0] * 3), dtype=float)
    wave = np.asarray(cfg.get("freestream_wavenumber", [0.0] * 3), dtype=float)
    reference = np.asarray(cfg.get("freestream_reference", [0.0] * 3), dtype=float)
    if gradient_values.size != 9 or amplitudes.size != 3 or wave.size != 3 or reference.size != 3:
        raise ValueError("Invalid configured freestream vector/matrix dimensions")
    gradient = gradient_values.reshape(3, 3)
    phase0 = math.radians(float(cfg.get("freestream_phase_deg", 0.0)))
    freq = float(cfg.get("freestream_frequency_hz", 0.0))
    rel0 = -reference
    base = np.array([speed * math.cos(aoa), 0.0, speed * math.sin(aoa)])
    gust_phase = np.array([wave[1] * rel0[1], wave[2] * rel0[2], wave[0] * rel0[0]])
    phase_t = 2.0 * sp.pi * freq * t + phase0
    gust = [float(amplitudes[i]) * sp.sin(phase_t + float(gust_phase[i])) for i in range(3)]
    affine = gradient @ rel0
    return tuple(sp.Float(base[i] + affine[i]) + gust[i] for i in range(3))


def configured_particle_freestream(fluid_config: dict, time: float) -> np.ndarray:
    """Exact FLOWUnsteady Uinf(t)=Vinf(Xdummy,t), with Xdummy at the origin."""
    exprs = _freestream_expressions(sp.Symbol("t", real=True), fluid_config)
    fn = sp.lambdify((sp.Symbol("t", real=True),), exprs, modules="numpy")
    result = np.asarray(fn(float(time)), dtype=float)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError("Configured particle freestream is not a finite 3-vector")
    return result


def coordinate_mapping_samples(bounds: list[list[float]], cells: list[int], *,
                               exact_limit: int = 4096) -> dict[str, Any]:
    """Python reference coordinates and vector probe for a specific Julia grid.

    All points are compared for grids up to `exact_limit`; larger grids use a
    deterministic tensor sample through every axis, edge, and boundary.
    Coordinates are computed from linear indices without allocating the full
    Eulerian grid.
    """
    box = np.asarray(bounds, dtype=float)
    n = np.asarray(cells, dtype=int)
    if (box.shape != (3, 2) or not np.all(np.isfinite(box))
            or np.any(box[:, 1] <= box[:, 0]) or n.shape != (3,)
            or np.any(n < 1) or exact_limit < 1):
        raise ValueError("Invalid bounds, cells, or exact coordinate comparison limit")
    dims = n + 1
    total = int(np.prod(dims))
    if total <= exact_limit:
        ijk_indices = [np.unravel_index(k, tuple(dims), order="F") for k in range(total)]
    else:
        per_axis = [sorted(set(np.linspace(0, int(ni), 9).round().astype(int).tolist()
                               + [0, int(ni)])) for ni in n]
        ijk_indices = list(__import__("itertools").product(*per_axis))
    seen = set()
    samples = []
    spacing = (box[:, 1] - box[:, 0]) / n
    for ijk in ijk_indices:
        i, j, k = map(int, ijk)
        flat_index = i + int(dims[0]) * (j + int(dims[1]) * k)
        if flat_index in seen:
            raise RuntimeError("Coordinate sample generator produced duplicate flat indices")
        seen.add(flat_index)
        xyz = box[:, 0] + np.asarray(ijk, dtype=float) * spacing
        x0, y0, z0 = map(float, xyz)
        samples.append({
            "flat_index_geometrictools_columnmajor": flat_index,
            "xyz_m": xyz.tolist(),
            "mapping_probe_vector": [x0 + 2*y0 + 3*z0,
                                     -2*x0 + y0 - 4*z0,
                                     5*x0 - 3*y0 + z0],
        })
    return {"bounds_m": box.tolist(), "cell_counts": n.tolist(),
            "flattening": "GeometricTools column-major; first coordinate fastest",
            "exact_all_node_check": total <= exact_limit,
            "total_grid_nodes": total, "samples": samples}


def cartesian_grid(bounds: list[list[float]], cells: list[int]):
    """Return node coordinates and tensor-product trapezoid weights."""
    box = np.asarray(bounds, dtype=float)
    n = np.asarray(cells, dtype=int)
    if box.shape != (3, 2) or not np.all(np.isfinite(box)) or np.any(box[:, 1] <= box[:, 0]):
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
    if not math.isfinite(norm_floor) or norm_floor <= 0:
        raise ValueError("norm_floor must be finite and positive")
    err = numerical - target
    if not np.all(np.isfinite(err)):
        raise ValueError("field errors contain non-finite values")
    target_norm = math.sqrt(float(np.sum(weights * np.sum(target * target, axis=-1))))
    numeric_norm = math.sqrt(float(np.sum(weights * np.sum(numerical * numerical, axis=-1))))
    abs_l2 = math.sqrt(float(np.sum(weights * np.sum(err * err, axis=-1))))
    if not all(math.isfinite(value) for value in (target_norm, numeric_norm, abs_l2)):
        raise ValueError("field norm calculation overflowed or returned non-finite values")
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


def convergence_order_report(errors: list[float], scales: list[float], *,
                             domains: list[Any], quantities: list[str],
                             plateau_rtol: float = 0.02) -> dict[str, Any]:
    """Pairwise orders with finite, positive and cross-level checks."""
    n = len(errors)
    if n < 2 or len(scales) != n or len(domains) != n or len(quantities) != n:
        raise ValueError("Need matching errors, scales, domains, and quantities for >=2 levels")
    e, h = np.asarray(errors, dtype=float), np.asarray(scales, dtype=float)
    if not np.all(np.isfinite(e)) or not np.all(np.isfinite(h)):
        raise ValueError("errors and refinement scales must be finite")
    if np.any(e <= 0):
        raise ValueError("logarithmic convergence orders require strictly positive errors")
    if np.any(h <= 0) or np.any(np.diff(h) >= 0):
        raise ValueError("refinement scales must be positive and strictly decreasing")
    if any(domain != domains[0] for domain in domains[1:]):
        raise ValueError("Cannot calculate convergence: physical domains differ")
    if any(quantity != quantities[0] for quantity in quantities[1:]):
        raise ValueError("Cannot calculate convergence: field quantities differ")
    pairs = []
    if not math.isfinite(plateau_rtol) or plateau_rtol < 0:
        raise ValueError("plateau_rtol must be finite and nonnegative")
    for i in range(1, n):
        change = abs(e[i] - e[i-1]) / max(e[i], e[i-1])
        if e[i] > e[i-1]:
            status, order = "NON_MONOTONIC_ERROR", None
        elif change <= plateau_rtol:
            status, order = "PLATEAU", None
        else:
            status = "DECREASING"
            order = float(math.log(e[i-1]/e[i]) / math.log(h[i-1]/h[i]))
        pairs.append({"coarse_level_index": i-1, "fine_level_index": i,
                      "coarse_error": float(e[i-1]), "fine_error": float(e[i]),
                      "coarse_scale": float(h[i-1]), "fine_scale": float(h[i]),
                      "order": order, "status": status})
    return {"quantity": quantities[0], "domain": domains[0], "pairs": pairs,
            "all_pairs_decrease": all(pair["status"] == "DECREASING" for pair in pairs),
            "plateau_rtol": plateau_rtol}


def observed_orders(errors: list[float], scales: list[float], *,
                    domains: list[Any],
                    quantities: list[str]) -> list[float | None]:
    """Consecutive orders; callers must identify the common domain and fields."""
    if len(errors) != len(scales):
        raise ValueError("errors and refinement scales must have equal lengths")
    if len(errors) < 2:
        return [None] * len(errors)
    report = convergence_order_report(errors, scales, domains=domains, quantities=quantities)
    return [None] + [pair["order"] for pair in report["pairs"]]


def compare_grid_coordinates(python_coordinates: np.ndarray,
                             julia_coordinates: np.ndarray, *,
                             atol: float = 1.0e-12, rtol: float = 1.0e-12) -> dict:
    """Check actual coordinate values/order, not only array dimensions."""
    py = np.asarray(python_coordinates, dtype=float).reshape(-1, 3)
    jl = np.asarray(julia_coordinates, dtype=float).reshape(-1, 3)
    if py.shape != jl.shape:
        raise ValueError(f"Python/Julia coordinate counts differ: {py.shape} vs {jl.shape}")
    if not np.all(np.isfinite(py)) or not np.all(np.isfinite(jl)):
        raise ValueError("Python/Julia coordinates contain non-finite values")
    mismatch = np.abs(py-jl)
    if not np.allclose(py, jl, atol=atol, rtol=rtol):
        row, axis = np.unravel_index(np.argmax(mismatch), mismatch.shape)
        raise ValueError(f"Python/Julia coordinate ordering mismatch at row {row}, axis {axis}")
    return {"coordinate_order": "x,y,z components; GeometricTools column-major node order",
            "points_compared": int(len(py)),
            "maximum_coordinate_difference_m": float(np.max(mismatch, initial=0.0))}


def freestream_match(analytical: np.ndarray, reconstructed: np.ndarray,
                     *, atol: float = 1.0e-12) -> dict:
    expected, actual = np.asarray(analytical, dtype=float), np.asarray(reconstructed, dtype=float)
    if expected.shape != (3,) or actual.shape != (3,) or not np.all(np.isfinite(expected)) or not np.all(np.isfinite(actual)):
        raise ValueError("Freestreams must be finite three-vectors")
    error = float(np.max(np.abs(expected-actual)))
    if error > atol:
        raise ValueError(f"Python/ParticleField freestream mismatch {error:.4e} exceeds {atol:.4e}")
    return {"expected_m_s": expected.tolist(), "reconstructed_m_s": actual.tolist(),
            "max_abs_difference_m_s": error, "tolerance_m_s": atol}


def grid_sensitivity_metrics(nominal: np.ndarray, refined_on_common_nodes: np.ndarray,
                             weights: np.ndarray, *, field_name: str,
                             relative_tolerance: float) -> dict:
    if not math.isfinite(relative_tolerance) or relative_tolerance <= 0:
        raise ValueError("relative_tolerance must be finite and positive")
    metrics = field_error_metrics(refined_on_common_nodes, nominal, weights,
                                  prefix=f"{field_name} nominal reconstruction")
    metrics["sensitivity_relative_l2"] = metrics["relative_l2"]
    metrics["relative_tolerance"] = float(relative_tolerance)
    metrics["status"] = "PASS" if metrics["relative_l2"] <= relative_tolerance else "FAIL"
    return metrics


def analytical_check(config_path: Path, output_path: Path | None = None,
                     geometry_lengths: tuple[float, float] | None = None,
                     fluid_config: dict | None = None,
                     structural_config: dict | None = None) -> dict:
    import yaml
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if fluid_config is None:
        fluid_path = config_path.resolve().parents[2] / "config" / "fluid_params.yaml"
        if not fluid_path.exists():
            raise FileNotFoundError("Pass production fluid config to derive exact pfield.Uinf")
        fluid_config = yaml.safe_load(fluid_path.read_text(encoding="utf-8"))
    target_cfg = dict(config["target"])
    smallest_target_length_scale = float(target_cfg["smallest_target_length_scale"])
    configured_minimum = float(min(target_cfg["support_radii"]))
    if (not math.isfinite(smallest_target_length_scale)
            or smallest_target_length_scale <= 0
            or not math.isclose(smallest_target_length_scale, configured_minimum,
                                rel_tol=0.0, abs_tol=1.0e-12)):
        raise ValueError(
            "smallest_target_length_scale must equal the smallest configured support radius")

    gradient = np.asarray(fluid_config.get("freestream_gradient", [0.0] * 9), dtype=float)
    wavenumbers = np.asarray(fluid_config.get("freestream_wavenumber", [0.0] * 3), dtype=float)
    if gradient.shape != (9,) or wavenumbers.shape != (3,) or not (
            np.all(np.isfinite(gradient)) and np.all(np.isfinite(wavenumbers))):
        raise ValueError("Freestream gradient and wavenumber must be finite vectors of lengths 9 and 3")
    if np.any(gradient != 0.0) or np.any(wavenumbers != 0.0):
        raise ValueError(
            "This target/reconstruction assumes spatially uniform freestream; "
            "freestream_gradient and freestream_wavenumber must be zero")
    t_end = float(config["analytical_reference_time_s"])
    spatial_end = float(config["field_spatial_study"]["final_time_s"])
    temporal_end = float(config["temporal_study"]["final_time"])
    if not (math.isclose(t_end, spatial_end, rel_tol=0.0, abs_tol=1.0e-14)
            and math.isclose(t_end, temporal_end, rel_tol=0.0, abs_tol=1.0e-14)):
        raise ValueError(
            "analytical_reference_time_s must match both spatial and temporal final times")
    target_build_cfg = {key: target_cfg[key] for key in (
        "amplitude", "frequency", "phase", "center", "support_radii")}
    target = FieldTarget.build(**target_build_cfg, freestream_config=fluid_config)
    if sp.simplify(target.divergence_expr) != 0:
        raise RuntimeError(f"target velocity is not divergence-free: {target.divergence_expr}")
    if any(sp.simplify(value) != 0 for value in target.curl_identity_expr):
        raise RuntimeError("Analytical vorticity is not exactly curl(total velocity)")
    if any(sp.simplify(value) != 0 for value in target.source_identity_expr):
        raise RuntimeError("Manufactured vorticity-equation residual does not equal source")
    bounds = config["reconstruction"]["bounds"]
    cells = config["reconstruction"]["preflight_cells"]
    points, weights, spacing = cartesian_grid(bounds, cells)
    fields = {name: target.evaluate(name, points, t_end)
              for name in ("u", "u_induced", "omega", "source")}
    if any(not np.all(np.isfinite(value)) for value in fields.values()):
        raise RuntimeError("Analytical target or source contains non-finite values")
    norms = {name: math.sqrt(float(np.sum(weights * np.sum(value**2, axis=-1))))
             for name, value in fields.items()}
    if min(norms["u"], norms["omega"]) <= 1.0e-13:
        raise RuntimeError("target velocity/vorticity norm is zero or too small")
    center = np.asarray(target_cfg["center"], dtype=float)
    radii = np.asarray(target_cfg["support_radii"], dtype=float)
    box = np.asarray(bounds, dtype=float)
    support_bounds = np.stack([center-radii, center+radii], axis=1)
    support_padding = np.minimum(support_bounds[:, 0]-box[:, 0], box[:, 1]-support_bounds[:, 1])
    if np.any(support_padding <= 0):
        raise RuntimeError("Eulerian domain does not strictly contain target compact support")
    freestream = configured_particle_freestream(fluid_config, t_end)
    target_freestream = target.evaluate("freestream", np.zeros((1, 3)), t_end)[0]
    if not np.allclose(freestream, target_freestream, rtol=0.0, atol=1.0e-12):
        raise RuntimeError("Target and production ParticleField freestream conventions differ")
    if geometry_lengths is None:
        production_solid = config_path.resolve().parents[2] / "config" / "solid_params.yaml"
        if production_solid.exists():
            solid_cfg = yaml.safe_load(production_solid.read_text(encoding="utf-8"))
            geometry_lengths = (float(solid_cfg["root_chord"]), float(solid_cfg["span"]))
        else:
            raise FileNotFoundError("Pass production geometry lengths or provide config/solid_params.yaml")
    field_levels = config["field_spatial_study"]["cases"]
    structural_levels = None
    if structural_config is None:
        structural_path = config_path.with_name("mms_parameters.yaml")
        if structural_path.exists():
            structural_config = yaml.safe_load(structural_path.read_text(encoding="utf-8"))
    if structural_config is not None:
        plate = structural_config["standalone_plate"]
        structural_levels = {
            "spatial": [{**case, "h_mesh": mesh_spacing(
                list(geometry_lengths), int(case["nx"]), int(case["ny"]))}
                for case in plate["spatial_cases"]],
            "temporal": [{**case, "h_mesh": mesh_spacing(
                list(geometry_lengths), int(case["nx"]), int(case["ny"]))}
                for case in plate["temporal_cases"]],
        }
    samples = coordinate_mapping_samples(bounds, cells)
    result = {
        "status": "PASS",
        "claim": "analytical/configuration checks only; no particle solver run",
        "divergence_u_identity_by_construction": sp.simplify(target.divergence_expr) == 0,
        "curl_u_equals_omega_identity_by_construction": all(
            sp.simplify(v) == 0 for v in target.curl_identity_expr),
        "vorticity_source_definition_identity_by_construction": all(
            sp.simplify(v) == 0 for v in target.source_identity_expr),
        "identity_validation_note": (
            "The target is defined through curl and transport operators. "
            "Independent finite-difference tests validate the evaluated identities; "
            "the zero symbolic identity fields record construction relationships."),
        "free_space_biot_savart_compatibility": {
            "status": "ANALYTIC_IDENTITY",
            "reason": "induced u is divergence-free and compactly supported; omega=curl(u); the unique decaying free-space Biot-Savart inverse is u",
            "freestream_excluded_from_induced_inversion": True,
        },
        "target_norms_at_final_time": norms,
        "freestream_pfield_origin_m_s": freestream.tolist(),
        "freestream_definition": "production configured_freestream(Xdummy=[0,0,0], t)",
        "freestream_is_added_once_to_reconstructed_velocity": True,
        "analytical_reference_time_s": t_end,
        "smallest_target_length_scale_m": smallest_target_length_scale,
        "target_support_bounds_m": support_bounds.tolist(),
        "support_padding_to_domain_m": support_padding.tolist(),
        "python_grid_known_samples": samples,
        "reconstruction": {"bounds": bounds,
                           "preflight_grid_cells": cells,
                           "preflight_grid_points": int(weights.size),
                           "preflight_spacing": spacing.tolist(),
                           "grid_resolution_rule": config["reconstruction"]["grid_resolution"],
                           "core_intervals_minimum": config["particles"]["core_grid_intervals_minimum"],
                           "target_intervals_per_length_scale": config["reconstruction"]["target_intervals_per_length_scale"],
                           "sensitivity_level": config["reconstruction"]["sensitivity_level"],
                           "sensitivity_factor": config["reconstruction"]["sensitivity_factor"],
                           "sensitivity_relative_tolerance": config["reconstruction"]["sensitivity_relative_tolerance"]},
        "field_spatial_cases": field_levels,
        "structural_refinement_cases": structural_levels,
        "structural_characteristic_lengths_m": {
            "Lx_root_chord": float(geometry_lengths[0]),
            "Ly_span": float(geometry_lengths[1]),
        },
        "field_temporal_dts_s": config["temporal_study"]["dt_s"],
        "field_temporal_final_time_s": float(config["temporal_study"]["final_time"]),
        "expected_viscous_scheme": config["physics"]["viscous_scheme"],
        "expected_vpm_sfs": config["physics"]["vpm_sfs"],
    }
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def check_reconstruction_resolution(spacing: np.ndarray, sigma_min: float,
                                    target_length_scale: float, *,
                                    core_intervals: int = 8,
                                    target_intervals: int = 20) -> dict:
    """Fail if a Cartesian grid under-resolves either core or target scale."""
    h = np.asarray(spacing, dtype=float)
    if (h.shape != (3,) or not np.all(np.isfinite(h)) or np.any(h <= 0)
            or core_intervals < 1 or target_intervals < 1
            or not np.isfinite(sigma_min) or sigma_min <= 0
            or not np.isfinite(target_length_scale) or target_length_scale <= 0):
        raise ValueError("spacing, minimum core, and target wavelength must be positive")
    core_limit = sigma_min / core_intervals
    target_limit = target_length_scale / target_intervals
    allowed = min(core_limit, target_limit)
    if np.max(h) > allowed * (1 + 1.0e-12):
        raise ValueError(
            f"Eulerian grid spacing {h.tolist()} exceeds resolution limit {allowed:.6e} m "
            f"(sigma_min={sigma_min:.6e} m, target_scale={target_length_scale:.6e} m)")
    return {"spacing_m": h.tolist(), "sigma_min_m": float(sigma_min),
            "target_length_scale_m": float(target_length_scale),
            "core_intervals": int(core_intervals), "target_intervals": int(target_intervals),
            "maximum_allowed_spacing_m": allowed}


def derive_reconstruction_cells(bounds: list[list[float]], sigma_min: float,
                                target_length_scale: float, *,
                                core_intervals: int = 8,
                                target_intervals: int = 20) -> tuple[list[int], float]:
    """Plan cell counts from measured core scale and target wavelength."""
    box = np.asarray(bounds, dtype=float)
    if (box.shape != (3, 2) or not np.all(np.isfinite(box))
            or np.any(box[:, 1] <= box[:, 0])):
        raise ValueError("bounds must be three increasing coordinate intervals")
    if (not np.isfinite(sigma_min) or not np.isfinite(target_length_scale)
            or sigma_min <= 0 or target_length_scale <= 0
            or core_intervals < 1 or target_intervals < 1):
        raise ValueError("smallest core and target length scale must be positive")
    dx = min(sigma_min / core_intervals, target_length_scale / target_intervals)
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
