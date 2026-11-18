"""Independent numerical checks for the localized velocity-vorticity target."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from field_mms import FieldTarget, analytical_check


REPO = HERE.parents[1]
CONFIG = yaml.safe_load((HERE / "field_mms_parameters.yaml").read_text())
FLUID = yaml.safe_load((REPO / "config" / "fluid_params.yaml").read_text())
TARGET_CFG = CONFIG["target"]
TIME = float(CONFIG["analytical_reference_time_s"])
RADII = np.asarray(TARGET_CFG["support_radii"], dtype=float)
CENTER = np.asarray(TARGET_CFG["center"], dtype=float)


@pytest.fixture(scope="module")
def target():
    return FieldTarget.build(
        **{key: TARGET_CFG[key] for key in
           ("amplitude", "frequency", "phase", "center", "support_radii")},
        freestream_config=FLUID,
    )


def jacobian(field, point, time, h):
    result = np.empty((3, 3), dtype=float)
    for axis in range(3):
        delta = np.zeros(3)
        delta[axis] = h
        result[:, axis] = (
            field(point + delta, time) - field(point - delta, time)
        ) / (2.0 * h)
    return result


def points_by_support_region(seed=20261117):
    rng = np.random.default_rng(seed)
    interior = []
    exterior = []
    while len(interior) < 3 or len(exterior) < 3:
        normalized = rng.uniform(-1.0, 1.0, 3)
        point = CENTER + RADII * normalized
        radius2 = float(np.sum(normalized**2))
        if radius2 < 0.75 and len(interior) < 3:
            interior.append(point)
        elif radius2 > 1.25 and len(exterior) < 3:
            exterior.append(point)
    near_boundary = [CENTER + RADII * np.array([1.0 - 2e-3, 0.0, 0.0]),
                     CENTER + RADII * np.array([1.0 + 2e-3, 0.0, 0.0])]
    return interior + exterior + near_boundary


def test_preflight_validates_target_scale_time_and_uniform_freestream(tmp_path):
    report = analytical_check(HERE / "field_mms_parameters.yaml")
    assert report["smallest_target_length_scale_m"] == pytest.approx(min(RADII))
    assert report["analytical_reference_time_s"] == TIME

    bad_scale = dict(CONFIG)
    bad_scale["target"] = dict(CONFIG["target"], smallest_target_length_scale=0.5)
    bad_path = tmp_path / "bad-scale.yaml"
    bad_path.write_text(yaml.safe_dump(bad_scale))
    with pytest.raises(ValueError, match="smallest_target_length_scale"):
        analytical_check(bad_path, fluid_config=FLUID)

    bad_fluid = dict(FLUID, freestream_gradient=[0.0] * 8 + [1e-6])
    with pytest.raises(ValueError, match="spatially uniform"):
        analytical_check(HERE / "field_mms_parameters.yaml", fluid_config=bad_fluid)

    bad_time = dict(CONFIG, analytical_reference_time_s=0.04)
    bad_time_path = tmp_path / "bad-time.yaml"
    bad_time_path.write_text(yaml.safe_dump(bad_time))
    with pytest.raises(ValueError, match="must match both"):
        analytical_check(bad_time_path, fluid_config=FLUID)


def test_numerical_divergence_is_zero(target):
    h = 2e-5 * float(np.min(RADII))
    values = []
    for point in points_by_support_region():
        grad_u = jacobian(lambda x, t: target.evaluate("u_total", x, t), point, TIME, h)
        values.append(float(np.trace(grad_u)))
    # Centered-difference truncation is O(h^2); 2e-5 of the support scale
    # leaves ample headroom for floating-point cancellation in the curl field.
    assert max(abs(value) for value in values) < 2e-7, values


def test_centered_difference_curl_matches_analytical_vorticity(target):
    h = 2e-5 * float(np.min(RADII))
    worst = 0.0
    for point in points_by_support_region():
        grad_u = jacobian(lambda x, t: target.evaluate("u_total", x, t), point, TIME, h)
        curl_fd = np.array([
            grad_u[2, 1] - grad_u[1, 2],
            grad_u[0, 2] - grad_u[2, 0],
            grad_u[1, 0] - grad_u[0, 1],
        ])
        curl_exact = target.evaluate("omega", point, TIME)
        worst = max(worst, float(np.linalg.norm(curl_fd - curl_exact, ord=np.inf)))
    assert worst < 3e-7, f"max centered-difference curl error={worst:.6e} 1/s"


def test_independent_vorticity_transport_residual(target):
    hx = 3e-5 * float(np.min(RADII))
    ht = 2e-5
    worst = 0.0
    for point in points_by_support_region():
        u = target.evaluate("u_total", point, TIME)
        omega = target.evaluate("omega", point, TIME)
        grad_u = jacobian(lambda x, t: target.evaluate("u_total", x, t), point, TIME, hx)
        grad_omega = jacobian(lambda x, t: target.evaluate("omega", x, t), point, TIME, hx)
        omega_t = (target.evaluate("omega", point, TIME + ht)
                   - target.evaluate("omega", point, TIME - ht)) / (2.0 * ht)
        source = target.evaluate("source", point, TIME)
        residual = omega_t + grad_omega @ u - grad_u @ omega - source
        worst = max(worst, float(np.linalg.norm(residual, ord=np.inf)))
    # Spatial and temporal differences are centered and second order; this
    # tolerance bounds their truncation at the stated h and ht independently
    # of the symbolic expression used to create the target.
    assert worst < 2e-5, f"max independent source residual={worst:.6e} 1/s^2"


def test_support_boundary_continuity(target):
    direction = np.array([1.0, 0.0, 0.0])
    delta = 2e-5
    inside = CENTER + RADII * (1.0 - delta) * direction
    outside = CENTER + RADII * (1.0 + delta) * direction
    for name, tolerance in (("A", 2e-16), ("u_total", 2e-13),
                            ("omega", 2e-9), ("source", 2e-5)):
        a = target.evaluate(name, inside, TIME)
        b = target.evaluate(name, outside, TIME)
        assert np.linalg.norm(a - b, ord=np.inf) < tolerance, (
            f"{name} is not continuous across support boundary: {a} vs {b}")


def _biot_savart_ray_quadrature(target, point, time, order):
    """Independent spherical-ray quadrature, with the 1/r singularity canceled."""
    mu, w_mu = np.polynomial.legendre.leggauss(order)
    radial_nodes, radial_weights = np.polynomial.legendre.leggauss(order)
    phi = 2.0 * np.pi * np.arange(2 * order) / (2 * order)
    dphi = 2.0 * np.pi / len(phi)
    directions = np.array([
        [math.sqrt(1.0 - m*m) * math.cos(p),
         math.sqrt(1.0 - m*m) * math.sin(p), m]
        for m in mu for p in phi
    ])
    angular_weights = np.repeat(w_mu, len(phi)) * dphi

    relative = point - CENTER
    inv_r2 = 1.0 / (RADII * RADII)
    aa = np.sum(directions * directions * inv_r2[None, :], axis=1)
    bb = 2.0 * np.sum(directions * relative[None, :] * inv_r2[None, :], axis=1)
    cc = float(np.sum(relative * relative * inv_r2) - 1.0)
    discriminant = bb*bb - 4.0*aa*cc
    assert np.all(discriminant > 0.0), "evaluation point must lie strictly inside support"
    rmax = (-bb + np.sqrt(discriminant)) / (2.0 * aa)

    radii = 0.5 * rmax[:, None] * (radial_nodes[None, :] + 1.0)
    weights_r = 0.5 * rmax[:, None] * radial_weights[None, :]
    eval_points = point[None, None, :] + radii[:, :, None] * directions[:, None, :]
    omega = target.evaluate("omega", eval_points.reshape(-1, 3), time).reshape(
        len(directions), order, 3)
    integrand = np.cross(omega, -directions[:, None, :])
    ray_integral = np.sum(integrand * weights_r[:, :, None], axis=1)
    return np.sum(angular_weights[:, None] * ray_integral, axis=0) / (4.0 * np.pi)


def test_free_space_biot_savart_quadrature_and_sensitivity(target, record_property):
    points = [CENTER + RADII * np.array([0.17, -0.12, 0.09]),
              CENTER + RADII * np.array([-0.21, 0.14, -0.08])]
    orders = (10, 16, 24, 36)
    errors = []
    final_sensitivities = []
    for point in points:
        exact = target.evaluate("u_induced", point, TIME)
        approximations = [_biot_savart_ray_quadrature(target, point, TIME, n)
                          for n in orders]
        err = float(np.linalg.norm(approximations[-1] - exact))
        sensitivity = float(np.linalg.norm(approximations[-1] - approximations[-2]))
        scale = max(float(np.linalg.norm(exact)), 1e-14)
        errors.append(err / scale)
        final_sensitivities.append(sensitivity / scale)
    report = {"quadrature_orders": orders,
              "relative_error_at_finest_order": errors,
              "relative_finest_pair_sensitivity": final_sensitivities,
              "error_source": "free-space Biot-Savart quadrature only; no particle reconstruction"}
    record_property("free_space_biot_savart_quadrature", report)
    assert max(final_sensitivities) < 2e-3, report
    assert max(errors) < 5e-3, report
