"""Dependency-free generalized-alpha coefficient and update equations.

These arithmetic helpers are shared by the DOLFIN integrator and the scalar
oscillator verification, so the test exercises the production update formulas.
They also accept UFL/DOLFIN expressions through their ordinary arithmetic.
"""


def parameters(alpha_m, alpha_f):
    # alpha_m/alpha_f are old-state weights in this implementation:
    # x_alpha = alpha*x_n + (1-alpha)*x_(n+1).  Converting the usual
    # second-order Newmark relation to these weights gives this sign.
    gamma = 0.5 + alpha_m - alpha_f
    beta = (gamma + 0.5) ** 2 / 4.0
    return gamma, beta


def parameters_from_spectral_radius(rho_inf):
    """Return old-state weights for the Chung-Hulbert rho-infinity method."""
    rho = float(rho_inf)
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho_inf must lie in [0, 1]")
    # Standard new-state weights are (2*rho-1)/(rho+1) and rho/(rho+1).
    # Convert each to this module's old-state weighting convention.
    return (2.0 - rho) / (1.0 + rho), 1.0 / (1.0 + rho)


def acceleration(q_new, q_old, v_old, a_old, dt, beta):
    return (q_new - q_old - dt * v_old) / (beta * dt**2) - (1.0 - 2.0 * beta) / (2.0 * beta) * a_old


def velocity(a_new, v_old, a_old, dt, gamma):
    return v_old + dt * ((1.0 - gamma) * a_old + gamma * a_new)


def intermediate(old, new, alpha):
    return alpha * old + (1.0 - alpha) * new
