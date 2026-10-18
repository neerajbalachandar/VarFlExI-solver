"""Dependency-free generalized-alpha coefficient and update equations.

These arithmetic helpers are shared by the DOLFIN integrator and the scalar
oscillator verification, so the test exercises the production update formulas.
They also accept UFL/DOLFIN expressions through their ordinary arithmetic.
"""


def parameters(alpha_m, alpha_f):
    gamma = 0.5 + alpha_f - alpha_m
    beta = (gamma + 0.5) ** 2 / 4.0
    return gamma, beta


def acceleration(q_new, q_old, v_old, a_old, dt, beta):
    return (q_new - q_old - dt * v_old) / (beta * dt**2) - (1.0 - 2.0 * beta) / (2.0 * beta) * a_old


def velocity(a_new, v_old, a_old, dt, gamma):
    return v_old + dt * ((1.0 - gamma) * a_old + gamma * a_new)


def intermediate(old, new, alpha):
    return alpha * old + (1.0 - alpha) * new
