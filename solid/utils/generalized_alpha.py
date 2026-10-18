from dolfin import *
from .generalized_alpha_coefficients import acceleration, intermediate, parameters, velocity


class GeneralizedAlphaIntegrator:
    """Generalized-alpha method with old-state weight ``alpha_m/alpha_f``.

    ``average(old,new,alpha) = alpha*old + (1-alpha)*new``. The model thus
    evaluates inertia at ``a_(n+alpha_m)`` and damping/internal force at
    ``v_(n+alpha_f)`` / ``q_(n+alpha_f)``. Kinematic updates follow the
    Newmark forms using ``gamma=.5+alpha_f-alpha_m`` and
    ``beta=(gamma+.5)^2/4``.
    """

    def __init__(self, dt, alpha_m, alpha_f):
        self.dt = dt
        self.alpha_m = alpha_m
        self.alpha_f = alpha_f
        gamma, beta = parameters(float(self.alpha_m), float(self.alpha_f))
        self.gamma = Constant(gamma)
        self.beta = Constant(beta)

    def average(self, x_old, x_new, alpha):
        return intermediate(x_old, x_new, alpha)

    def update_a(self, q_new, q_prev, v_prev, a_prev, ufl=True):
        dt_ = self.dt if ufl else float(self.dt)
        beta_ = self.beta if ufl else float(self.beta)
        return acceleration(q_new, q_prev, v_prev, a_prev, dt_, beta_)

    def update_v(self, a_new, q_prev, v_prev, a_prev, ufl=True):
        dt_ = self.dt if ufl else float(self.dt)
        gamma_ = self.gamma if ufl else float(self.gamma)
        return velocity(a_new, v_prev, a_prev, dt_, gamma_)

    def update_fields(self, q_fun, q_prev, v_prev, a_prev):
        q_vec = q_fun.vector()
        q0_vec = q_prev.vector()
        v0_vec = v_prev.vector()
        a0_vec = a_prev.vector()

        a_vec = self.update_a(q_vec, q0_vec, v0_vec, a0_vec, ufl=False)
        v_vec = self.update_v(a_vec, q0_vec, v0_vec, a0_vec, ufl=False)

        v_prev.vector()[:] = v_vec
        a_prev.vector()[:] = a_vec
        q_prev.vector()[:] = q_vec


__all__ = ["GeneralizedAlphaIntegrator"]
