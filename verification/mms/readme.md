# Coupled MMS Target: Construction and Current Boundary

## Purpose

This note defines what a single manufactured trajectory must contain for the
production staggered FLOWVLM/rVPM--Reissner--Mindlin solver. It is a derivation
and implementation contract, not a claim that the coupled MMS is implemented.
The bound-circulation solve, rVPM stage source, particle lifecycle
instrumentation, moving-interface CRM/RBF tests, and socket run remain
outstanding.

Let the target be

```text
S_hat(t) = {q_s_hat(t), X_a_hat(t), Gamma_b_hat(t), P_hat(t), F_a_hat(t)}.
```

All members are coupled by the equations below; they must not be prescribed as
independent files of values.

## Structural State and Exact Aerodynamic Geometry

Use the smooth plate fields in `plate.py` as the initial structural candidate.
With `B(x,y)=sin(pi*x/Lx)^4 sin(pi*y/Ly)^4` and `omega=2.3 rad/s`,

```text
u0_hat = 1e-4 B cos(omega t)       v0_hat = 1e-4 B sin(omega t)
w_hat  = 1e-3 B sin(omega t)      theta_x_hat = 1e-3 B cos(omega t)
                                    theta_y_hat = -1e-3 B sin(omega t).
```

The structural target includes analytical first and second time derivatives.
The existing symbolic plate module derives inertia, implemented Rayleigh
damping, membrane, bending, and shear terms. This target is for the current
unswept rectangular plate assumptions; swept/tapered geometry requires a
re-derived field and boundary verification.

Let `X_s(x,y)` be the reference midsurface and `D(X;q)` the production
Reissner--Mindlin displacement/rotation kinematic map, including the
through-thickness offset used to place LE/TE. Define continuous target
geometry by evaluation at aerodynamic geometry coordinates:

```text
X_a_hat(eta,xi,t) = X_s(eta,xi) + D(X_s(eta,xi); q_s_hat(t)).
```

For LE/TE geometry, evaluate at `xi=0` and `xi=1`. Each panel is constructed
from the adjacent shared points `LE_i, TE_i, TE_(i+1), LE_(i+1)`. Reconstruct
CP and bound-vortex locations from those same points at chord fractions 3/4
and 1/4. There is no independent CP geometry target.

This is the analytic reference geometry. The numerical path remains production
structural-field evaluation followed by the separate RBF geometry map. Their
difference is a geometry-transfer error and must be measured, not folded into
the analytic reference.

## Bound Circulation and Wake Target

At each fluid time, target circulation is the solution of the installed
FLOWVLM no-penetration system on `X_a_hat`, using target incoming flow,
target wake-induced velocity, and production panel normals/regularization:

```text
A_VLM(X_a_hat) Gamma_b_hat
  = - n_hat dot (V_infinity + u_induced(P_hat)).
```

It must be computed by the real circulation solver. Prescribing
`Gamma_b_hat` is valid only if its residual in this discrete system is shown
to be at the solver tolerance.

For every active target particle `p`, let
`z_p_hat=(x_p_hat,Gamma_p_hat,sigma_p_hat)`. Specify deterministic birth time,
lifetime/removal event, and initial data. Between events, define its source
against the production particle RHS:

```text
s_MMS,p(t) = d z_p_hat/dt
             - G_p(P_hat,Gamma_b_hat,q_s_hat,dot(q_s_hat),t).
```

Evaluate `G_p` independently at the integrator's actual Runge--Kutta stages;
an endpoint-only source does not manufacture the RK3 method. At birth, the
target must match the production shedding event map. At removal, record the
event and ensure the removed particle no longer contributes. This source may
correct particle ODE residuals but must not disable circulation, RK stages,
configured particle management, or field reconstruction.

Compare particle states by identity only while lifetimes overlap. Also compare
reconstructed velocity/vorticity fields on a common grid. Report birth/removal
event mismatches separately; do not infer a particle-wise order across
different event histories.

## Aerodynamic Force and Structural Manufactured Load

Compute target force from the installed force law on the jointly computed
target state:

```text
F_a_hat = F_circulatory(X_a_hat,Gamma_b_hat,P_hat,dot(X_a_hat),t)
          + F_unsteady(as defined by the installed FLOWUnsteady version)
          + F_parasitic (only when a sectional polar is configured).
```

The unsteady term uses the previous target VLM state required by the installed
`dGamma/dt` implementation; time-varying geometry alone does not establish
it. The circulatory contribution appears exactly once. Integrate force onto
structural virtual fields using an independent reference surface quadrature
`L_ref(F_a_hat)`. This quadrature must not call the CRM/RBF operator being
tested.

For test function `delta q`, construct the structural manufactured load as
the residual of the intended structural equation:

```text
L_s,MMS(delta q,t) = M(a_hat,delta q)
                    + C(dot(q_hat),delta q)
                    + K_int(q_hat;delta q)
                    - L_ref(F_a_hat,delta q).
```

The numerical run still computes force, applies the selected production CRM
or conservative RBF transfer, and advances the structure. It must not insert
`F_a_hat` directly into the numerical structural right-hand side. The
independent reference load is only used to derive the manufactured source,
preventing a transfer defect from canceling itself in that source.

## Discrete Chronology

The production coupling is staggered, not monolithic. The fluid uses its
currently installed geometry, solves circulation, advances/manages particles,
evaluates force using the previous VLM state, and sends force. The solid maps
that force, advances generalized-alpha, evaluates updated LE/TE displacement,
and sends geometry for a subsequent fluid step. Structural inertia is at
`n+alpha_m`; velocity and internal force are at `n+alpha_f`; the aerodynamic
force has the fluid callback's own time level. Tag every manufactured value
with the physical time/stage actually used; do not silently treat these levels
as coincident.

## Exit Checks

1. Verify the structural target/source against independent strong equations,
   production weak forms, and generalized-alpha time levels.
2. Verify CRM and RBF geometry/force on evolving interfaces with constant and
   varying fields, endpoints, multiple resolutions, resultant force/moment,
   interpolation error, and virtual work. Accuracy and conservation are
   separate metrics.
3. Add rVPM source hooks at every RK stage while preserving the real VLM
   circulation, force law, and particle birth/removal; independently test
   source residuals and event consistency.
4. Run the socket-coupled staggered solver and log its chronology/events.
5. Only after at least three defensible levels, report separate structural,
   circulation, particle-field, force, transfer, and coupled response orders.

Until these checks execute, this is a construction specification. No full
coupled target or convergence order is claimed.
