# Coupled MMS Target: Construction and Current Boundary

## Verification Guide and Status

**Overall status: PARTIAL.** The repository has a structural plate MMS,
symbolic identities, a production weak/strong structural check, a discrete
FLOWVLM circulation component reference, and production transfer diagnostics.
It does not yet have a consistent rVPM manufactured source for the complete
particle state, a time-dependent circulation/force target, or a socket-coupled
dynamic FSI MMS. No component result is a full-FSI pass.

### Run Commands

Run from the repository root. The legacy DOLFIN checks use the `fenics-env`
environment; the standalone fluid study needs the Julia project used by
`run.sh`.

| Command | Purpose and principal outputs |
|---|---|
| `conda run -n fenics-env python verification/mms/run_mms.py --output-dir /tmp/mms-run` | Symbolic plate source/edge checks, target-derivative finite-difference checks, nonlinear membrane variation, scalar generalized-alpha study, and the RBF algebra component. Writes run-scoped JSON/CSV/report/plots under the selected directory; failed checks preserve a traceback in `mms_summary.json` and return nonzero. Existing FLOWVPM CSVs are never counted as results from this Python invocation. |
| `conda run -n fenics-env python verification/mms/strong_weak_check.py --mode continuous --formulation nonlinear --resolution 12x24` | Independent symbolic strong-form versus production UFL weak-form consistency; writes its selected CSV. This is not a transient solve. |
| `conda run -n fenics-env python verification/mms/jacobian_check.py` | Finite-difference directional check of the nonlinear structural residual/Jacobian. |
| `conda run -n fenics-env python verification/mms/fenics_plate.py --production-formulation --nx 12 --ny 16 --dt 0.001 --duration 0.01 --output /tmp/plate.csv` | One production Reissner-Mindlin transient structural-only MMS case; output includes continuous-target, FE projection, derived-field, and Newton metrics. |
| `conda run -n fenics-env python verification/mms/refinement.py --study spatial --execute --formulation nonlinear --output-dir /tmp/plate-spatial` | Five-level structural spatial refinement. Use `--study temporal` for six time steps on a fixed `48x128` mesh. The temporal series is expensive; do not assume it diagnoses the previously observed error floor without same-mesh time-reference checks. |
| `conda run -n fenics-env python verification/mms/transfer_moving_interface.py --config-dir config --levels 8x16,12x24 --csv /tmp/transfer.csv --json /tmp/transfer.json` | Calls production CRM/RBF transfer methods at undeformed and two deformed states on each mesh, with constant and varying forces; emits component-only geometry, resultant, moment, and work metrics. |
| `conda run -n fenics-env bash -lc 'LD_LIBRARY_PATH="$CONDA_PREFIX/lib" julia --project=.. verification/mms/circulation_reference_check.jl'` | Rebuilds the FLOWVLM horseshoe influence matrix and no-penetration RHS independently, compares the circulation with the installed solve, and checks a manufactured discrete circulation vector. Shares the installed vortex kernel; static geometry only. |
| `julia --project=.. verification/mms/particle_position_stage_check.jl` | Fixed-population component check of the native rVPM RK3 position update with a constant source and a zero-source regression. This does not verify circulation/core evolution or native wake lifecycle. |
| `julia --project=PATH_TO_FLOWUNSTEADY verification/mms/run_refinement.jl --study particles --execute` | Standalone FLOWUnsteady/rVPM sensitivity run, not particle MMS and not coupled FSI. Each case records config and particle-count observations under its results folder. `analyze.py` compares histories against a numerical reference, not an exact solution. |

The recent corrected structural spatial run used nonlinear production settings,
levels `6x8`, `12x16`, `24x32`, `48x64`, `96x128`, `dt=6.25e-5 s`, and final
time `0.01 s`; outputs were isolated at `/tmp/varflexi-spatial-corrected`.
Continuous-target pairwise orders coarse-to-fine were `2.94, 3.11, 3.12, 3.05`
for in-plane displacement, `2.97, 3.67, 3.83, 3.74` for transverse
displacement, `2.90, 3.63, 3.78, 3.68` for rotation, and `2.99, 3.66, 3.81,
3.69` for time-integrated transverse displacement. This result is
structural-only. The old temporal
time-integrated error floor near `0.289` has not been resolved by a same-mesh
fine-step comparison; no temporal plate rate is claimed. To preserve existing
outputs, send new refinement runs to a fresh `--output-dir`.

The project-owned `fluid/src/mms_rk3.jl` wraps the installed RK3 routine and
adds an opt-in source to the particle-position RHS at the native stages
`t`, `t+dt/3`, and `t+3dt/4`. The production solver enables this component hook
only when `VARFLEXI_MMS_POSITION_SOURCE="sx,sy,sz"` is set; values are constant
source velocities in m/s. The default path passes `vpm.rungekutta3` unchanged.
Enabled runs write `*_mms_rk3_stage_history.csv`; the source hook requires
`vpm_sfs=none` so it cannot leak into subfilter-scale strength evolution. The
fixed-population regression reports zero position error for constant velocity
forcing and zero complete-particle-state difference against native RK3 when
the source is zero. This hook manufactures only particle position evolution; it does not supply compatible sources for
circulation, core width, shedding, particle birth/removal, force, or structure.
It is not a coupled or complete rVPM MMS.

`PlateMMS.target_derivatives` is the canonical symbolic derivative collection
for the structural target. `PlateMMS.velocities` and `.accelerations` are used
directly by the structural MMS solver, and the Python component runner checks
them (plus first spatial derivatives) against centered finite differences.

The current Julia solver uses FLOWVLM's discrete horseshoe-vortex
no-penetration solve coupled to FLOWVPM/rVPM, not an actuator-line model. The
installed RK3 stages advance particle positions and circulation; core
spreading is a separate staged update and can remap particle strengths.
Native shedding, birth, core evolution, and removal are event-bearing
operations. There is no verification source hook at those stages and no
immutable particle event ledger matched against manufactured lifetimes.
Therefore neither the Taylor-Green continuum identity (also unavailable when
`fluid.py` is absent) nor the standalone particle sensitivity driver verifies
the production particle equations.

The installed force callback retains the previous VLM system and adds the
separately exposed unsteady force vector once. This force-path behavior is
production implementation context, not an independent force verification.
`circulation_reference_check.jl` independently assembles the discrete
horseshoe matrix and RHS and checks a manufactured circulation vector, while
reusing the installed production vortex kernel. Its 12-panel static check
produced relative circulation difference `2.77e-17`, production
no-penetration residual `2.26e-16`, and manufactured-target error `1.71e-16`.
It does not construct a moving target geometry, include wake induction, or
calculate a target aerodynamic force. No independent time-dependent target
circulation/force sequence or quadrature source for a coupled structural MMS
is currently implemented.
Similarly, `transfer_moving_interface.py` does not re-mesh/rebuild the
operators under moving topology and does not sweep RBF support or conditioning;
its conservation metrics cannot substitute for those accuracy tests.

Every result must be interpreted by its scope: `PASS` applies only to the
named check; `NOT_AVAILABLE`, `NOT_RUN`, `NOT_IMPLEMENTED`, and
`INCONCLUSIVE` are not passes. The full socket path, variable-population
event-aware coupled sources, coupled plots, and coupled spatial/temporal
refinement remain outstanding. Keep the overall status **PARTIAL** until
those production-path runs produce a consistent target, event/error records,
and defensible multi-level trends.

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
