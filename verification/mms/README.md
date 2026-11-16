# MMS verification workflow

This directory contains the manufactured targets, source construction, driver,
and configuration for the structural, particle, interface-transfer, and
coupled verification studies. The configuration uses SI units.

## Scope and current measurement limits

- The standalone plate study constructs the five Reissner–Mindlin generalized
  fields `(u0, v0, w, theta_x, theta_y)` and their analytical loads. Its
  component errors are FE spatial relative L2 norms evaluated by quadrature.
- The particle study checks the discrete fixed-population rVPM ODE. It does not
  verify continuum Navier–Stokes equations or particle insertion, shedding,
  pruning, and deletion.
- The coupled particle source uses the analytic Gaussian-erf kernel Jacobian
  for FLOWVPM 4.0.3. The separate standalone particle study still uses its
  fixed-step finite-difference Jacobian and has not been brought under that
  analytic derivative audit.
- `MMSRK3StateSource` is a stage-aware adapter that mirrors the installed
  FLOWVPM low-storage RK3 recurrence while calling production state-update and
  viscous-diffusion operations. It is not an unmodified call to FLOWVPM's
  integrator. The adapter must be rechecked whenever the pinned dependency
  implementation changes. The coupled MMS requires `vpm_sfs=none` and
  `viscous_scheme=none`, because its independent particle source omits those
  effects; the driver rejects configurations that enable them.
- The coupled bound-circulation target is
  `Gamma0 + a*sin(omega*t) + b*cos(omega*t)`, with derivative
  `omega*(a*cos(omega*t) - b*sin(omega*t))`. The analytic derivative is used
  for the unsteady-circulation force. Python, the JSON manifest, and Julia use
  this convention; at `t=0` the state is `Gamma0+b`.
- The coupled target aerodynamic branch is evaluated on a separate particle
  field containing copied static particles and only the manifest's target
  dynamic particles. Its configured `bound_circulation` enters an isolated
  production VLM solve through a manufactured normal-velocity RHS correction.
  The coupled numerical wing solve is still separate and is measured against
  that target. The reference panel force is independently assembled from the
  three Kutta–Joukowski bound segments and the two unsteady-circulation panel
  triangles. The driver writes a per-case MMS fluid configuration that turns
  off `regularize_vlm` and boundary-layer shedding, records those overrides
  and their reasons in `case_metadata.json`, and leaves production YAML files
  untouched. Nonempty section polar settings and unsupported viscous/SFS terms
  are rejected. The generated coupling config sets `force_relax=1` so the
  applied load matches the manufactured reference.
- The structural history now records componentwise FE spatial L2 errors using
  domain integration. The coupled extractor additionally requires the active
  bound-circulation target, production VLM residual, geometry error, and
  separate force, moment, and work transfer metrics, including independently
  recomputed target-VLM circulation error and a hard algebraic-residual gate.
  The configured common moment origin is checked against transfer diagnostics
  with a documented CSV precision tolerance.
- The reference panel traction uses a normalized beta profile over each chord
  with mean at configured `eta_cp`, preserving the intended panel resultant
  and chordwise first moment under the affine swept-plate mapping.
- The coupled structural source is a fully discrete generalized-α MMS
  residual. It advances a separate target velocity/acceleration history with
  the production kinematic coefficients, assembles independently written
  weak operators at `q_(n+alpha_f)`, `v_(n+alpha_f)`, and `a_(n+alpha_m)`,
  then subtracts the target aerodynamic force at the recorded `t_(n+1)`
  exchange time. Endpoint FE errors remain against `q*(t_(n+1))`.

No numerical convergence values or pass claims are maintained in this README.
Generated CSV/JSON records are the source of measured results. The driver
rejects missing or non-finite quantities rather than filling result columns
with `NaN` or zero.

## Targets and norms

The structural target is

`q*(x,y,t) = (u0*, v0*, w*, theta_x*, theta_y*)`,

with a smooth spatial envelope and configured temporal modes. The structural
outputs `e_u0`, `e_v0`, `e_w`, `e_theta_x`, and `e_theta_y` are componentwise
relative FE spatial L2 errors. They are evaluated at final time for each
standalone refinement case.

Particle state targets retain fixed particle identity and ordering. The
component errors `e_pos`, `e_circ`, and `e_core` compare matched target and
numerical arrays at the same time. Reconstructed `e_velocity` and
`e_vorticity` compare target and numerical values at the fixed physical probes
recorded in the fluid history. A particle-count refinement is a population
and reconstruction study; source magnitude versus particle count alone is a
source-scaling diagnostic, not solution convergence.

Interface metrics are separate: `e_transfer_force`, `e_transfer_moment`, and
`e_transfer_work`. Moment
comparisons must use the same declared origin and force sign convention; work
comparisons must use the implemented weighted displacement/force inner
products. Force conservation alone does not establish moment conservation or
work conjugacy.

## Configuration

`mms_parameters.yaml` is the authority for MMS amplitudes, frequencies,
refinement cases, solver tolerances, and output location. Coupled refinement
levels are configured under `coupled` (`spatial_levels` and
`temporal_levels`). Structural standalone spatial and temporal cases are
listed under `standalone_plate`.

The particle-level source convergence is a separate discrete-model study; it
does not request coupled-result extraction without coupled histories. Relative
particle, probe, force, geometry, circulation, and transfer errors are
normalized by their matched target/resultant norms and reject zero target
norms. Coupled convergence output labels the final-time norm, reports `dt` or
mesh size, particle count, final time, and observed orders; invalid orders are
left blank in CSV and null in JSON.

## Running

The author can run the selected case after configuring the required solver
environments:

```bash
python verification/mms/mms_driver.py structural --study spatial
python verification/mms/mms_driver.py structural --study temporal
python verification/mms/mms_driver.py transfer
python verification/mms/mms_driver.py fluid
python verification/mms/mms_driver.py coupled --study spatial
python verification/mms/mms_driver.py all
```

The automated consistency command writes per-check states. `PASS` means a
check was evaluated and met its criterion. `FAIL` and `ERROR` block overall
success. The suite checks the assembled production structural Jacobian by
centered directional differences, verifies the five-field strong-to-weak
identities including boundary fluxes, audits target derivatives and the
resolved FLOWVPM RK3 source, and exercises CRM/RBF force, moment, origin-shift,
and virtual-work diagnostics. Thresholds are recorded with each result and in
`mms_parameters.yaml` where they are configurable.

`python verification/mms/mms_driver.py all` and
`verification/mms/run_pipeline.sh all` run the same sequence: consistency
checks, standalone structural study, transfer study, particle-level Julia
study, and coupled FSI study.

Each coupled case records the resolved FLOWUnsteady, FLOWVLM, and FLOWVPM
versions, repository revisions and tree hashes from `Manifest.toml`, along
with the active DOLFIN/Python versions, in `case_metadata.json`. The automated
source audit also checks that runtime-resolved FLOWVPM matches the manifest
and inspects the actual installed RK3 implementation.
