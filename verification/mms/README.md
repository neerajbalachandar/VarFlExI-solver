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
- `MMSRK3StateSource` is a stage-aware adapter that mirrors the installed
  FLOWVPM low-storage RK3 recurrence while calling production state-update and
  viscous-diffusion operations. It is not an unmodified call to FLOWVPM's
  integrator. The adapter must be rechecked whenever the pinned dependency
  implementation changes. The coupled MMS requires `vpm_sfs=none` and
  `viscous_scheme=none`, because its independent particle source omits those
  effects; the driver rejects configurations that enable them.
- The coupled runtime computes a target-state aerodynamic/VLM branch and
  records its algebraic circulation residual. This is a consistency
  diagnostic, not VLM MMS solution verification. `bound_circulation` in the
  manifest is a prescribed, currently inactive target and must not be
  interpreted as the circulation used to construct coupled target forces.
- The structural history currently contains coefficient arrays and scalar
  metadata, but not componentwise FE spatial norms. The coupled driver refuses
  to label a coefficient-vector norm as an FE L2 error. It also requires
  recorded common-origin moment and virtual-work metrics before writing a
  coupled convergence row. Those producer fields are not yet available, so a
  coupled run cannot produce a defensible convergence table until the history
  schema is extended.

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

Interface metrics are separate: `e_force`, `e_moment`, and `e_work`. Moment
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

The driver reads interface transfer levels, evaluation time, quadrature order,
particle time steps, and particle population levels from the manifest. Relative
particle and probe errors use the matched analytical target norm as their
normalization and reject a zero target. Explicit configured dimensional scales
for force, moment, and work are still required before those coupled metrics can
be recorded.

## Running

The author can run the selected case after configuring the required solver
environments:

```bash
python verification/mms/mms_driver.py structural --study spatial
python verification/mms/mms_driver.py structural --study temporal
python verification/mms/mms_driver.py transfer
python verification/mms/mms_driver.py coupled --study spatial
```

The automated consistency command writes per-check states. `PASS` means a
check was evaluated and met its criterion. `FAIL` and `ERROR` block overall
success; required checks that are not yet implemented are explicitly marked
`NOT RUN` and also block overall success. The current suite does not establish
the production structural Jacobian directional derivative, full five-field
strong-to-weak identity, or complete independent transfer diagnostics.

The stage-aware RK3 code is pinned to the FLOWVPM/FLOWUnsteady versions
resolved by the project environment. Record the resolved dependency commits
and the FEniCS version in generated verification metadata when the workflow is
run; this directory does not infer version identifiers from source comments.
