# `mms_2`: velocity–vorticity MMS groundwork

`mms_2` is isolated under this directory. It contains the copied structural
tools, a symbolic continuum target, reconstruction utilities, and a separate
dispatcher. The analytical target and reconstruction helper do not yet form a
standalone fluid verification pipeline: `fluid` and `coupled` deliberately
exit with status 2 until the continuum source is projected consistently onto
the full rVPM particle state and applied at its RK3 stages.

## Implementation status

| Area | Status | What the status means |
|---|---|---|
| Analytical target | Implemented; current edits not run | Localized vector-potential target, total-velocity source, and symbolic identities are defined. |
| Standalone structural MMS | Inherited implementation available; not run here | Uses `mms_parameters.yaml` through the copied legacy driver. |
| Particle-discrete verification | Inherited discrete ODE study available; not run here | Verifies the inherited particle-state formulation, not a continuum field MMS. |
| Eulerian particle-field reconstruction | Helper functions implemented; not wired to a driver | Requires a valid complete-particle snapshot manifest and a caller-supplied FLOWUnsteady `ParticleField` and grids. No reconstruction result is claimed. |
| Standalone continuum fluid MMS | Unsupported; fail closed | Full-state source projection and its RK3-stage treatment are unresolved. |
| Coupled fluid–structure MMS | Unsupported; fail closed | Fluid source projection plus consistent interface force and coupled residuals are unresolved. |

No field or coupled convergence result is available. Do not interpret a
successful analytical preflight as a numerical solver verification.

## Analytical target

`field_mms.py` builds a spatially compact vector potential with an ellipsoidal
support. The potential is a `C4` bump with value and its first four normal
derivatives vanishing at the support boundary. SymPy constructs

```text
u_induced* = curl(A*)
u_total*   = U_inf + u_induced*
omega*     = curl(u_total*)
s_omega*   = d_t omega* + (u_total* . grad) omega*
                         - (omega* . grad) u_total*
```

The configured uniform freestream is evaluated at the same origin and time
convention as production `pfield.Uinf(pfield.t)`. `reconstruct_fields.jl`
requests `add_Uinf=true` once; its API supplies induced velocity and the
wrapper checks the explicit production freestream against the Python value.
The source uses total velocity. The compact induced field is divergence-free,
and `omega=curl(u_induced)` is its vorticity. In unbounded free space, the
decaying Biot–Savart inverse of this divergence-free induced velocity’s curl
recovers that induced velocity; uniform freestream is kept outside that
induced inversion. This is an analytical compatibility identity, not a
discrete reconstruction accuracy result.

The configured Eulerian box must strictly contain the target support. Runtime
reconstruction separately checks the complete particle/core envelope against
the requested box. Preflight’s `8×8×8` grid is for lightweight analytical
checks only; it is not used for final norms or convergence measurements.

## Reconstruction helpers and safeguards

`reconstruct_fields.jl` provides separate helpers to load a complete saved
particle population manifest, inspect particle counts and core sizes,
reconstruct `U` and `Wapprox`, audit fixed counts over supplied time-step
records, and compare a nominal grid with a refined grid on common nodes. The
manifest loader requires one dynamic population first and all required static
populations, rejects duplicate population names/files, and compares declared
counts against each HDF5 snapshot’s saved `np` metadata and the number actually
loaded. These helpers are not called by `mms2_driver.py`; complete-population
loading and fixed-count verification have therefore not been demonstrated for
a production field case.

The reconstruction wrapper uses the installed FLOWUnsteady particle field,
kernel, and `zeta_fmm` RBF vorticity field without changing particle strengths
or core sizes. It adds the actual `pfield.Uinf(pfield.t)` exactly once. Sampling
probes have zero circulation and must be removed after evaluation. Runtime
checks require at least 8 intervals per smallest particle-core scale and 20
intervals per smallest prescribed target length scale. The latter is the
localized target’s minimum support radius, used as a conservative feature
scale; it is not a periodic wavelength. The actual spacings, particle-core
minimum, and target scale are returned in reconstruction metadata.

The grid sensitivity helper uses the configured refinement factor and
relative tolerance (currently 2 and 1%). It compares velocity and vorticity
on shared nodes, requires identical domains and nested grid coordinates, and
fails if either field exceeds the threshold. It has not been run on a particle
field. The helper must be supplied Python coordinate samples for the actual
nominal grid; coordinate values, flattening, and a known polynomial vector
probe are checked in Julia before field errors are calculated.

## Metrics and refinement ownership

`field_mms.py` retains tensor-product Cartesian trapezoid weights and weighted
vector/component L2 errors. Inputs, errors, scales, and norms must be finite;
target normalization must be nonzero. Logarithmic order calculations require
positive errors and strictly decreasing scales, explicit common domain and
field-quantity metadata, and report each adjacent pair. Increasing errors are
flagged as non-monotonic; small changes are flagged as plateaus. These
utilities have not been applied to numerical field data.

- `mms_parameters.yaml` controls inherited standalone structural and legacy
  particle-discrete settings.
- `field_mms_parameters.yaml` controls the analytical target, reconstruction
  box/resolution/sensitivity settings, and field refinement sequence.
- Field refinement levels are not executed by the structural mode. There are
  no active coupled refinement settings while coupled MMS is unsupported.

The fixed-population control requirements in `field_mms_parameters.yaml`
record the intended starting-wake, unsteady-shedding, boundary-layer-shedding,
and pruning settings. The mms_2 driver does not override production controls;
the report marks lifecycle controls unaudited. A future runnable field case
must provide per-step particle counts and fail on any count change.

## Commands

From the repository root:

```bash
# Analytical and configuration preflight only; writes an ignored result file
python3 verification/mms_2/mms2_driver.py preflight

# Inherited structural run, or --dry-run to skip its solve
verification/mms_2/run_pipeline.sh structural --study spatial

# These preflight and then exit nonzero while source projection is unresolved
verification/mms_2/run_pipeline.sh fluid
verification/mms_2/run_pipeline.sh coupled

# Inherited particle-discrete ODE verification; this is not continuum field MMS
verification/mms_2/run_pipeline.sh particle-discrete
```

Generated results, snapshots, plots, and caches are ignored by the local
`.gitignore`. No tests or simulations were run while applying the latest
corrections.
