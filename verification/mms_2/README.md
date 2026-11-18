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
| Analytical target | Implemented; independent numerical tests pending | Localized vector-potential target, explicit uniform-freestream checks, and reference-time consistency are defined. |
| Standalone structural MMS | Inherited implementation available; not run here | Uses `mms_parameters.yaml` through the copied legacy driver. |
| Particle-discrete verification | Inherited discrete ODE study available; not run here | Verifies the inherited particle-state formulation, not a continuum field MMS. |
| Eulerian particle-field reconstruction | Executable snapshot entry point implemented | `reconstruct-snapshot` loads every declared particle population, derives nominal and doubled grids, checks cross-language coordinates and freestream, reconstructs `U`/`Wapprox`, and writes HDF5 plus JSON audits. A production reconstruction has not been run. |
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
reconstruction separately checks a declared 4-sigma Gaussian truncation
envelope against the requested box; this is a clearance guard, not a finite
kernel support boundary. Preflight’s `8×8×8` grid is for lightweight analytical
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
checks require at least 8 intervals per smallest FLOWVPM `sigma` Gaussian
radial scale and 20
intervals per smallest prescribed target length scale. The latter is the
localized target’s minimum support radius, used as a conservative feature
scale; it is not a periodic wavelength. The actual spacings, particle-core
minimum, and target scale are returned in reconstruction metadata.

The grid sensitivity helper uses the configured refinement factor and
relative tolerance (currently 2 and 1%). It compares total velocity, induced
velocity, and `Wapprox` on shared nodes with tensor-product trapezoidal
weights. The primary induced-velocity and vorticity metrics normalize by the
fine-field norm and fail below a configured norm floor. It requires identical
domains and nested coordinates. The executable generates Python coordinates
for each actual grid; Julia checks coordinates, flattening, dimensions, and a
known polynomial probe before reconstruction.

### Snapshot reconstruction command

The manifest must be JSON with `snapshot`, `physical_time_s`,
`run_metadata_path`, `particle_directory`, `required_population_names`, and
`populations`. For the current production metadata schema, the complete
inventory is `dynamic` followed by `static_or_bound`. Each population entry
must include `name`, `role`, `file_prefix`, `read_path`, `required`,
`expected_count`, and `saved_metadata_count`. The loader checks these counts
against each HDF5 `np` value and the actual `vpm.read!` result; saved `t`
values must agree across populations and with the required manifest time. The
output includes a validated manifest copy with the actual loaded count added
to each population entry.

```json
{
  "snapshot": 200,
  "physical_time_s": 0.05,
  "run_metadata_path": "/path/to/fluid/_run_metadata/run-id/run_metadata.json",
  "particle_directory": "/path/to/fluid/",
  "required_population_names": ["dynamic", "static_or_bound"],
  "populations": [
    {"name":"dynamic", "role":"dynamic", "file_prefix":"run_pfield",
     "read_path":"/path/to/fluid/", "required":true,
     "expected_count":1000, "saved_metadata_count":1000},
    {"name":"static_or_bound", "role":"static", "file_prefix":"run_staticpfield",
     "read_path":"/path/to/fluid/", "required":true,
     "expected_count":200, "saved_metadata_count":200}
  ]
}
```

```bash
verification/mms_2/run_pipeline.sh reconstruct-snapshot \
  --manifest /path/to/particle_population_manifest.json \
  --snapshot 200 \
  --grid-config verification/mms_2/field_mms_parameters.yaml \
  --output verification/mms_2/results/reconstruction_case
```

The command writes `reconstruction/snapshot_metadata.json`,
`particle_population_audit.json`, `grid_metadata.json`, and nominal/doubled
HDF5 files for total velocity, induced velocity, and `Wapprox`. A single saved
snapshot cannot establish stable identity or absence of creation/removal and
role-change events; its audit reports that limitation explicitly. Do not
interpret the snapshot count as a time-history identity audit.
The derived grid is never silently coarsened; the configured maximum point
count is a memory safety limit and exceeding it stops reconstruction.

## Metrics and refinement ownership

The structural solver stores `alpha_m` and `alpha_f` as old-state weights:
`x_stage = alpha*x_n + (1-alpha)*x_(n+1)`. Its second-order relation is
therefore `gamma = 1/2 + alpha_m - alpha_f`; this is equivalent to the usual
relation after converting from new-state weights. The selected `rho_inf=0.9`
weights are recorded in `config/solid_params.yaml`. The scalar check and plate
study use the same coefficients and evaluate the manufactured load at
`t_n + (1-alpha_f)*dt`, the physical time of the force stage under this
weighting convention.

The standalone structural spatial study uses `dt=1e-4`; the temporal study
uses the fixed `48x64` mesh and a common `0.2 s` endpoint. The structural
Newton absolute and relative tolerances are `1e-12` for MMS runs.

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

`fluid_mms_config.yaml` is a copy of the production fluid configuration with
fixed-population overrides. The production CLI accepts an independent
`--fluid` config. `particles_per_step`, `start_with_wake`, `unsteady`,
`wake_coupled`, and `shed_boundarylayer` map through `FluidConfig.jl` into
the setup controls. This FLOWUnsteady version has no pruning-off switch; the
isolated config sets extreme finite thresholds that make the sphere, strength,
and sigma pruning predicates no-ops for finite MMS states. This must be
rechecked if state scales change. The standalone snapshot command does not
retroactively prove those controls were used. Production per-step identity and
lifecycle events are not written by the current solver, so full identity
preservation remains blocked without a solver-side event log or persistent IDs.

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

# Complete-particle Eulerian reconstruction from a saved snapshot manifest
verification/mms_2/run_pipeline.sh reconstruct-snapshot --manifest /path/to/manifest.json --snapshot 200 --output verification/mms_2/results/reconstruction_case
```

Generated results, snapshots, plots, and caches are ignored by the local
`.gitignore`. No tests or simulations were run while applying the latest
corrections.
