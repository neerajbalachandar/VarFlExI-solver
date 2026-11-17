# `mms_2`: velocity–vorticity MMS

`mms_2` is an isolated copy of the previous MMS tools plus a new analytical
continuum velocity–vorticity target and FLOWUnsteady particle-field
reconstruction utilities. Its target source is derived independently of the
production solver. The structural study retains the copied Reissner–Mindlin
formulation.

## Current implementation boundary

The analytical target, continuum source, grid error metrics, preflight, and
full-particle-field reconstruction are implemented. The standalone structural
mode is runnable. **The continuum fluid and coupled solver modes fail closed**
until a defensible projection from the continuum forcing to the production
rVPM particle state and RK3 stage update is implemented. This is intentional:
the older `particle-discrete` mode verifies a fixed particle ODE and cannot be
reported as a full-domain field MMS. No field convergence result is claimed.

The field source must not be added to particle circulation by simply sampling
`s_omega` at particle locations. In general, particle positions, circulation,
and core radii all affect the reconstructed field. A valid source needs to
specify the particle-state projection, its residual in the Eulerian field
norm, and how it is applied at each production RK3 stage. The present tree
does not yet implement that derivation or connect a field-source callback to
the production fluid runtime. Consequently, `fluid` and `coupled` stop after
preflight rather than launch a different, misleading calculation.

## Continuum formulation

There is no pressure field. `field_mms.py` defines a smooth vector potential
`A*(x,t)`, then constructs

```text
u*      = curl(A*)
omega*  = curl(u*)
s_omega = d_t omega* + (u* . grad) omega* - (omega* . grad) u*
```

SymPy differentiates these expressions. The curl construction makes
`div(u*) = 0` identically. The first case is inviscid and has no SFS terms;
the source above is therefore the complete continuum vorticity residual.
Velocity/vorticity target norms are checked at the common final time and a
near-zero normalization is rejected.

## What the inherited implementation measured

- `mms_problem.py` constructs the five structural manufactured fields and
  independent structural loads.
- The inherited Julia particle study checks a fixed-population discrete rVPM
  ODE. It is not a continuum Navier–Stokes field MMS.
- The coupled particle target/source and aerodynamic target are independent
  discrete-model machinery. The target aerodynamic particle field is not the
  production fluid field and is not used by the new reconstruction.
- Earlier fluid diagnostics used a handful of fixed probes; these are not
  full-domain velocity/vorticity errors.
- The production particle field is assembled from dynamic particles and
  static particles. FLOWVPM's `ParticleField` carries both populations; the
  reconstruction wrapper counts both and passes that complete field once.
  Static bound-vorticity particles already present in the field are included
  exactly once. It does not append the isolated target field.

## Reconstruction and errors

`reconstruct_fields.jl` uses the installed
`FLOWUnsteady.computefluiddomain(pfield, grids; ...)` overload. The pinned local
environment is FLOWUnsteady 3.4.0 / FLOWVLM 2.1.4 / FLOWVPM 4.0.3. For this API:

- `U` and `W` are induced fields; the docs and installed source say they omit
  freestream. The wrapper adds `pfield.Uinf(pfield.t)` once to velocity.
- `Wapprox` is the installed `FLOWVPM.zeta_fmm` RBF vorticity reconstruction.
  This is the field compared to analytical `omega*`, rather than a finite
  difference of sampled velocity.
- The production kernel is kept; all particle cores use `scale_sigma=1.0`.
  No circulation clipping, core shrinkage, local particle filtering, or extra
  smoothing is applied. Grid probes have zero circulation.
- The wrapper checks complete dynamic/static counts, core-size statistics,
  source particle bounds plus four maximum core radii, and a minimum two-core
  boundary clearance. It requires preallocated capacity for all grid probes
  and verifies those probes are removed afterward.
- The installed Gaussian kernel has infinite support. The four-core-radius
  margin is an envelope safety check, not a particle cutoff; every particle
  remains in the field evaluation regardless of its distance from the grid.
- Use one fixed box across levels. `field_mms_parameters.yaml` records bounds,
  resolution criteria, and a doubled-grid sensitivity case. Grid cell counts
  are derived from the measured minimum particle core and target wavelength;
  the wrapper rejects grids that fail the three-interval core and
  eight-interval target criteria. The resolution plan is deliberately
  separate from FEM, VLM, and particle spacing.

`field_mms.py` supplies common Cartesian trapezoid weights and full-domain
absolute/relative L2, component L2, target/numerical norms, and maximum
absolute errors. It rejects zero/near-zero target norms and nonfinite values.
`observed_orders` omits an order unless the error decreases and the spacing is
strictly refined. Refinement configuration keeps structural elements, VLM
stations, reconstruction cells, and time step as separate values.

## Runtime model inspection

The production YAML options are parsed in `fluid/src/FluidConfig.jl`:
`viscous_scheme: none` maps to FLOWVPM `Inviscid()`, whose
`viscousdiffusion(pfield, ::Inviscid, dt)` is a no-op. `vpm_sfs: none` maps in
`fluid/src/vpm_config.jl` to FLOWVPM `SFS_none` (`NoSFS`); FLOWVPM defines
`isSFSenabled(::NoSFS) == false`, and its before/after UJ hooks do nothing.
The production RK3 does call viscous diffusion and gates the SFS calculation
on `isSFSenabled`, so the effective objects matter. `Estr` implementations
exist and can be active for non-`NoSFS` models. The FLOWVPM 4.0.3 source checked
here uses `Eadv` in time-integration comments but contains no `Eadv` model
implementation in its source tree. The reconstruction wrapper rejects any
runtime field whose effective viscous or SFS type is not `Inviscid`/`NoSFS`.
The repository `config/fluid_params.yaml` currently specifies both as `none`.

## Configuration and commands

All MMS-specific settings are local to `mms_2/field_mms_parameters.yaml`;
structural settings stay in the copied `mms_parameters.yaml`. Production
geometry and fluid settings are read from `config/`. Spatial defaults follow
the requested `(16,96,96)`, `(32,192,192)`, `(64,384,384)` structural/VLM
levels and fixed `dt=0.00025 s`. Temporal defaults use `0.001`, `0.0005`, and
`0.00025 s` at the fixed finest spatial setup and common `t_final=0.05 s`.
Each case's structural `h_mesh` is computed as
`max(Lx/nx, Ly/ny)` from the production geometry at runtime. Eulerian cell
counts are derived at runtime from the actual particle core distribution
rather than guessed before a production particle field exists.

From the repository root:

```bash
# Symbolic identities, nonzero target norms, refinement/config checks only
python3 verification/mms_2/mms2_driver.py preflight

# Retained structural solve (use --dry-run to skip the PDE solve)
verification/mms_2/run_pipeline.sh structural --study spatial

# These currently preflight, then stop because source projection is unresolved
verification/mms_2/run_pipeline.sh fluid
verification/mms_2/run_pipeline.sh coupled

# Explicitly run the inherited, discrete particle ODE study
verification/mms_2/run_pipeline.sh particle-discrete
```

The field preflight writes `results/fluid/analytical_preflight.json`. Structural,
particle, and coupled baseline outputs are routed under `verification/mms_2/`.
The reconstruction helper returns runtime metadata; production integration
must pass the actual complete `ParticleField` and a grid sized from its
smallest core before field error analysis can run.

## Remaining work before field MMS runs

1. Define and derive the full particle-state source projection and quantify
   its field residual independently.
2. Connect the projected source to the production RK3 stages without changing
   `verification/mms/` or silently modifying shared production runtime code.
3. Seed cases and prove all particle populations are resident exactly once;
   derive the common particle-envelope box and core-resolved grid from actual
   trajectory/core data.
4. Export full `U` and `Wapprox` fields at identical target nodes, calculate
   metrics, run doubled-grid sensitivity, and produce convergence tables.
5. Integrate the same fluid field errors and consistent interface force into
   coupled FSI, then inspect numerical convergence.

Do not call the method verified or convergent until those runs complete and
the errors/orders are reviewed.
