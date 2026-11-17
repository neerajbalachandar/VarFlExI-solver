# `mms_2` implementation report

## Scope delivered

Created an isolated `verification/mms_2/` copy of the existing MMS files and
added an analytically derived velocity–vorticity continuum target, source,
field norm/error utilities, a full-particle-field FLOWUnsteady reconstruction
wrapper, a separate configuration, and an isolated dispatcher. The existing
`verification/mms/` tree was not edited as part of this work. Existing
uncommitted changes in the original MMS directory predated this task.

The standalone structural solver remains available. The field target and
reconstruction pieces are implemented, but standalone continuum fluid and
coupled runs are deliberately disabled until their particle-source projection
is mathematically derived and connected to the production RK3 stages. The
legacy discrete particle study remains available under the explicit name
`particle-discrete`; it is not presented as a field MMS.

## Files copied from `verification/mms/`

- `README.md` (replaced with new formulation documentation)
- `__init__.py`
- `check_coupled_jacobian.jl`
- `julia_with_openssl.sh`
- `mms_driver.py`
- `mms_parameters.yaml`
- `mms_problem.py`
- `mms_rvpm.jl`
- `run_pipeline.sh` (rewritten as an `mms_2` dispatcher)

No generated results, caches, or old output directories were copied. The
copied implementation's defaults and script references were redirected to
`verification/mms_2/`.

## New files

- `field_mms.py`: SymPy target/source construction, finite/nonzero checks,
  Cartesian quadrature, full vector/component error metrics, convergence
  orders, mesh-size calculation, and core/target scale grid planning.
- `field_mms_parameters.yaml`: standalone field settings and separate
  structural/VLM/time/refinement and reconstruction-domain settings.
- `reconstruct_fields.jl`: reconstruction wrapper for the actual production
  `ParticleField` and a supplied 3-D Cartesian grid.
- `mms2_driver.py`: isolated `preflight`, `structural`, `fluid`, `coupled`, and
  `particle-discrete` modes.
- `IMPLEMENTATION_REPORT.md`: this report.
- `results/fluid/analytical_preflight.json`: generated lightweight preflight
  record.

## Analytical formulation

`field_mms.py` starts from a smooth vector potential and derives
`u*=curl(A*)`, `omega*=curl(u*)`, and
`s_omega=d_t omega*+(u*·grad)omega*-(omega*·grad)u*`. SymPy confirms
`div(u*)=0` exactly. At `t_final=0.05 s` with the configured target, sampled
weighted norms are `||u*||_L2=1.5594e-1`,
`||omega*||_L2=1.4307`, and `||s_omega||_L2=1.5224e1` in the configured domain;
velocity and vorticity references are nonzero. These are target checks only,
not numerical solver errors or convergence results.

The field metrics calculate absolute and relative weighted L2 errors, all
three component errors, target and numerical norms, and maximum absolute
error. Invalid or near-zero target normalization and nonfinite data are
rejected. Orders are returned only for decreasing errors and strictly refined
spacings.

## Production model inspection

The installed project manifest resolves FLOWUnsteady 3.4.0, FLOWVLM 2.1.4,
and FLOWVPM 4.0.3. The repository config currently says
`viscous_scheme: none` and `vpm_sfs: none`.

- `fluid/src/FluidConfig.jl` maps `none` to the effective scheme name `None`;
  the setup maps that to FLOWVPM `Inviscid()`. FLOWVPM's
  `viscousdiffusion(..., ::Inviscid, ...)` is a no-op. The RK3 integrator does
  invoke viscous diffusion, so the effective type is checked by the
  reconstruction wrapper.
- `fluid/src/vpm_config.jl` maps `vpm_sfs: none` to `vpm.SFS_none`. FLOWVPM
  represents that as `NoSFS`; `isSFSenabled(::NoSFS)` is false and its before
  and after UJ callbacks are no-ops. An `Estr` SFS implementation is present
  and can be active for other models. The checked FLOWVPM 4.0.3 sources mention
  `Eadv` in time-integration comments but do not define an `Eadv` model in the
  searched source tree.
- Python preflight rejects non-disabled config labels; the Julia wrapper also
  rejects effective runtime objects other than `Inviscid` and `NoSFS`.

## Complete-particle reconstruction

`reconstruct_complete_field!` accepts the actual complete production
`ParticleField`, counts dynamic and static entries, and uses the installed
`FLOWUnsteady.computefluiddomain` overload. Existing free, static, and bound
particles in that field participate once; the isolated aerodynamic target
field is never appended. It uses FLOWVPM's installed kernel and `zeta_fmm`
RBF field for vorticity, keeps `scale_sigma=1`, leaves circulation unchanged,
and gives sampling nodes zero circulation. The helper adds
`pfield.Uinf(pfield.t)` once because the installed fluid-domain routine omits
freestream by default.

The box is common and fixed by configuration. At each call the helper checks
particle position/core support (`4*sigma_max`), at least `2*sigma_max`
clearance, grid resolution against the observed minimum core and target
wavelength, particle/grid probe capacity, and that temporary probes are
removed. It returns particle counts, core-size statistics, bounds, spacings,
freestream, effective model types, and kernel metadata. The required Eulerian
cell counts are derived from actual minimum core size and target wavelength;
they are not guessed from the FEM, VLM, or particle mesh. The configured
doubled-grid sensitivity factor is intended for a representative case.

## Why continuum source injection is outstanding

The continuum equation yields a vector field source, while production rVPM
advances particle positions, circulations, and core sizes through native
stretching, RK3 accumulators, and potentially viscous/SFS maps. Sampling
`s_omega` at particle locations and adding it to circulation rates is not a
derived projection and generally omits basis motion and core evolution. A
valid projection needs to determine full particle-state rates (or a
mathematically equivalent constrained residual), apply them at the actual RK3
stage times, and report the resulting reconstruction residual independently.
That derivation and runtime wiring are not present in the repository. The
user's brief explicitly forbids claiming that circulation-only forcing is
automatically consistent, so `fluid` and `coupled` preflight then exit with a
clear status rather than run the inherited discrete source under a new label.

The coupled case also needs the field-consistent interface force to be passed
to the structural manufactured residual. Existing discrete target force
machinery does not satisfy that condition automatically.

## Additional decisions (“out of the box” choices)

1. Used FLOWUnsteady's `Wapprox` / FLOWVPM `zeta_fmm` field for primary
   vorticity comparison, not a finite-difference curl of sampled velocity.
   This follows the installed particle RBF representation and avoids
   introducing grid-differencing error into the primary vorticity metric.
2. Set grid-probe circulation to zero. The installed utility otherwise gives
   probes a small nonzero default strength; those probes could contribute a
   sampling-grid-dependent field back into the production reconstruction.
3. Added the actual runtime freestream explicitly once and preserved all
   particle core radii/strengths. This avoids hidden default omissions and
   visualization-oriented filtering.
4. Derived Eulerian cell counts from the smallest observed particle core and
   target length scale rather than copying FEM/VLM refinement numbers into a
   grid. At high refinement this may make the requested 3-D calculation
   expensive; the wrapper rejects under-resolution instead of silently
   coarsening the field.
5. Made continuum `fluid` and `coupled` modes fail closed until the particle
   source projection exists. This is the principal scope limitation and
   preserves the distinction between target equation derivation and a valid
   production solver verification.

## Checks run

- `python3 -m py_compile` passed for the new and copied Python modules.
- `python3 verification/mms_2/mms2_driver.py preflight` passed symbolic
  divergence, finite/nonzero target norm, config, geometry, and refinement
  checks. The production geometry read was span `0.96 m`, root chord `0.24 m`;
  calculated structural `h_mesh` values were `0.015`, `0.0075`, and
  `0.00375 m`.
- `bash -n` passed for both copied shell scripts.
- Julia `Meta.parseall` passed for `reconstruct_fields.jl`.
- No structural, fluid, or coupled numerical simulation was launched.

## Still required before claiming full field MMS

Derive and independently audit the full particle-state projection and RK3
stage source; connect it to an `mms_2` runtime; establish preflight particle
envelopes and core-derived fixed grids; export and compare all full-domain
fields; write numerical error/convergence outputs; run grid sensitivity; and
use the same consistent fluid force for coupled structure. Until those steps
are done, the new continuum target and reconstruction helper are groundwork,
not a completed or verified solver MMS.
