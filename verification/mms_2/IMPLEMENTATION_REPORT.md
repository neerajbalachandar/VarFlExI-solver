# `mms_2` implementation and correction report

## Latest review closure

This pass keeps changes isolated to `verification/mms_2/`. The shared source
and original MMS changes from earlier branch history were reverted in the
separate cleanup commit `2425ef59`; generated tracked MMS result artifacts
were removed there. A backup branch preserves the prior branch tip.

### Implemented and checked

- Analytical preflight filters reconstruction metadata out of `FieldTarget`
  construction, validates the configured minimum support radius, rejects
  spatially varying freestreams, and requires the explicit analytical time to
  match spatial and temporal study endpoints.
- Independent numerical tests cover divergence, curl, vorticity transport,
  support continuity, and free-space Biot–Savart quadrature with a recorded
  refinement sensitivity. These are continuum identity checks, not solver
  convergence tests.
- `reconstruct_snapshot.jl` and `reconstruct-snapshot` load complete dynamic
  and static populations, validate manifest/HDF5/loaded counts and times,
  check particle flags, derive nominal and doubled grids from actual `sigma`,
  compare actual-grid Python/Julia coordinates, check freestream equality,
  run full-particle `U`/`Wapprox` reconstruction, and write HDF5 fields and
  JSON metadata/audits.
- Sensitivity reports total velocity, induced velocity, and vorticity with
  nested-node tensor-product quadrature. Induced velocity and vorticity are
  the required primary metrics and normalize against the fine-field norm.
- `fluid_mms_config.yaml` is derived from production config and explicitly
  sets the requested fixed-population controls. The production CLI supports
  an independent fluid config. Since this solver has no pruning boolean, the
  config uses finite extreme thresholds verified against its strength,
  sigma, and sphere removal predicates to make them no-ops over finite MMS
  states.
- The pinned FLOWVPM check against saved dynamic/static fixtures passed:
  4 dynamic particles loaded with overwrite, then 24 static particles
  appended with `overwrite=false`, with static flags preserved and equal
  physical times.

### Implemented but not established by a production snapshot

- The standalone reconstruction entry point is wired, but has not yet been
  run against a production manifest. Grid capacity, clearance, exact runtime
  metadata layout, and HDF5 output need confirmation on the intended case.
- The isolated fluid config records no-wake/no-shedding and no-op pruning
  controls, but no time-history run has verified those lifecycle settings.
- One snapshot provides no persistent particle IDs or creation/removal/role
  change event log. The generated audit states this limitation; count and
  static-flag checks cannot prove identity continuity.

### Unsupported pending mathematical work

- Continuum fluid and coupled modes remain fail closed with exit status 2.
  The continuum-to-rVPM projection for the full `(X, Gamma, sigma)` particle
  state, initial representability, and stage-aware RK3 forcing remain
  unresolved. Coupled interface force/moment consistency is also unresolved.
  No fluid or coupled convergence claim is made.

### Checks run in this pass

- Python byte-compilation passed.
- Julia parsing passed for the reconstruction helper, executable, and
  append-check script.
- `vpm.read!` overwrite/append fixture check passed as described above.
- `git diff --check` passed.
- The analytical pytest suite passed (6 tests), and analytical preflight
  passed.
  No full fluid, structural, or coupled simulation was launched.

## Scope and repository boundary

The changes in this correction pass are confined to `verification/mms_2/`.
No file under `verification/mms/` or `fluid/src/` was edited. Existing changes
in the isolated field target and reconstruction helper were continued rather
than discarded. Generated `results/` output was removed and local ignore rules
were added.

## Corrections implemented

1. **Isolation and artifacts:** Added local ignore rules for results, caches,
   snapshots, and plot outputs. Removed the generated preflight JSON. Final
   Git status was inspected; all changed paths are in `verification/mms_2/`.
2. **Freestream consistency:** The target defines induced and total velocity
   separately and builds the vorticity source from total velocity. Python
   mirrors production `configured_freestream(Xdummy=[0,0,0], t)` including the
   row-major gradient matrix and gust phase convention. The reconstruction
   compares against `pfield.Uinf(pfield.t)` and requests the freestream once.
   A preflight equality record is generated when preflight is run.
3. **Free-space compatibility:** Replaced the periodic potential with a
   compact ellipsoidal `C4` bump. Curl and source relationships are recorded
   by construction; independent finite-difference tests check divergence,
   curl, and the manufactured residual. The induced target is analytically
   compatible with the decaying free-space Biot–Savart inverse, and independent
   ray quadrature checks that identity with separate quadrature sensitivity.
4. **Particle populations:** The Julia loader requires an explicit manifest,
   one dynamic snapshot first, required static populations, unique names and
   files, and matching manifest/HDF5 `np` counts. It checks counts after each
   load and reports dynamic, static/bound, and total counts. The new snapshot
   entry point wires these helpers; it has not been run against production
   output.
5. **Fixed population:** Added per-time-step count assertions, a history
   audit helper, and a fail-closed persistent-ID/role/event validator.
   Configuration records required no-starting-wake,
   no-unsteady-shedding, no-boundary-layer-shedding, and no-pruning controls.
   The isolated fluid config sets these controls. No production time history
   has been audited, and the field run remains disabled.
6. **Grid resolution and sensitivity:** Raised defaults to 8 intervals per
   minimum core scale and 20 per smallest target scale. Added a configured
   doubled-grid factor and 1% relative tolerance. The Julia helper compares
   velocity and vorticity on common nodes, validates nested coordinates and
   domain, reports actual spacings and fails above tolerance. It is not wired
   to a production run and has not produced sensitivity results.
7. **Metrics and convergence:** Retained weighted Cartesian L2 metrics and
   added finite-value checks. Pairwise order reporting rejects nonpositive
   errors, nonfinite or non-decreasing scales, and mismatched domains or field
   quantities. It reports adjacent-pair statuses and identifies
   non-monotonic errors and plateaus. No numerical orders are reported.
8. **Python/Julia ordering:** Python sample points are flattened with the
   first coordinate fastest, matching GeometricTools column-major indexing.
   Julia validates sampled coordinates and a known polynomial vector probe,
   including its three-component order, before a caller uses the reference.
   This check is a helper and has not yet been run against a numerical field.
9. **Fail-closed modes:** `fluid` and `coupled` still preflight and exit with
   status 2. The unresolved blocker is a mathematically justified projection
   of continuum forcing onto `(X_p, Gamma_p, sigma_p)` through production RK3
   stages. Coupled verification additionally lacks consistent interface
   forcing and residuals.
10. **Configuration ownership:** `mms_parameters.yaml` is identified as the
    inherited structural/particle-discrete configuration;
    `field_mms_parameters.yaml` owns analytical field and reconstruction
    settings. Field refinement levels are not presented as structural levels;
    dormant coupled refinement settings are absent.
11. **Documentation:** README and this report distinguish analytical target,
    structural MMS, particle-discrete verification, reconstruction helpers,
    continuum fluid MMS, and coupled MMS. They explicitly label the last two
    unsupported, identify helpers that are not driver-integrated, and make no
    convergence claim.
12. **Checks:** Bounded source checks and the independent analytical suite
    passed; no full solver simulations were run. The original MMS and shared
    production source directories remain untouched in this task's diff.

## Files changed in this correction pass

- `field_mms.py`
- `field_mms_parameters.yaml`
- `reconstruct_fields.jl`
- `mms2_driver.py`
- `README.md`
- `IMPLEMENTATION_REPORT.md`
- `.gitignore`

The isolated files copied from the prior MMS work remain in this directory;
they were not modified in this correction pass except where listed above.

## Remaining blockers and user-run commands

The reconstruction command establishes field recovery for a supplied complete
snapshot; it does not establish a time-history identity audit or solve the
continuum source-projection problem. Complete the source projection and
stage-aware RK3 treatment before enabling continuum fluid verification, then
complete interface forcing and residual checks before enabling coupled MMS.
