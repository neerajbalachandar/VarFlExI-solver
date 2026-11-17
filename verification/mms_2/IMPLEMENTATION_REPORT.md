# `mms_2` implementation and correction report

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
   compact ellipsoidal `C4` bump. Symbolic checks cover divergence, curl, and
   the manufactured residual. The induced target is compact and is described
   as analytically compatible with the decaying free-space Biot–Savart
   inverse. No numerical Biot–Savart accuracy is claimed or measured.
4. **Particle populations:** The Julia loader requires an explicit manifest,
   one dynamic snapshot first, required static populations, unique names and
   files, and matching manifest/HDF5 `np` counts. It checks counts after each
   load and reports dynamic, static/bound, and total counts. These helpers are
   not connected to a driver, so a production snapshot has not been audited.
5. **Fixed population:** Added per-time-step count assertions and a history
   audit helper. Configuration records required no-starting-wake,
   no-unsteady-shedding, no-boundary-layer-shedding, and no-pruning controls.
   The driver does not override or verify production lifecycle controls and
   does not claim a fixed population; the field run remains disabled.
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
12. **Checks requested:** No tests, preflight, or simulation were run in this
    pass, following the user's instruction to leave tests to them. `git diff
    --check -- verification/mms_2` completed with no whitespace errors; the
    final status/diff was inspected manually. The
    original MMS and shared production source directories remain untouched in
    this task's diff.

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

The reconstruction and population-audit routines remain helpers rather than
an executable field pipeline. The source projection, RK3 integration,
production population lifecycle wiring, and coupled interface treatment must
be completed before numerical field or coupled verification can be enabled.
No commands are provided as successful numerical verification commands.
After the source projection and wiring exist, run the intended analytical
preflight and fluid/coupled refinement commands from the README, inspect the
population and sensitivity records, and report only measured errors/orders.
