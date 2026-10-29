# Fixed population coupled FSI MMS

This case drives the production coupling server, Python structural solver, and
FLOWUnsteady Julia fluid solver. It uses one analytic time-dependent target
with a fixed four-particle population. FLOWVLM solves target circulation on
the target geometry and target wake; the configured production force
calculator forms the target loads and includes its unsteady history term.
The fluid source is evaluated at the native RK3 stages from a private shadow
field that runs the production particle update and core-spreading/remap order.
The structural source is assembled in the production weak-form loop against
the target load using an independent panel-force integration. Numerical
particle, circulation, force, geometry, and structural states remain solver
states; target states are used only for residual construction and comparison.

## Scope and limitations

- The case verifies one fixed-population, short-time coupled discrete model.
  It does not verify particle birth, shedding, deletion, or wake lifecycle.
- It is not a continuum Navier–Stokes MMS and does not establish CRM/RBF
  conservation independently.
- The target force and particle source use configured production operators,
  but discrete time integration and interface interpolation leave measurable
  error. A smoke run has no convergence-order claim.
- Probe vorticity is a centered finite-difference curl of reconstructed
  velocity at fixed points. It is a diagnostic with probe-spacing error.
- Refinement compares each run to its analytic target. The target particle
  coefficients and structural functions are the same across levels; target
  span stations are evaluated at each fluid/coupling mesh resolution.
- The code and documentation have received source-level inspection only in
  this change. No tests, Python processes, Julia processes, or solver runs
  have been executed here.

## Manual run order

Run commands from the repository root. Use a Python environment with the
legacy FEniCS/DOLFIN and project Python dependencies installed. Use the
FLOWUnsteady Julia project at `/home/dysco/FLOWUnsteady` (or set
`FLOWUNSTEADY_PROJECT` to that valid project directory); ensure `julia` is on
`PATH` in the Python process environment.

1. Check the analytic target serialization and derivatives:

   ```sh
   python verification/mms/checks_target_roundtrip.py
   python verification/mms/checks_target_derivatives.py
   ```

2. Check the structural manufactured residual helper in the legacy FEniCS
   environment:

   ```sh
   python verification/mms/checks_coupled_solid.py
   ```

3. Generate a target manifest, then inspect the actual Julia implementation
   and manifest in the FLOWUnsteady project:

   ```sh
   python -c 'from verification.mms.coupled_target import build_default_target, write_manifest; write_manifest(build_default_target("config", 0.03, 16), "verification/mms/results/manual/target.json")'
   julia --project=/home/dysco/FLOWUnsteady verification/mms/checks_manifest_loading.jl verification/mms/results/manual/target.json
   ```

4. Run the source-enabled production socket-coupled smoke case. Execute this
   from the legacy FEniCS Python environment with the FLOWUnsteady project
   available as shown above:

   ```sh
   FLOWUNSTEADY_PROJECT=/home/dysco/FLOWUnsteady python verification/mms/run_coupled_mms.py --output verification/mms/results/coupled_smoke --steps 3 --dt 0.01 --n-span 16 --nx 8 --ny 16
   ```

5. After the smoke case passes, run coupled time and spatial refinement
   comparisons (this launches all production solvers for every case):

   ```sh
   FLOWUNSTEADY_PROJECT=/home/dysco/FLOWUnsteady python verification/mms/run_coupled_refinement.py --study both --levels 1,2,4 --duration 0.04 --dt 0.01 --n-span 8 --nx 4 --ny 8 --output verification/mms/results/coupled_refinement
   ```

The commands above are instructions only; this source-level task did not run
them. Do not treat the helper check as evidence that the production coupling
path works. The smoke command is the first end-to-end check. The refinement
driver is the only command here that produces coupled convergence data.

## Outputs

The smoke output is under `verification/mms/results/coupled_smoke/`:

- `target.json`: shared analytic structural, geometry, particle, and
  freestream target manifest.
- `fluid/coupled_mms_force_payload_history.jsonl`: physical-time target and
  numerical circulation, forces, particles, geometry, fixed-probe velocity
  and vorticity, circulation residual, and RK stage source values/norms.
- `fluid/coupled_mms_mms_rk3_stage_history.csv`: position, circulation, and
  core source values by RK stage and target particle slot.
- `solid_mms_history.csv`: physical time, target/numerical target-force
  integration, target/numerical structural states, error norm, and source
  norm.
- `coupling/coupling/forces_sent_received_history.jsonl`: socket timestamps,
  fluid/solid stage timing, force and geometry transfer residuals.
- `mms_summary.json`: per-time error histories and explicit scope/error norm
  definitions.
- `process.log`, plus solver-specific fluid and solid output directories.

The refinement output is under the selected output directory, with one folder
per study/level containing each run's manifest and solver histories, and:

- `coupled_refinement_report.json`
- `coupled_refinement_report.csv`

The report gives relative space-time L2 errors against each run's analytic
target for structural state, particle position/circulation/core, bound
circulation, force, and probe velocity/vorticity. It includes descriptive
adjacent-level observed orders. It does not compare against the finest run.

Raw per-time `*_error_l2` quantities are Euclidean norms over the stated
panels, particles, or probe vector components. Relative space-time L2 is
defined as `sqrt(sum_k dt_k ||numerical_k-target_k||_2^2) /
max(sqrt(sum_k dt_k ||target_k||_2^2), 1e-30)`. Within each refinement series
the time spacing is uniform, so the common `dt` factor cancels in the ratio.
The independently rebuilt FLOWVLM target circulation residual is
`||A Γ_target - b||_2 / max(||b||_2, machine_epsilon)`.

## Expected pass and interpretation

A smoke run is complete only if all three production processes exit normally,
the expected number of physical-time fluid rows and solid rows are present,
and the emitted fields are finite. In particular, target and numerical
histories must coexist; there must be target force/circulation and no-
penetration residuals, target/numerical particle states, fixed probe velocity
and vorticity, solid target/numerical states, nonempty source-stage history,
and coupling transfer/timing rows. Numerical evolution is validated by
observing source callback records and changing numerical particle state over
physical time; a nonzero numerical-versus-target error is **not** a pass
requirement because exact agreement is allowed.

If the Julia hook inspection fails, resolve manifest or source parsing issues
before running solvers. If the Python structural helper fails, inspect the
weak-form source sign and target force integration. If the socket smoke fails,
use `process.log` and the partial histories to distinguish startup/config
problems, source-stage failures, target circulation/force failures, missing
rows, and transfer errors. Large target circulation residuals indicate the
target VLM solve or residual reconstruction is inconsistent. Missing RK
stages or unchanging numerical states indicate source hook/integrator issues.
Large structural/geometry/force errors indicate target mapping, sign, or
transfer discrepancies; they are reported rather than hidden by replacing
numerical states.

One successful smoke run demonstrates execution and comparison only. A
convergence claim requires complete time and spatial series, decreasing
analytic-target errors, and reported observed slopes with enough levels to
support the interpretation. If errors plateau or rise, report that result and
investigate which discrete operator, source residual, transfer, or solver
tolerance dominates; do not call it convergence.
