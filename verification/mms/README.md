# Verification and convergence workflow

This directory contains three distinct kinds of evidence: individual solid
MMS convergence, individual FLOWUnsteady resolution sensitivity, and a
fixed-population coupled fluid-structure MMS. The coupled case verifies the
implemented discrete coupling/source path; it is not a continuum
Navier–Stokes MMS and it does not exercise particle birth, shedding, or
deletion. The standalone fluid study uses the finest completed case as a
numerical reference, so it is a sensitivity study rather than a formal exact
solution convergence proof.

## Files

| File | Role |
|---|---|
| `coupled_target.py` | Defines the analytic coupled target and serializes its shared manifest. Used by the coupled smoke and refinement drivers. |
| `coupled_solid_source.py` | Builds the structural manufactured source and target force functional for the coupled discrete model. |
| `run_coupled_mms.py` | Runs one source enabled production socket coupled case and writes target/numerical histories. A smoke run checks execution, not order. |
| `run_coupled_refinement.py` | Runs coupled temporal and spatial levels, compares every run with its analytic target, and writes error and observed order reports. This is the coupled convergence driver. |
| `checks_manifest_loading.py` | G0 end to end source check. Runs constant source and target coupled cases and checks active source stages and evolving particle state. It launches solvers. |
| `checks_manifest_loading.jl` | Parses the target manifest and production Julia source hooks. A source/manifest preflight only; does not run a simulation. |
| `checks_target_roundtrip.py` | Checks analytic target serialization and recovered values. |
| `checks_target_derivatives.py` | Checks analytic target derivatives against finite differences. |
| `checks_coupled_solid.py` | Checks the coupled structural manufactured residual/source identity with the production weak form. |
| `plate.py` | Defines the standalone plate MMS functions, derivatives, and symbolic manufactured load. |
| `fenics_plate.py` | Solves one standalone plate MMS case using the production Reissner–Mindlin model. It is called by `refinement.py`. |
| `refinement.py` | Plans or executes plate spatial and temporal MMS refinements; writes errors, observed rates, plots, and a run manifest. Add `--execute` to run it. |
| `jacobian_check.py` | Finite difference directional check of the production nonlinear plate residual Jacobian. This checks tangent consistency, not the PDE itself. |
| `strong_weak_check.py` | Compares the production plate weak form against its independent manufactured strong form. Individual structural formulation check. |
| `scalar_generalized_alpha.py` | Standalone scalar oscillator check of the generalized-alpha integrator. Kept as a time integrator component check; not part of the primary plate/coupled convergence sequence. |
| `run_refinement.jl` | Runs individual FLOWUnsteady parameter/refinement cases. `time` and `panels` vary time step and bound-panel resolution; other available studies are `particles`, `core`, `fmm`, and `wake`. |
| `julia_with_openssl.sh` | Launches Julia with the project’s OpenSSL_jll artifact ahead of system OpenSSL. Used by the pipeline and available for direct Julia commands. |
| `analyze.py` | Validates standalone fluid histories and compares QoIs against the finest completed numerical reference. Produces sensitivity reports and plots, not exact-solution errors. |
| `circulation_reference_check.jl` | Checks the individual FLOWVLM circulation solve against its linear-system residual. |
| `particle_position_stage_check.jl` | Checks particle position source evaluation across production RK3 stages. |
| `transfer_moving_interface.py` | Audits production CRM/RBF transfer on moving interfaces over selected mesh levels. Component evidence only; it does not establish conservation generally. |
| `progress.py` | Shared flushed progress/timing helpers for long Python solver/refinement cases. |
| `__init__.py` | Marks this directory as an importable Python package. |
| `run_pipeline.sh` | Runs the ordered verification stages below. It runs actual solver cases; it is not a dry run. |

`results/` contains generated reports and case histories, including earlier
runs. It is retained as evidence and is not source code. New pipeline outputs
go under `results/pipeline/`.

The obsolete `run_mms.py` aggregator was removed. It mixed old diagnostics
with an outdated partial-report format and was not used by the current
convergence drivers; its still useful component checks remain as separate
files above.

## Run order

Run from the repository root. The script automatically runs Python through
`conda run --name fenics-env` and uses the sibling `FLOWUnsteady/` directory
as Julia's project. It launches Julia through `julia_with_openssl.sh` to
avoid mixing the project’s OpenSSL 3.5 artifact with the system OpenSSL 3.0
library. Make sure `conda` and `julia` are on `PATH`. Override `FENICS_ENV`,
`FLOWUNSTEADY_PROJECT`, `CONDA_BIN`, `JULIA_BIN`, `JULIA_REAL_BIN`, or
`JULIA_OPENSSL_LIB` only if your installation uses different names or paths.

The complete sequence is:

```sh
./verification/mms/run_pipeline.sh all
```

It runs four phases in order. To resume at a phase, choose `preflight`,
`plate`, `fluid`, or `coupled` instead of `all`:

1. **Preflight and component checks.** Target roundtrip/derivatives,
   structural source identity, nonlinear plate Jacobian and strong/weak
   agreement, moving-interface transfer, particle RK3 stages, and VLM
   circulation residual. These isolate implementation issues before the
   expensive refinement studies.
2. **Individual solid MMS convergence.** Runs spatial then temporal plate
   refinement using the active production solid settings and independent
   analytic plate loads. It also runs the scalar generalized-alpha temporal
   convergence check, which isolates the integrator from FEM spatial error.
3. **Individual fluid resolution study.** Runs `time` and `panels` studies,
   then analyzes the histories. This checks sensitivity to time step and
   bound-panel resolution against the finest numerical case.
4. **Coupled verification.** Checks Julia manifest/source loading, runs a
   short coupled smoke, runs the G0 source-active production check, then runs
   coupled time and spatial refinement. The report compares each level to
   its analytic target.

The script stops on the first failed command. It does not install packages.
For example, to run only the coupled stage:

```sh
./verification/mms/run_pipeline.sh coupled
```

If you run a Julia verification file directly, use the same launcher so the
project's OpenSSL artifact is selected:

```sh
verification/mms/julia_with_openssl.sh --project=/home/dysco/FLOWUnsteady \
  verification/mms/circulation_reference_check.jl
```

If you want separate control, the same stages can be run individually:

```sh
./verification/mms/run_pipeline.sh preflight
./verification/mms/run_pipeline.sh plate
./verification/mms/run_pipeline.sh fluid
./verification/mms/run_pipeline.sh coupled
```

## Reading the results

### Standalone plate MMS

Find `plate_spatial_convergence.csv` and `plate_temporal_convergence.csv` in
their respective `results/pipeline/plate_*` directories, plus matching PNG
plots and manifests. Each error is measured against the analytic plate
solution. For a spatial order, use the listed mesh spacing
`max(hx, hy)`; for temporal order, use `dt`. A decreasing error sequence is
necessary. Treat the observed rate as credible only when at least three
levels decrease and adjacent rates settle. The driver marks short,
roundoff-level, nondecreasing, plateaued, or scattered sequences
`INCONCLUSIVE`; do not report those as an order.

### Individual fluid studies

`analyze.py` writes a `summary.json`, `metrics.csv`, and plots in each study
directory. `time` and `panels` include adjacent-level estimates for selected
QoIs, but all errors are differences from a finest numerical reference. Use
these to see whether outputs stabilize as `dt` decreases or panel count
increases. These results do not establish a formal order against an exact
fluid solution. A complete fluid history and consistent geometry/config are
required; failed cases are excluded and listed.

### Coupled MMS

The smoke output demonstrates that the coupled processes ran and produced
the required histories; it makes no convergence claim. The refinement
reports are `coupled_refinement_report.csv` and
`coupled_refinement_report.json` under `results/pipeline/coupled_refinement/`.
They contain analytic-target relative space-time L2 errors for structural
state, particle position/circulation/core, bound circulation, force, probe
velocity, and probe vorticity. Time studies refine `dt`; spatial studies
increase structural mesh and fluid span stations together. Adjacent observed
orders use the configured level ratios. Look for decreasing errors across
all levels and a reasonably consistent tail before stating an observed
order. An increasing/flat error or unstable slope means the series is
inconclusive and the dominant unrefined error source should be investigated.

Raw coupled per-time error fields are Euclidean norms. Relative space-time L2
is the space-time norm of numerical minus target divided by the target
space-time norm. The reference FLOWVLM target circulation residual is
`||A Γ_target - b||₂ / max(||b||₂, ε)`. Probe vorticity is the centered
finite-difference curl of reconstructed velocity at fixed points, so its
error also includes probe-spacing effects. The coupled model uses a fixed
particle population and does not verify particle lifecycle or general
CRM/RBF conservation.

## Direct driver commands

The runner above is the prescribed path. For manual use, these are the key
commands it invokes:

```sh
# Individual plate convergence
python verification/mms/refinement.py --study spatial --execute --output-dir verification/mms/results/plate_spatial
python verification/mms/refinement.py --study temporal --execute --output-dir verification/mms/results/plate_temporal

# Individual fluid sensitivity (run and analyze)
verification/mms/julia_with_openssl.sh --project=/home/dysco/FLOWUnsteady verification/mms/run_refinement.jl --study time,panels --execute --output verification/mms/results/fluid
python verification/mms/analyze.py --study time --results verification/mms/results/fluid
python verification/mms/analyze.py --study panels --results verification/mms/results/fluid

# Coupled smoke, then analytic-target refinement
FLOWUNSTEADY_PROJECT=/home/dysco/FLOWUnsteady JULIA_BIN=verification/mms/julia_with_openssl.sh python verification/mms/run_coupled_mms.py --output verification/mms/results/coupled_smoke --steps 3 --dt 0.01 --n-span 16 --nx 8 --ny 16
FLOWUNSTEADY_PROJECT=/home/dysco/FLOWUnsteady JULIA_BIN=verification/mms/julia_with_openssl.sh python verification/mms/run_coupled_refinement.py --study both --levels 1,2,4 --duration 0.03 --dt 0.01 --n-span 8 --nx 4 --ny 8 --output verification/mms/results/coupled_refinement
```
