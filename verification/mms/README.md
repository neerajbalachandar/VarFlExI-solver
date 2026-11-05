# Verification and convergence

The verification studies in this directory examine the structural solver, the FLOWUnsteady fluid solver, and their coupled implementation. They are organised separately because each addresses a different numerical question.

The structural study uses the Method of Manufactured Solutions (MMS) to measure the error in the production Reissner–Mindlin plate solver against a prescribed analytical solution. The standalone fluid study examines how FLOWUnsteady results change with time-step size and bound-panel resolution. Since there is no analytical solution for these fluid cases, the results are compared against the finest completed numerical run. The coupled study introduces an analytical target and corresponding manufactured source terms into the production fluid–structure interaction (FSI) implementation. It checks whether the structural and fluid quantities evolve consistently with the prescribed target when the two solvers communicate through the actual coupling interface.

These studies should not be interpreted as interchangeable convergence tests. The plate and coupled studies measure errors against analytical targets, whereas the standalone fluid study measures differences between numerical solutions at different resolutions. The coupled MMS is specifically intended to verify the implemented discrete coupling and source path. It is not a manufactured solution of the continuum Navier–Stokes equations, and it does not verify particle birth, shedding, or deletion. Similarly, the fluid resolution study provides evidence of numerical sensitivity and stabilization, rather than a formal convergence proof against an exact fluid solution.

## 1. Verification studies

### 1.1 Structural solver — standalone plate MMS

The standalone structural MMS checks the spatial discretization and time integration of the production Reissner–Mindlin plate model implemented in FEniCS.

A smooth analytical displacement and rotation field is prescribed over the plate. Its time derivatives and the corresponding manufactured external load are evaluated analytically. The solver is then given this load and asked to recover the prescribed solution. Since the analytical fields are known, the numerical displacement and rotation fields can be compared directly against them.

The advantage of this approach is that the expected solution is available independently of the numerical solve. Consequently, the difference between the computed and analytical fields provides a direct measure of the discretization error for the manufactured problem.

Two refinements are performed separately.

**Spatial refinement** increases the number of finite-element cells in both in-plane directions while keeping the time step fixed. This measures how the solution changes as the structural mesh is refined, with temporal discretization error intended to remain smaller than the spatial error.

**Temporal refinement** reduces the time step while keeping the structural mesh and physical simulation duration fixed. This isolates the time-integration error associated with the generalized-α scheme.

The structural refinement driver uses the production Reissner–Mindlin model and the active material and integration settings in `config/solid_params.yaml`. The manufactured load is obtained independently from the analytical solution rather than from the residual of the numerical solution being tested.

A separate scalar oscillator test is also included. It applies generalized-α integration to a problem with a known analytical solution, allowing the time integrator to be examined without finite-element spatial discretization entering the error.

The purpose of these tests is to establish that the structural formulation, its assembled weak form, and its time integration reproduce the expected behaviour on problems for which the exact solution is known. They do not, by themselves, establish the accuracy of the fluid–structure coupling.

### 1.2 Fluid solver — FLOWUnsteady resolution studies

The standalone fluid studies examine the numerical behaviour of FLOWUnsteady using the production reformulated Vortex Particle Method (rVPM), FLOWVLM bound-vortex representation, and the associated numerical settings.

Unlike the structural MMS, these studies do not prescribe an analytical continuum solution. Instead, multiple simulations are performed with different numerical resolutions, and their aerodynamic quantities and particle histories are compared against a selected numerical reference.

For each study, the finest completed case is used as the reference. For example, the smallest time step is used as the reference in the time study, while the highest bound-panel resolution is used in the panel study. The resulting differences indicate whether the quantities of interest are approaching a stable numerical solution as the selected resolution is increased.

The two studies included in the default pipeline are as follows.

**Time-step study.** The time step is successively halved. The number of particles released per step is adjusted from 4 to 2 to 1 so that the nominal particle release spacing remains approximately constant. This avoids changing the physical particle release spacing simply because the time step has changed. The comparison therefore examines time-step sensitivity while controlling the particle release scale.

**Bound-panel resolution study.** The number of spanwise bound panels is increased from 40 to 80 to 160. This examines the sensitivity of the aerodynamic solution to the discretization of the bound-vortex representation.

The driver also supports four additional studies. The particle study varies the number of particles released per time step, the core study varies the VPM regularization parameter `lambda_vpm`, the FMM study changes the fast multipole method evaluation tolerances, and the wake study changes the criteria used to retain or remove particles. These studies can be executed separately when investigating the influence of a particular numerical setting.

The quantities examined include lift and drag coefficients, bound circulation distributions, probe velocities, probe vorticity, particle counts, and computational runtime. The analysis script aligns the histories in time and compares the circulation distributions on a common spanwise coordinate before computing the integrated differences.

These comparisons are useful for determining whether a numerical quantity is sensitive to a particular discretization or solver parameter. However, a small difference from the finest computed case is not necessarily a small error relative to the physical solution. The reference itself is a numerical approximation, and its accuracy has not been established through a continuum MMS.

### 1.3 Coupled solver — fixed-population discrete-model MMS

The coupled MMS checks the structural and fluid components while they communicate through the production socket-based coupling implementation.

The test begins with an analytical target describing the structural displacement and rotation fields, particle positions and circulation, particle core parameters, bound circulation, aerodynamic forces, and selected flow quantities. This target is serialized into a shared manifest so that the Python structural solver and Julia fluid solver use the same prescribed data.

The structural manufactured source is assembled using the production Reissner–Mindlin weak form and the generalized-α time-integration maps. In essence, the analytical structural target is evaluated in the discrete structural equations, and the corresponding manufactured source is constructed by accounting for the target aerodynamic force. This allows the numerical structural solution to be checked against the prescribed target without simply assigning the target to the numerical solution.

On the fluid side, the target particle states, bound circulation, aerodynamic forces, and probe quantities are evaluated alongside the production solver. The source and target histories are checked across the production Runge–Kutta stages and the actual socket-coupling chronology.

This distinction matters because a successful run alone does not establish that the manufactured source was applied correctly. The source must be active at the intended stages, the target and numerical histories must be recorded, and the numerical particle state must actually evolve. The coupled checks therefore examine these conditions before the refinement results are considered.

The coupled verification has three parts.

**Target and source checks** verify that the analytical target can be serialized, recovered, differentiated, and used to construct the structural manufactured source. The Julia-side check also verifies that the target manifest and the relevant production source hooks can be loaded.

**Production coupling checks** execute short socket-coupled cases and verify that the structural and fluid histories are generated, the manufactured source is active, and the expected RK3 stage records and particle evolution are present. A baseline case is also compared against a source-enabled target case.

**Coupled refinement** repeats the production coupled simulation at different temporal and spatial resolutions. Every simulation is compared against the analytical target, and relative space-time L2 errors are computed for structural state, particle position, particle circulation, particle core, bound circulation, panel force, probe velocity, and probe vorticity.

The scope of this study is deliberately restricted to a fixed particle population. It does not exercise native particle insertion, shedding, or deletion. It also does not establish general conservation properties of the Common Refinement Method (CRM) or RBF transfer operators. The moving-interface transfer check provides additional component-level evidence, but it should not be interpreted as a general proof of work conservation.

The coupled MMS therefore provides evidence that the implemented discrete structural source, fluid source, transfer, and communication paths behave consistently with the prescribed target. It should not be presented as a continuum Navier–Stokes MMS or as a complete verification of every physical and numerical mechanism in the FSI solver.

## 2. Files and their roles

The files in this directory are grouped below according to the part of the verification process they support.

### Coupled target and source

| `coupled_target.py` | Defines the analytical structural, geometric, and particle targets. It also generates the shared JSON manifest consumed by the coupled verification drivers. |
| `coupled_solid_source.py` | Constructs the structural manufactured source using the production weak form and generalized-α stage values. It also defines the reference aerodynamic force functional used in the structural source calculation. |
| `run_coupled_mms.py` | Executes one source-enabled production socket-coupled case and records the analytical target and numerical histories. It is intended as a short smoke test, not as a convergence study. |
| `run_coupled_refinement.py` | Executes the temporal and spatial coupled refinement levels, compares each simulation with its analytical target, and writes the relative errors and adjacent-level observed orders. |
| `plot_coupled_refinement.py` | Generates the coupled refinement error plots, plots numerical histories against their targets, and writes a separate table of pairwise observed orders. This script is run after the refinement driver and is not currently invoked automatically by `run_pipeline.sh`. |

### Coupled implementation checks

| `checks_manifest_loading.py` | Runs the G0 end-to-end source check using baseline and target coupled simulations. It verifies that the target source is active, that the production RK3 stages are recorded, and that the numerical particle state evolves. It launches the solvers. |
| `checks_manifest_loading.jl` | Loads the target manifest and checks the relevant production Julia source hooks. This is a source and manifest preflight; it does not execute a simulation. |
| `checks_target_roundtrip.py` | Checks that the analytical target can be serialized and recovered without changing its prescribed values. |
| `checks_target_derivatives.py` | Compares analytical target derivatives with finite-difference approximations to check their implementation. |
| `checks_coupled_solid.py` | Checks the identity between the structural manufactured source and the target residual using the production structural weak form. |

### Standalone structural MMS

| `plate.py` | Defines the analytical plate target, its spatial and temporal derivatives, and the corresponding manufactured load. |
| `fenics_plate.py` | Executes one structural MMS case using the production Reissner–Mindlin model and the prescribed manufactured load. |
| `refinement.py` | Plans or executes the structural spatial and temporal refinements. It writes numerical errors, observed-order assessments, convergence plots, and a run manifest. Simulations are launched only when `--execute` is supplied. |
| `jacobian_check.py` | Checks the production nonlinear structural residual Jacobian against finite-difference directional derivatives. This tests the consistency of the implemented tangent matrix, not the correctness of the underlying PDE. |
| `strong_weak_check.py` | Compares the production structural weak form against an independently evaluated manufactured strong form. This provides a separate check of the structural formulation. |
| `scalar_generalized_alpha.py` | Tests the generalized-α time integrator on a scalar oscillator with a known analytical solution. It is a component check rather than part of the main plate or coupled convergence sequence. |

### Standalone fluid studies

| `run_refinement.jl` | Runs the individual FLOWUnsteady parameter and resolution studies. The available studies are `particles`, `time`, `core`, `panels`, `fmm`, and `wake`. |
| `analyze.py` | Checks the generated fluid histories, compares quantities of interest against the finest completed numerical reference, estimates adjacent-level differences, and generates summary reports and plots. It does not calculate errors against an exact fluid solution. |
| `circulation_reference_check.jl` | Checks the FLOWVLM circulation solution using the residual of its underlying linear system. |
| `particle_position_stage_check.jl` | Checks particle-position source evaluation at the production RK3 integration stages. |
| `transfer_moving_interface.py` | Audits the production CRM/RBF transfer implementation on moving interfaces and selected mesh resolutions. This provides component-level transfer evidence, not a general conservation proof. |

### Shared utilities and execution

| `progress.py` | Provides shared progress and timing utilities so that long refinement cases report their execution status as they run. |
| `julia_with_openssl.sh` | Launches Julia with the project's compatible OpenSSL_jll library selected ahead of the system OpenSSL library. |
| `run_pipeline.sh` | Executes the verification stages in the prescribed order. It launches actual solver cases and is not a dry-run or package-installation script. |
| `__init__.py` | Marks the directory as an importable Python package. |

The older `run_mms.py` aggregator has been removed. It combined diagnostics from earlier versions of the verification workflow and generated an outdated partial-report format. The useful checks from that script are retained as independent drivers, making it easier to identify which component has failed.

## 3. Running the verification workflow

Run all commands from the root of the `VarFlExI-solver` repository.

The verification scripts use both Python/FEniCS and Julia/FLOWUnsteady. The pipeline runs Python through the Conda environment named `fenics-env` and launches Julia using the bundled OpenSSL-compatible launcher. By default, it expects the `FLOWUnsteady` repository to be located alongside `VarFlExI-solver`.

Before running the pipeline, make sure that `conda` and `julia` are available on your `PATH`, that the `fenics-env` environment is configured with the required Python and FEniCS dependencies, and that the sibling `FLOWUnsteady` project is available.

The complete verification sequence is launched using:

```sh
./verification/mms/run_pipeline.sh all
```

The pipeline executes four stages in order.

### Stage 1 — Preflight and component checks

The preflight stage checks individual components before starting the larger refinement studies.

It checks target serialization and derivatives, the structural manufactured source identity, the production nonlinear plate Jacobian, agreement between the manufactured strong and production weak structural formulations, moving-interface transfer, particle-position evaluation across RK3 stages, and the FLOWVLM circulation residual.

These checks are intended to isolate implementation issues. For example, if the structural Jacobian is inconsistent with finite differences, there is little value in interpreting a later nonlinear convergence failure as a mesh-resolution problem. Similarly, checking particle-position evaluation at individual RK3 stages helps distinguish an integration-stage implementation issue from an error accumulated over a complete simulation.

### Stage 2 — Standalone structural MMS

The structural stage first runs the scalar generalized-α oscillator check, followed by the plate spatial and temporal refinement studies.

Spatial refinement changes the structural mesh while keeping the time step fixed. Temporal refinement changes the time step while maintaining the same mesh and physical simulation duration.

Both studies use the production structural model and analytical manufactured loads. The outputs include error tables, order assessments, plots, and manifests recording the configuration and completed cases.

### Stage 3 — Standalone fluid resolution studies

The fluid stage runs the `time` and `panels` studies using FLOWUnsteady. After the simulations finish, `analyze.py` processes the completed histories and writes the sensitivity reports.

The time study examines the effect of reducing the integration time step while maintaining the nominal particle release spacing. The panel study examines the effect of increasing the number of bound spanwise panels.

The remaining fluid studies can be executed separately using `run_refinement.jl` if the influence of particle release, VPM core regularization, FMM accuracy, or wake-retention criteria needs to be investigated.

### Stage 4 — Coupled MMS and refinement

The coupled stage first generates the shared analytical target manifest. The Julia manifest check then verifies that the target can be loaded and that the required production source hooks are present.

Next, the pipeline executes a short source-enabled socket-coupled smoke test. This checks that the structural and fluid solvers communicate and generate the expected histories. It is followed by the G0 source-active production check, which runs baseline and target cases and verifies that the manufactured source is actually applied during the production integration stages.

Finally, the coupled temporal and spatial refinement cases are executed. Each level is compared against the analytical target, and the errors and adjacent-level observed orders are written to the coupled refinement report.

The pipeline stops when a command fails. It does not silently skip failed cases or install missing packages. A failed stage should therefore be investigated before interpreting the outputs of later stages.

To execute only one stage, pass its name instead of `all`.

```sh
./verification/mms/run_pipeline.sh preflight
./verification/mms/run_pipeline.sh plate
./verification/mms/run_pipeline.sh fluid
./verification/mms/run_pipeline.sh coupled
```

These commands run the selected stage directly. They do not execute the preceding stages automatically.

### Environment configuration

The default settings in `run_pipeline.sh` are intended for the repository layout used during development. They can be overridden through environment variables when the local installation differs.

| `FENICS_ENV` | `fenics-env` | Selects the Conda environment used for Python verification scripts. |
| `CONDA_BIN` | `conda` | Specifies the Conda executable. |
| `FLOWUNSTEADY_PROJECT` | Parent directory of the solver repository | Specifies the Julia project directory containing FLOWUnsteady. |
| `JULIA_BIN` | `verification/mms/julia_with_openssl.sh` | Specifies the Julia launcher used by the pipeline. |
| `JULIA_REAL_BIN` | Automatically detected | Specifies the underlying Julia executable used by the launcher. |
| `JULIA_OPENSSL_LIB` | Automatically detected | Overrides the OpenSSL_jll library directory when automatic detection is unsuitable. |
| `JULIA_NUM_THREADS` | Available CPU count | Sets the Julia thread count for the fluid refinement study. |

For example, if FLOWUnsteady is installed in a different directory, set its project path before running the coupled stage.

```sh
export FLOWUNSTEADY_PROJECT=/path/to/FLOWUnsteady
./verification/mms/run_pipeline.sh coupled
```

The Julia launcher is important because the project uses an OpenSSL_jll artifact whose library version may differ from the system OpenSSL installation. Using the launcher ensures that the Julia process loads the compatible project library rather than accidentally mixing the two installations.

## 4. Reading and interpreting the results

All pipeline-generated outputs are written under:

```text
verification/mms/results/pipeline/
```

The results directory contains generated simulation histories, reports, manifests, and plots. It is not source code. The repository's `.gitignore` currently ignores directories named `results`, so these outputs remain local unless they are deliberately archived or added to version control.

### 4.1 Standalone structural MMS

The spatial and temporal results are stored separately.

```text
results/pipeline/
    plate_spatial/
    plate_temporal/
    scalar_generalized_alpha/
```

The principal convergence files are:

```text
plate_spatial/plate_spatial_convergence.csv
plate_spatial/plate_spatial_convergence.png
plate_spatial/plate_spatial_manifest.json

plate_temporal/plate_temporal_convergence.csv
plate_temporal/plate_temporal_convergence.png
plate_temporal/plate_temporal_manifest.json
```

The CSV files contain the relative errors of the numerical structural fields against the analytical plate solution, together with the refinement parameters and observed-order assessments. The manifests record the configurations, case status, and information needed to reproduce the study.

For spatial refinement, both in-plane mesh directions are refined. The mesh spacing used for a single scalar convergence axis is

\[
h=\max(h_x,h_y).
\]

The individual spacings and mesh aspect ratios should also be inspected when interpreting the results.

For temporal refinement, the relevant refinement variable is the time step \(\Delta t\). The mesh remains fixed, so the resulting error variation is primarily associated with time integration.

An observed order is estimated from adjacent levels using

\[
p_{\mathrm{obs}}
=
\frac{\log(E_{\mathrm{coarse}}/E_{\mathrm{fine}})}
{\log(h_{\mathrm{coarse}}/h_{\mathrm{fine}})},
\]

where \(E\) is the error measured against the analytical solution and \(h\) represents the relevant spatial or temporal resolution.

A decreasing error sequence is necessary before an order can be interpreted. The refinement driver requires at least three levels and rejects sequences that are non-finite, at roundoff scale, nondecreasing, plateaued, or associated with strongly scattered adjacent rates. Such cases are marked `INCONCLUSIVE`.

Even when the driver reports a measured order, it should be described as an observed numerical rate for the tested configuration. It should not automatically be treated as proof of the formal theoretical order of the complete solver.

### 4.2 Standalone fluid studies

The fluid results are stored separately for each study.

```text
results/pipeline/fluid/
    time/
    panels/
```

After analysis, each study directory contains:

```text
summary.json
metrics.csv
report.md
error_vs_resolution.png
runtime_vs_error.png
particles_vs_error.png
```

The `summary.json` records the selected numerical reference, the completed cases, the failed cases, and the calculated metrics. The CSV provides the numerical results in tabular form, while the Markdown report summarizes the study and its reference interpretation.

The main error plots show the differences in lift coefficient, drag coefficient, and circulation relative to the selected finest numerical reference. The runtime and particle-count plots provide additional information about the computational cost associated with the observed differences.

Only cases marked complete are considered during analysis. Failed or incomplete simulations are excluded and listed in the report. At least two completed cases are required to perform a comparison.

The adjacent-level estimates in the time and panel studies can indicate whether the solution is stabilizing with refinement. However, because the reference is itself a numerical solution, these estimates do not establish convergence to the exact continuum flow.

The standalone rVPM solver does not use an Eulerian grid for its primary flow representation. The velocity and vorticity probes are therefore samples of the reconstructed particle field at fixed physical locations, rather than values obtained from a separate Eulerian grid solution.

### 4.3 Coupled MMS

The coupled results are stored under:

```text
results/pipeline/
    coupled_target.json
    coupled_smoke/
    G0/
    coupled_refinement/
```

The smoke directory contains the short coupled simulation histories, process logs, and coupling records. The G0 directory contains the results of the end-to-end source-active production check.

The principal refinement reports are:

```text
coupled_refinement/coupled_refinement_report.csv
coupled_refinement/coupled_refinement_report.json
```

The report contains the analytical-target relative space-time L2 errors for:

- Structural displacement and rotation state.
- Particle position.
- Particle circulation.
- Particle core parameters.
- Bound circulation.
- Aerodynamic panel force.
- Probe velocity.
- Probe vorticity.

The relative space-time L2 error for a numerical quantity \(q\) is calculated as

\[
E_{\mathrm{rel}}
=
\frac{
\left(
\displaystyle\int_0^T
\int_{\Omega}
\left\|q_{\mathrm{num}}-q_{\mathrm{target}}\right\|^2
\,d\Omega\,dt
\right)^{1/2}
}{
\left(
\displaystyle\int_0^T
\int_{\Omega}
\left\|q_{\mathrm{target}}\right\|^2
\,d\Omega\,dt
\right)^{1/2}
}.
\]

Here, \(\Omega\) represents the relevant spanwise or component domain for the quantity being measured. The numerical and target histories are evaluated over the same physical time interval.

The temporal study reduces the time step while keeping the structural mesh and fluid span resolution fixed. The spatial study increases the structural mesh dimensions and fluid spanwise resolution together. In the default sequence, the refinement levels are 1, 2, and 4.

Because the structural and fluid spatial resolutions are increased simultaneously, the spatial order obtained from this study is a combined coupled-system estimate. It cannot independently identify the spatial convergence order of the FEniCS solver or FLOWVLM discretization.

The report calculates adjacent-level observed orders using the configured refinement ratios. A credible rate requires errors to decrease across the refinement levels and the adjacent estimates to approach a reasonably consistent range. A flat or increasing error, or a strongly varying sequence of observed rates, should be treated as inconclusive. In such cases, the dominant error source may not be the variable being refined.

A few additional details are important when interpreting the coupled results.

The raw per-time error fields are Euclidean norms. The reported relative space-time L2 errors are integrated norms, normalized by the corresponding target norm. These are different quantities and should not be compared as though they were defined identically.

The target bound-circulation residual is evaluated using

\[
r_{\Gamma}
=
\frac{
\left\|A\Gamma_{\mathrm{target}}-b\right\|_2
}{
\max(\|b\|_2,\varepsilon)
},
\]

where \(A\) and \(b\) are the FLOWVLM circulation system matrix and right-hand side. This checks the consistency of the prescribed target circulation with the reference circulation system.

Probe vorticity is calculated as the centered finite-difference curl of the reconstructed velocity field at fixed physical locations. Its error therefore includes the effect of the probe spacing and numerical differentiation, in addition to the error in the reconstructed velocity itself.

The coupled particle population remains fixed throughout the test. Consequently, the results do not establish the correctness of native particle insertion, shedding, deletion, or wake-management procedures. Likewise, the moving-interface transfer test provides evidence for the tested configurations but does not establish general CRM/RBF conservation.

### Coupled refinement plots

The coupled refinement driver writes the numerical error and order reports, but the plots are generated separately. After a successful refinement run, execute:

```sh
conda run --name fenics-env python verification/mms/plot_coupled_refinement.py \
    --results verification/mms/results/pipeline/coupled_refinement
```

This produces:

```text
coupled_refinement_error_metrics.png
coupled_refinement_target_histories.png
coupled_refinement_observed_orders.csv
```

The first plot shows the analytical-target error across temporal and spatial refinement levels. The second compares selected numerical histories with their analytical targets. The CSV contains the pairwise observed orders and indicates whether the errors decrease monotonically.

The plotting script currently expects the default refinement levels `1,2,4`, including the `time_level_4` and `spatial_level_4` case directories. When using a different level sequence, the case selections in the plotting script should be adjusted accordingly.

## 5. Running individual drivers

The pipeline is the recommended way to execute the complete sequence because it sets up the required Python environment, Julia project, output paths, and OpenSSL-compatible launcher.

For more controlled execution, the individual drivers can also be run manually.

The following Python commands assume that the `fenics-env` Conda environment is active.

**Standalone plate convergence**

```sh
python verification/mms/refinement.py \
    --study spatial \
    --execute \
    --output-dir verification/mms/results/plate_spatial

python verification/mms/refinement.py \
    --study temporal \
    --execute \
    --output-dir verification/mms/results/plate_temporal
```

**Standalone fluid resolution studies**

```sh
verification/mms/julia_with_openssl.sh \
    --project="${FLOWUNSTEADY_PROJECT:-../FLOWUnsteady}" \
    verification/mms/run_refinement.jl \
    --study time,panels \
    --execute \
    --output verification/mms/results/fluid

python verification/mms/analyze.py \
    --study time \
    --results verification/mms/results/fluid

python verification/mms/analyze.py \
    --study panels \
    --results verification/mms/results/fluid
```

**Coupled smoke test**

```sh
FLOWUNSTEADY_PROJECT="${FLOWUNSTEADY_PROJECT:-../FLOWUnsteady}" \
JULIA_BIN=verification/mms/julia_with_openssl.sh \
python verification/mms/run_coupled_mms.py \
    --output verification/mms/results/coupled_smoke \
    --steps 3 \
    --dt 0.01 \
    --n-span 16 \
    --nx 8 \
    --ny 16
```

**Coupled temporal and spatial refinement**

```sh
FLOWUNSTEADY_PROJECT="${FLOWUNSTEADY_PROJECT:-../FLOWUnsteady}" \
JULIA_BIN=verification/mms/julia_with_openssl.sh \
python verification/mms/run_coupled_refinement.py \
    --study both \
    --levels 1,2,4 \
    --duration 0.03 \
    --dt 0.01 \
    --n-span 8 \
    --nx 4 \
    --ny 8 \
    --output verification/mms/results/coupled_refinement
```

After the coupled refinement finishes, generate its plots using `plot_coupled_refinement.py` as described above.

The direct commands do not automatically install dependencies or reproduce the environment setup performed by the prescribed pipeline. Python drivers should be executed in the configured FEniCS environment, and Julia verification files should be launched through `julia_with_openssl.sh` using the correct FLOWUnsteady project.

## 6. Scope of the current verification

The present verification workflow provides three complementary types of numerical evidence.

The standalone plate MMS measures structural errors against an analytical solution and provides separate spatial and temporal refinement results. The standalone fluid studies examine the sensitivity of the rVPM/FLOWVLM solution to selected numerical parameters using the finest completed simulation as a reference. The coupled MMS checks the production discrete FSI communication and source path against a prescribed analytical target over a fixed particle population.

Together, these studies help identify errors in individual solver components, quantify the effect of numerical resolution, and examine whether the coupled implementation behaves consistently when the two solvers exchange information.

They do not yet amount to a complete verification of the continuum fluid equations, particle lifecycle operations, or general interface conservation. Those questions require additional manufactured solutions, independent conservation checks, and dedicated tests of particle insertion, shedding, and deletion.

The results should therefore be interpreted according to the specific question each study was designed to answer. A successful simulation demonstrates execution, a component check establishes consistency of a particular implementation, a refinement study measures the behaviour of numerical errors or differences under refinement, and an analytical-target coupled test provides evidence for the discrete coupling path. Keeping these distinctions explicit is necessary when reporting the accuracy and verification status of VarFlExI.