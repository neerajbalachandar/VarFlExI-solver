# Solver Verification Workflows

This directory separates analytic MMS checks, solver-component tests, and physical resolution studies. Passing an analytic identity is not evidence that the coupled FSI solver has passed continuum MMS.

## Current findings

The structural model uses mixed CG2 fields `(u_x,u_y,w,theta_x,theta_y)` on a rectangular Reissner-Mindlin plate, clamped at `y=0`. Its damping is `eta_m M + eta_k K_ref`; the generalized-alpha implementation uses `gamma=1/2+alpha_f-alpha_m` and `beta=(gamma+1/2)^2/4`. The default configuration is geometrically nonlinear with full quadratic in-plane membrane strain enabled.

The fluid solver uses FLOWUnsteady 3.4.0, FLOWVLM 2.1.4 and FLOWVPM 4.0.3 from `/home/dysco/FLOWUnsteady/Manifest.toml`. It advances a ReformulatedVPM particle field using FLOWVPM's RK3 default, with Gaussian-erf regularization and CoreSpreading. FLOWUnsteady releases `p_per_step` particles along each shed segment; the modular setup sets the initial radius to `lambda_vpm * U * dt / p_per_step`. The measured current scales are in [current_scales.csv](resolution/results/current_scales.csv). The fluid has no Eulerian volume mesh.

The structural constitutive audit found that `nonlinear_membrane_variation()` included the quadratic in-plane Green-Lagrange term while `nonlinear_membrane_strain()` omitted it. The default is documented as the full Green-Lagrange option, so the strain implementation was corrected to match the variation. A symbolic directional-derivative regression check now passes.

## Reproduce analytic/component checks

```bash
python3 verification/mms/run_mms.py
julia --project="$(dirname "$PWD")" verification/mms/fluid_components.jl
python3 verification/mms/run_mms.py
```

`run_mms.py` is a fast analytic/component suite; it does not call `run.sh` or solve the plate/fluid PDEs. It checks Taylor-Green identities, plate symbolic source/edge identities, nonlinear strain variation, RBF resultant/work algebra, and a six-level scalar generalized-alpha oscillator convergence run. It incorporates Julia component results from the existing CSV; it does not invoke Julia itself. The Julia command checks kernel normalization/limits, actual direct Gaussian-erf particle velocity, an RK3 core-spreading increment, an RK3 particle advance under prescribed uniform velocity and zero gradient, and FLOWVLM freestream registration.

The assumed fluid reference is the two-dimensional Taylor-Green vortex (zero third component):

`u = U [sin(kx) cos(ky), -cos(kx) sin(ky), 0] exp(-2 nu k^2 t)`

with `p/rho = U^2/4 [cos(2kx)+cos(2ky)] exp(-4 nu k^2 t)`. The Python test substitutes it into incompressible Navier-Stokes and the vorticity transport equation and checks that divergence and source residuals are identically zero. This verifies the analytic reference, not FLOWVPM's particle solution.

The plate reference is the smooth envelope `B=sin(pi*x/Lx)^4 sin(pi*y/Ly)^4`, with angular frequency `omega=2.3 rad/s`:

`u_x=1e-4 B cos(omega t)`, `u_y=1e-4 B sin(omega t)`, `w=1e-3 B sin(omega t)`, `theta_x=1e-3 B cos(omega t)`, `theta_y=-1e-3 B sin(omega t)`.

Displacements are in metres and rotations are dimensionless. The envelope is zero with sufficient derivatives on all edges; the root at `y=0` is clamped and other edges satisfy the generated natural conditions. `plate.py` derives forcing from the configured elastic, inertial, shear, and damping laws. `fenics_plate.py` advances the production FEM model under those loads and compares the numerical final-time fields to the analytic fields.

### Solid convergence sweep values

These are the current plate-PDE MMS convergence sweeps, separate from the scalar integrator test and the physical parameter sweeps under `verification/resolution/`:

| Study | `nx x ny` elements | `dt` [s] | fixed quantity |
|---|---|---|---|
| Spatial | `6x8`, `12x16`, `24x32`, `48x64`, `96x128` | `0.0000625` | time step; duration `0.01 s` |
| Temporal | `48x128` | `0.01`, `0.005`, `0.0025`, `0.00125`, `0.000625`, `0.0003125` | mesh; duration `0.05 s` |

By default each study runs both linear and geometrically nonlinear formulations (10 spatial or 12 temporal cases per formulation). The raw table records `hx`, `hy`, DOFs, initial interpolation errors, Newton diagnostics, and runtime. Pairwise orders are generated for every relative-L2 field when both errors are finite and positive. Both directional spacings are recorded; `max(hx,hy)` is the explicitly identified scalar spatial axis. The order sequence is coarse-to-fine. Plans alone do not solve anything:

```bash
python3 verification/mms/refinement.py --study spatial
python3 verification/mms/refinement.py --study temporal
```

To execute the plate cases, use the legacy FEniCS interpreter in `fenics-env` and add `--execute`; or restrict to `--formulation linear` / `--formulation nonlinear`. `run_mms.py` does not call these studies.

The plate MMS load in `fenics_plate.py` uses production `mass_form`, `damping_form`, and `internal_work`, with manufactured acceleration, velocity, and displacement all evaluated at the single continuous force time `t_f=t_(n+1)-alpha_f*dt`. The numerical residual still evaluates acceleration at `alpha_m` and displacement/velocity at `alpha_f`; this difference is the intended generalized-alpha time-discretization error. The independent strong source in `plate.py` is checked against production weak forms at one common physical time by `strong_weak_check.py`. The complete production residual/Jacobian finite-difference check is `jacobian_check.py`. Both require DOLFIN and are separate from the continuous plate convergence sweep.

This is the **continuous-equation MMS**. No discrete-equation consistency test is currently implemented; such a test would construct forcing from the actual generalized-alpha stage values and is useful for residual/update debugging, but it would suppress temporal truncation error and must not be used to claim temporal convergence. Other open items from the audit include a tolerance-backed moment test and complete CRM/geometry-to-FE transfer tests, plus direct-versus-FMM velocity/Jacobian comparisons. The existing Julia particle checks do not establish those properties.

### Fluid convergence status

There is currently **no fluid PDE or particle-field convergence sweep**. The Julia tests exercise the actual direct Gaussian-erf velocity kernel for a two-particle case, core spreading, and the RK3 state update under prescribed constant U/J; they do not establish FMM accuracy, wake evolution, or full rVPM PDE convergence. A meaningful rVPM convergence study must control particle release spacing, core radius, timestep, bound-surface panels, wake insertion/remapping/removal, and FMM settings. There is no compatible manufactured source for the actual filtered rVPM equations in the current solver; Taylor–Green remains an F1 continuum identity only. No fluid order is measured or claimed. See the full source audit in [AUDIT.md](AUDIT.md).

Run these from the repository root. `conda run` selects the environment without relying on the shell already being activated:

```bash
conda run -n fenics-env python verification/mms/strong_weak_check.py
conda run -n fenics-env python verification/mms/jacobian_check.py
conda run -n fenics-env python verification/mms/fenics_plate.py --nx 12 --ny 32 --dt 0.0025 --duration 0.05
conda run -n fenics-env python verification/mms/fenics_plate.py --nx 12 --ny 32 --dt 0.0025 --duration 0.05 --nonlinear
conda run -n fenics-env python verification/mms/refinement.py --study spatial --execute
conda run -n fenics-env python verification/mms/refinement.py --study temporal --execute
```

The strong/weak check writes `plate_strong_weak.csv/.json`; the Jacobian check writes `plate_jacobian_check.csv`; a single plate case appends to `plate_fenics.csv` and writes its exact/numerical/error field PNG. Refinement writes `plate_{spatial,temporal}_{plan,manifest,raw,convergence}` JSON/CSV files and a convergence PNG under `verification/mms/results/`. The manifest records the exact child command, case status/return code, and SHA-256 of `solid_params.yaml`; each completed raw row records the config hash, solver tolerances, mesh spacings/aspect ratio, and runtime. The analytic `run_mms.py` writes `mms_metrics.csv`, `mms_summary.json`, `MMS_REPORT.md`, and `plate_exact_displacement.png`; it does not execute any DOLFIN or Julia program.

## Resolution audit and run generation

```bash
python3 verification/resolution/analyze.py audit
python3 verification/resolution/analyze.py plan --study time
python3 verification/resolution/analyze.py plan --study structural
python3 verification/resolution/analyze.py plan --study panels
python3 verification/resolution/analyze.py plan --study particles
python3 verification/resolution/analyze.py plan --study transfer
python3 verification/resolution/analyze.py plan --study fmm
```

Every plan creates independent YAML copies below `resolution/runs/`; it never edits `config/` and does not launch simulations. The `time` cases hold total duration, particle release spacing, and initial core radius approximately fixed by inversely changing `particles_per_step` as `dt` changes. This is a **co-refinement**, not an isolated temporal study. Panel count and coupling station count are independent sweeps; `comm_nspan` controls structural/coupling sampling. Manifests now record mesh/panel/particle values, per-case config SHA-256, result paths, execution status, return code, and a run log when executed. `U*dt/hp` is equal to `particles_per_step` under the assumed uniform release model, and `Fo_p` is only a descriptive scale ratio, not a stability criterion. The optional `--execute` flag runs cases sequentially through `run.sh`; `VARFLEXI_CONFIG_DIR` selects each generated configuration while the default run continues to use `config/`.

After a completed sweep:

```bash
python3 verification/resolution/collect.py --study time
python3 verification/resolution/collect.py --study structural
```

The collector compares each result history against the finest completed case on a shared time interval, writes CSV/JSON and plots, and reports an estimated discretization difference, not error against an exact physical FSI solution.

The production generalized-alpha modal study is available when DOLFIN/SLEPc works:

```bash
python3 verification/resolution/modal_analysis.py --modes 12
python3 verification/resolution/modal_analysis.py --modes 12 --transverse-amplitude 0.001
```

## Status and limitations

| Capability | Current status |
|---|---|
| Taylor-Green exact momentum/vorticity identities | PASS, residuals exactly zero symbolically |
| Linear and nonlinear plate source derivation and boundary checks | PASS, symbolic only |
| Green-Lagrange energy/variation consistency | PASS, symbolic directional derivative |
| Scalar generalized-alpha oscillator | PASS, six levels; measured rates approximately 1.97-2.01 for selected errors |
| FLOWVLM stored freestream, FLOWVPM kernel/core and prescribed-velocity RK3 state update | PASS, installed package component tests; not production UJ/wake |
| Local-RBF resultant and paired virtual-work algebra | PASS, relative discrepancies around machine precision; moment discrepancy measured, not constrained |
| Production weak/independent strong plate identity and complete Jacobian finite difference | IMPLEMENTED; run explicitly in `fenics-env` |
| FEniCS numerical MMS, mesh/time convergence, FEM modes | IMPLEMENTED; run explicitly in `fenics-env`; no result is claimed until run |
| rVPM full equation MMS | NOT IMPLEMENTED; installed particle evolution has no compatible general manufactured vorticity-source injection |
| Lifting-surface circulation/wake and integrated communication tests | Not yet component-tested end-to-end; only FLOWVLM freestream registration is smoke-tested |
| Physical coupled-FSI convergence | Not run; no result histories were produced by the sweep plans |
| Full continuum FSI MMS | Not claimed; the two coupled solvers lack compatible manufactured interface source terms |

No plate or fluid PDE convergence order is claimed until the corresponding sweep is executed and its outputs inspected. The scalar oscillator order is a time-integrator-only result. The source audit and test scope are detailed in [AUDIT.md](AUDIT.md); generated metrics are in `mms/results/`. `resolution/results/RESOLUTION_ANALYSIS.md` distinguishes source-derived algorithm details, screening estimates, and unresolved empirical limits. These instructions do not run `run.sh`; the separate physical FSI workflow is not an MMS test.
