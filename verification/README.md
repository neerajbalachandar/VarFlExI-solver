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

The first Python command checks Taylor-Green incompressible Navier-Stokes/vorticity identities, derives linear and nonlinear Reissner-Mindlin strong-form source fields independently of the FEniCS residual, verifies edge conditions, and tests the RBF backend's weighted resultant identity. The Julia command checks the installed Gaussian-erf radial normalization and actual RK3 core-spreading implementation. Results are written under `mms/results/`.

Available legacy FEniCS numerical MMS commands (not executable until DOLFIN is available):

```bash
python3 verification/mms/fenics_plate.py --nx 12 --ny 32 --dt 0.0025 --duration 0.05
python3 verification/mms/fenics_plate.py --nx 12 --ny 32 --dt 0.0025 --duration 0.05 --nonlinear
python3 verification/mms/refinement.py --study spatial --execute
python3 verification/mms/refinement.py --study temporal --execute
```

The single-case driver applies the symbolic body loads to the production model and reports displacement, velocity, acceleration, membrane-strain, curvature, and shear errors. Refinement is explicitly opt-in. The active interpreter has no importable `dolfin`; the configured `fenics-env` Python reports that its DOLFIN pkg-config file cannot be found. Consequently there are no discrete plate errors or observed convergence orders in the generated report yet.

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

Every plan creates independent YAML copies below `resolution/runs/`; it never edits `config/` and does not launch simulations. Time cases hold total duration, particle release spacing and initial particle core radius fixed by scaling `particles_per_step` inversely with `dt`. The VLM panel and coupling-transfer station sweeps are independent; `comm_nspan` now controls only the structural/coupling sampling count. The optional `--execute` flag runs cases sequentially through `run.sh`; `VARFLEXI_CONFIG_DIR` selects each generated configuration while the default run continues to use `config/`.

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
| FLOWVLM stored freestream, FLOWVPM kernel normalization and RK3 core growth | PASS, installed package component tests |
| Local-RBF weighted resultant algebra | PASS, relative discrepancy `3.87e-16`; not the full CRM/RBF-to-FEniCS interface map |
| FEniCS numerical MMS, mesh/time convergence, FEM modes | NOT RUN, DOLFIN unavailable |
| rVPM full equation MMS | NOT IMPLEMENTED; installed particle evolution has no compatible general manufactured vorticity-source injection |
| Lifting-surface circulation/wake and integrated communication tests | Not yet component-tested end-to-end; only FLOWVLM freestream registration is smoke-tested |
| Physical coupled-FSI convergence | Not run; no result histories were produced by the sweep plans |
| Full continuum FSI MMS | Not claimed; the two coupled solvers lack compatible manufactured interface source terms |

No empirical discretization limit or convergence order is claimed. `resolution/results/RESOLUTION_ANALYSIS.md` distinguishes source-derived algorithm details, screening estimates, and experimentally unresolved limits. Its beam fundamental frequency is a strip approximation only; use the SLEPc eigenanalysis before relying on it for structural time-step selection. Julia was loaded successfully and the FLOWVLM freestream registration smoke test passed, but `run.sh` was not run because FEniCS is unavailable.
