# Fluid-Only Resolution Study

`run_refinement.jl` runs the same FLOWUnsteady/FLOWVPM formulation and fluid
configuration as `run.sh`, but without the FSI socket or structural motion.
It reads the three repository YAML files; sweep levels alter only the named
resolution parameter. The default run plan does not launch simulations.

```bash
julia --project=.. verification/fluid_resolution/run_refinement.jl --study all
julia --project=.. verification/fluid_resolution/run_refinement.jl --study particles --execute
```

Results, manifests, per-case configs, logs, and histories are written under
`verification/mms/results/` unless `--output PATH` is supplied.

`vpm_sfs: none` matches the production default. Set `vpm_sfs` to
`dynamic_two_level` or `dynamic_three_level` for a modeled turbulent-cascade
comparison. These are subfilter-scale closures, not turbulent inflow models;
the configured sinusoidal freestream perturbation remains a deterministic
gust. The standalone study is fixed-geometry fluid verification, not a coupled
FSI convergence study.
