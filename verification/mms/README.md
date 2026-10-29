# Coupled fixed-population FSI MMS

This verification case manufactures a time-dependent structural field and a fixed rVPM particle population. It uses the production FLOWVLM circulation solve and force calculator, the installed FLOWVPM RK3 update with stage-aware source terms, the Reissner–Mindlin structural residual, configured interface transfers, and the socket coupler.

The claim is Objective B: end-to-end transfer and FSI accuracy for the coupled structural source. The particle source is fixed-population. This case does not verify native wake birth/removal, continuum Navier–Stokes consistency, CRM/RBF moment conservation, or convergence from a single run.

`coupled_target.py` is the analytic source of truth and writes the shared JSON manifest. `run_coupled_mms.py` starts the actual coupling server and both production solvers, then writes fluid target/numerical force and circulation histories, RK stage source traces, solid state/load histories, and a summary. The fixed target particles are inserted into FLOWVPM's runtime field immediately before its first RK stage, so the manufactured source is active from the first fluid step.

Component checks are `checks_target_roundtrip.py`, `checks_target_derivatives.py`, and `checks_coupled_solid.py`. The first coupled milestone is accepted only when both fluid and solid histories exist and contain finite target and numerical values. No convergence order is implied by that run.
