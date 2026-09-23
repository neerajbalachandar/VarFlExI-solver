Implemented the time-complexity framework pieces.
Added:
- [scripts/solver/time_complexity.py](/home/dysco/FLOWUnsteady/VarFLEXI-rVPM-Fenics/scripts/solver/time_complexity.py)
  - Sweeps dt, spatial size, and particles_per_step
  - Runs n_runs >= 3 by default
  - Fixes thread env vars like OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, MKL_NUM_THREADS, JULIA_NUM_THREADS
  - Optional CPU pinning via --cpu-list
  - Samples process-tree RSS from /proc
  - Writes raw run data and aggregate mean/variance/std summaries
- [notebooks/time_complexity_analysis.ipynb](/home/dysco/FLOWUnsteady/VarFLEXI-rVPM-Fenics/notebooks/time_complexity_analysis.ipynb)
  - Loads benchmark CSVs
  - Plots total and per-step timings
  - Plots component timings and bottleneck fractions
  - Fits empirical log-log exponents
  - Plots peak memory and memory traces

  
Updated instrumentation:
- [solid/structural_solver.py](/home/dysco/FLOWUnsteady/VarFLEXI-rVPM-Fenics/solid/structural_solver.py)
  - Adds force_transfer_apply_time
  - Adds geometry_transfer_apply_time
  - Adds solid_solve_time
  - Adds solid_io_time
- [coupling/varflexi.py](/home/dysco/FLOWUnsteady/VarFLEXI-rVPM-Fenics/coupling/varflexi.py)
  - Carries those timing fields into coupling_history.csv
  - Includes CRM/RBF setup timings
- [coupling_transfer.py](/home/dysco/FLOWUnsteady/VarFLEXI-rVPM-Fenics/coupling/utils/coupling_transfer.py)
  - Records one-time CRM setup, force RBF setup, and geometry RBF setup time