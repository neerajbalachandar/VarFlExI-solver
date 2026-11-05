#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

CONDA_BIN="${CONDA_BIN:-conda}"
FENICS_ENV="${FENICS_ENV:-fenics-env}"
JULIA_BIN="${JULIA_BIN:-$ROOT/verification/mms/julia_with_openssl.sh}"
FLOWUNSTEADY_PROJECT="${FLOWUNSTEADY_PROJECT:-$(cd "$ROOT/.." && pwd)}"
export FLOWUNSTEADY_PROJECT JULIA_BIN

run_python() {
    "$CONDA_BIN" run --no-capture-output --name "$FENICS_ENV" python "$@"
}

usage() {
    cat <<'EOF'
Usage: verification/mms/run_pipeline.sh [preflight|plate|fluid|coupled|all]

Environment:
  FENICS_ENV              Conda environment name (default: fenics-env)
  CONDA_BIN               Conda executable (default: conda)
  JULIA_BIN               Julia launcher (default: bundled OpenSSL-compatible launcher)
  JULIA_REAL_BIN          Julia executable used by the bundled launcher
  JULIA_OPENSSL_LIB       OpenSSL artifact lib directory, if not the default
  JULIA_NUM_THREADS       Julia threads for the fluid FMM study (default: available CPUs)
  FLOWUNSTEADY_PROJECT    Julia project directory (default: ../FLOWUnsteady)

All execution phases run real solver cases and can take substantial time.
Outputs are written below verification/mms/results/pipeline/.
EOF
}

phase="${1:-all}"
case "$phase" in
    -h|--help|help) usage; exit 0 ;;
    preflight|plate|fluid|coupled|all) ;;
    *) usage >&2; exit 2 ;;
esac

run_preflight() {
    run_python verification/mms/checks_target_roundtrip.py
    run_python verification/mms/checks_target_derivatives.py
    run_python verification/mms/checks_coupled_solid.py
    run_python verification/mms/jacobian_check.py --output verification/mms/results/pipeline/plate_jacobian_check.csv
    run_python verification/mms/strong_weak_check.py --mode continuous --formulation nonlinear --resolution 12x24
    run_python verification/mms/transfer_moving_interface.py \
        --csv verification/mms/results/pipeline/transfer_moving_interface.csv \
        --json verification/mms/results/pipeline/transfer_moving_interface.json
    "$JULIA_BIN" --project="$FLOWUNSTEADY_PROJECT" verification/mms/particle_position_stage_check.jl
    "$JULIA_BIN" --project="$FLOWUNSTEADY_PROJECT" verification/mms/circulation_reference_check.jl
}

run_plate() {
    run_python verification/mms/scalar_generalized_alpha.py \
        --output-dir verification/mms/results/pipeline/scalar_generalized_alpha
    run_python verification/mms/refinement.py --study spatial --execute \
        --output-dir verification/mms/results/pipeline/plate_spatial
    run_python verification/mms/refinement.py --study temporal --execute \
        --output-dir verification/mms/results/pipeline/plate_temporal
}

run_fluid() {
    local fluid_threads="${JULIA_NUM_THREADS:-$(nproc)}"
    printf '[fluid] Julia threads=%s\n' "$fluid_threads"
    JULIA_NUM_THREADS="$fluid_threads" "$JULIA_BIN" --project="$FLOWUNSTEADY_PROJECT" verification/mms/run_refinement.jl \
        --study time,panels --execute --output verification/mms/results/pipeline/fluid
    run_python verification/mms/analyze.py --study time --results verification/mms/results/pipeline/fluid
    run_python verification/mms/analyze.py --study panels --results verification/mms/results/pipeline/fluid
}

run_coupled() {
    run_python -c 'from verification.mms.coupled_target import build_default_target, write_manifest; write_manifest(build_default_target("config", 0.03, 16), "verification/mms/results/pipeline/coupled_target.json")'
    "$JULIA_BIN" --project="$FLOWUNSTEADY_PROJECT" verification/mms/checks_manifest_loading.jl \
        verification/mms/results/pipeline/coupled_target.json
    run_python verification/mms/run_coupled_mms.py \
        --output verification/mms/results/pipeline/coupled_smoke \
        --steps 3 --dt 0.01 --n-span 16 --nx 8 --ny 16
    run_python verification/mms/checks_manifest_loading.py \
        --output verification/mms/results/pipeline/G0
    run_python verification/mms/run_coupled_refinement.py \
        --study both --levels 1,2,4 --duration 0.03 --dt 0.01 \
        --n-span 8 --nx 4 --ny 8 \
        --output verification/mms/results/pipeline/coupled_refinement
}

case "$phase" in
    preflight) run_preflight ;;
    plate) run_plate ;;
    fluid) run_fluid ;;
    coupled) run_coupled ;;
    all)
        run_preflight
        run_plate
        run_fluid
        run_coupled
        ;;
esac

printf 'Pipeline phase complete: %s\n' "$phase"
