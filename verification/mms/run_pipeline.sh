#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

CONDA_BIN="${CONDA_BIN:-conda}"
if ! command -v "$CONDA_BIN" &>/dev/null && [ -x "/home/dysco/anaconda3/bin/conda" ]; then
    CONDA_BIN="/home/dysco/anaconda3/bin/conda"
fi
FENICS_ENV="${FENICS_ENV:-fenics-env}"
JULIA_BIN="${JULIA_BIN:-$ROOT/verification/mms/julia_with_openssl.sh}"
FLOWUNSTEADY_PROJECT="${FLOWUNSTEADY_PROJECT:-$(cd "$ROOT/.." && pwd)}"
export FLOWUNSTEADY_PROJECT JULIA_BIN

run_python() {
    "$CONDA_BIN" run --no-capture-output --name "$FENICS_ENV" python "$@"
}

usage() {
    cat <<'EOF'
Usage: verification/mms/run_pipeline.sh [test|structural|transfer|fluid|coupled|all]

Unified MMS Verification Suite (CMAME Framework):
  test          Run automated consistency tests (derivatives, BCs, independence, RK3, work)
  structural    Standalone Reissner–Mindlin plate MMS (spatial & temporal refinement)
  transfer      Interface force/moment transfer MMS (CRM & RBF work duality)
  fluid         Particle-level rVPM MMS (standalone source, stretching FD, RK3 convergence)
  coupled       Coupled FSI MMS refinement across levels
  all           Run the full verification suite

Environment:
  FENICS_ENV              Conda environment name (default: fenics-env)
  CONDA_BIN               Conda executable (default: conda or /home/dysco/anaconda3/bin/conda)
  JULIA_BIN               Julia launcher (default: bundled OpenSSL-compatible launcher)
  FLOWUNSTEADY_PROJECT    Julia project directory (default: ../FLOWUnsteady)
EOF
}

phase="${1:-all}"
case "$phase" in
    -h|--help|help) usage; exit 0 ;;
    test|structural|plate|transfer|fluid|coupled|all) ;;
    *) usage >&2; exit 2 ;;
esac

run_test() {
    echo "=== Running Automated MMS Consistency Tests ==="
    run_python verification/mms/mms_driver.py test
}

run_structural() {
    echo "=== Running Standalone Structural MMS ==="
    run_python verification/mms/mms_driver.py structural
}

run_transfer() {
    echo "=== Running Interface Transfer MMS ==="
    run_python verification/mms/mms_driver.py transfer
}

run_fluid() {
    echo "=== Running Particle-Level rVPM MMS ==="
    # Generate manifest first if not present
    run_python verification/mms/mms_driver.py coupled --dry-run
    local manifest="verification/mms/results/coupled/manifest_level1.json"
    "$JULIA_BIN" --project="$FLOWUNSTEADY_PROJECT" verification/mms/mms_rvpm.jl \
        --manifest "$manifest" \
        --output verification/mms/results/particles \
        --study standalone
    "$JULIA_BIN" --project="$FLOWUNSTEADY_PROJECT" verification/mms/mms_rvpm.jl \
        --manifest "$manifest" \
        --output verification/mms/results/particles \
        --study refinement
}

run_coupled() {
    echo "=== Running Coupled FSI MMS ==="
    run_python verification/mms/mms_driver.py coupled --levels 1,2,4
}

case "$phase" in
    test)             run_test ;;
    structural|plate) run_structural ;;
    transfer)         run_transfer ;;
    fluid)            run_fluid ;;
    coupled)          run_coupled ;;
    all)
        run_test
        run_structural
        run_transfer
        run_fluid
        run_coupled
        ;;
esac
