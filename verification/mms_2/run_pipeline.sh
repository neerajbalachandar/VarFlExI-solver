#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
CONDA_BIN="${CONDA_BIN:-conda}"
if ! command -v "$CONDA_BIN" &>/dev/null && [ -x "/home/dysco/anaconda3/bin/conda" ]; then
    CONDA_BIN="/home/dysco/anaconda3/bin/conda"
fi
FENICS_ENV="${FENICS_ENV:-fenics-env}"
JULIA_BIN="${JULIA_BIN:-$ROOT/verification/mms_2/julia_with_openssl.sh}"
FLOWUNSTEADY_PROJECT="${FLOWUNSTEADY_PROJECT:-$(cd "$ROOT/.." && pwd)}"
export FLOWUNSTEADY_PROJECT JULIA_BIN

run_python() {
    "$CONDA_BIN" run --no-capture-output --name "$FENICS_ENV" python "$@"
}

usage() {
    cat <<'EOF'
Usage: verification/mms_2/run_pipeline.sh [preflight|structural|fluid|coupled|particle-discrete|reconstruct-snapshot]

  preflight          Symbolic target, norms, refinement and physics-config checks only
  structural         Copied standalone structural MMS
  fluid              Velocity-vorticity field MMS (currently fail-closed pending source projection)
  coupled            Coupled field MMS (currently fail-closed pending source projection)
  particle-discrete  Retained fixed-population particle-ODE MMS from the baseline
  reconstruct-snapshot Reconstruct one complete saved particle snapshot (requires --manifest, --snapshot)

No mode in this script is run as part of implementation. Fluid and coupled
field solves require the full-state source projection described in the README.
EOF
}

mode="${1:-preflight}"
case "$mode" in
    -h|--help|help) usage; exit 0 ;;
    preflight|structural|fluid|coupled|particle-discrete|reconstruct-snapshot) ;;
    *) usage >&2; exit 2 ;;
esac

run_python verification/mms_2/mms2_driver.py "$mode" "${@:2}"
