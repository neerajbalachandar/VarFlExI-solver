#!/usr/bin/env bash
set -euo pipefail


#config folder
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="${VARFLEXI_CONFIG_DIR:-$ROOT_DIR/config}"




# if command -v conda >/dev/null 2>&1; then
#   # shellcheck disable=SC1091
#   source "$(conda info --base)/etc/profile.d/conda.sh"
#   conda activate fenics-env
# else
#   echo "conda not found; make sure the Python/FEniCS environment is active." >&2
# fi

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  set +u
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate fenics-env
  set -u
else
  echo "conda not found; make sure the Python/FEniCS environment is active." >&2
fi


#kill all background processes on exit
cleanup() {
  jobs -pr | xargs -r kill
}
trap cleanup EXIT INT TERM

cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

python coupling/varflexi.py --config_path "$CONFIG_DIR" &
COUPLING_PID=$!

# Give the server a moment to bind before clients connect.
sleep 2

# solid solver
python solid/structural_solver.py --config_path "$CONFIG_DIR" &
SOLID_PID=$!

# fluid solver
# julia fluid/fluid.jl --fluid "$CONFIG_DIR/fluid_params.yaml" --solid "$CONFIG_DIR/solid_params.yaml" --coupling "$CONFIG_DIR/coupling_params.yaml"
# Julia project (~/FLOWUnsteady)
PROJECT_DIR="$(cd "$ROOT_DIR/.." && pwd)"

# fluid solver
julia --project="$PROJECT_DIR" \
    fluid/fluid.jl \
    --fluid "$CONFIG_DIR/fluid_params.yaml" \
    --solid "$CONFIG_DIR/solid_params.yaml" \
    --coupling "$CONFIG_DIR/coupling_params.yaml"


wait "$COUPLING_PID" "$SOLID_PID"
