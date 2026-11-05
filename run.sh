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

# Snapshot the exact inputs beside this run's data before any solver starts.
# This only copies existing settings; it does not change or relax them.
python - "$CONFIG_DIR" <<'PY'
import json
import shutil
import sys
from pathlib import Path

import yaml

config_dir = Path(sys.argv[1]).expanduser().resolve()
names = ("fluid_params.yaml", "solid_params.yaml", "coupling_params.yaml")
configs = {name: yaml.safe_load((config_dir / name).read_text()) or {} for name in names}
solid = configs["solid_params.yaml"]
coupling = configs["coupling_params.yaml"]
root_value = solid.get("results_root") or coupling.get("results_root") or configs["fluid_params.yaml"].get("save_directory", "results")
results_root = Path(root_value).expanduser()
if not results_root.is_absolute():
    results_root = (Path.cwd() / results_root).resolve()
snapshot_dir = results_root / "config"
snapshot_dir.mkdir(parents=True, exist_ok=True)
for name in names:
    shutil.copy2(config_dir / name, snapshot_dir / name)
(results_root / "run_configuration.json").write_text(
    json.dumps({"config_source": str(config_dir), "configs": configs}, indent=2) + "\n",
    encoding="utf-8",
)
print(f"Run configuration snapshot: {snapshot_dir} and {results_root / 'run_configuration.json'}")
PY

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
