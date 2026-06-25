#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${SCRIPT_DIR}/models"
WORLD_FILE="${SCRIPT_DIR}/worlds/warthog_flat_test.sdf"

set +u
source /opt/ros/jazzy/setup.bash
set -u
export GZ_SIM_RESOURCE_PATH="${MODEL_DIR}:${GZ_SIM_RESOURCE_PATH:-}"
export SDF_PATH="${MODEL_DIR}:${SDF_PATH:-}"

exec gz sim -r "${WORLD_FILE}"
