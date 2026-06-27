#!/usr/bin/env bash
set -euo pipefail

ROOT="${FIVE_TD_ROOT:-/home/bqtec/5TD}"
cd "$ROOT"

exec python3 tools/robot/vision_autodrive_trajectory.py \
  --camera "${CAMERA:-/dev/video0}" \
  --dry-run \
  --no-display \
  --cpu \
  --width "${WIDTH:-640}" \
  --height "${HEIGHT:-480}" \
  --fps "${FPS:-5}" \
  --rate-hz "${RATE_HZ:-1}"
