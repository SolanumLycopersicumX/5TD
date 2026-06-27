#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p lan_share

PORT="${PORT:-8000}"
exec python3 tools/lan_file_server.py \
  --directory lan_share \
  --host 0.0.0.0 \
  --port "$PORT"
