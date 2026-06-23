#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/tomato/5TD"
BATCH_DIR="${PROJECT_DIR}/data/annotation_batches/rgb_keyframes_2026-06-22"
IMAGES_DIR="${BATCH_DIR}/images"
LABELS_FILE="${BATCH_DIR}/labels.txt"
LABELME_BIN="${LABELME_BIN:-/home/tomato/miniconda3/bin/labelme}"

if [ ! -x "${LABELME_BIN}" ]; then
  echo "labelme not found or not executable: ${LABELME_BIN}" >&2
  echo "Install it with: python3 -m pip install labelme" >&2
  exit 1
fi

cd "${PROJECT_DIR}"
exec "${LABELME_BIN}" "${IMAGES_DIR}" --labels "${LABELS_FILE}" --nodata
