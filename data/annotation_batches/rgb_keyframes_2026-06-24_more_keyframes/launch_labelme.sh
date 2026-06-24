#!/usr/bin/env bash
set -euo pipefail
cd /home/tomato/5TD
labelme data/annotation_batches/rgb_keyframes_2026-06-24_more_keyframes/images \
  --labels data/annotation_batches/rgb_keyframes_2026-06-24_more_keyframes/labels.txt \
  --nodata
