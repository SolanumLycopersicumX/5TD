# Jetson Video Model Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the existing fused passable-road model on the copied 2026-06-28 Jetson videos and produce visual and CSV evidence of recognition quality.

**Architecture:** Add one focused batch CLI under `tools/passable_segmentation/`. It reuses the existing model loading, prediction, fusion, and overlay helpers from `visualize_fused_passable_boundary.py`, and owns only video discovery, frame sampling, metrics, contact sheet generation, and CLI output.

**Tech Stack:** Python 3.10, OpenCV, PyTorch, existing `tools.passable_segmentation` helpers, CSV output.

---

### Task 1: Add Batch Video Evaluation Utility

**Files:**
- Create: `tools/passable_segmentation/evaluate_recorded_videos.py`
- Test manually with: `/home/tomato/miniconda3/envs/lerobot/bin/python tools/passable_segmentation/evaluate_recorded_videos.py --video-root jetson-recordings/2026-06-28 --output-dir runs/video_model_eval/2026-06-28 --sample-fps 1`

- [ ] **Step 1: Create the CLI and helper functions**

Implement:
- `discover_videos(video_root: Path) -> list[Path]`
- `sample_frame_indices(total_frames: int, source_fps: float, sample_fps: float) -> list[int]`
- `mask_ratios(fused: dict[str, np.ndarray]) -> dict[str, float]`
- `evaluate_video(...) -> VideoSummary`
- `write_contact_sheet(...) -> None`
- `main() -> None`

Use the existing `load_model`, `predict_probabilities`, `fuse_passable_boundary_predictions`, and `make_fused_overlay` helpers.

- [ ] **Step 2: Run a smoke import**

Run:

```bash
/home/tomato/miniconda3/envs/lerobot/bin/python -m py_compile tools/passable_segmentation/evaluate_recorded_videos.py
```

Expected: exit code 0.

- [ ] **Step 3: Run the utility on the copied videos**

Run:

```bash
/home/tomato/miniconda3/envs/lerobot/bin/python tools/passable_segmentation/evaluate_recorded_videos.py \
  --video-root jetson-recordings/2026-06-28 \
  --output-dir runs/video_model_eval/2026-06-28 \
  --sample-fps 1
```

Expected: writes overlays, `frame_metrics.csv`, `video_summary.csv`, and `contact_sheet.jpg`.

### Task 2: Verify Outputs

**Files:**
- Read: `runs/video_model_eval/2026-06-28/frame_metrics.csv`
- Read: `runs/video_model_eval/2026-06-28/video_summary.csv`
- Read: `runs/video_model_eval/2026-06-28/contact_sheet.jpg`

- [ ] **Step 1: Count overlays**

Run:

```bash
find runs/video_model_eval/2026-06-28/overlays -type f -name '*.jpg' | wc -l
```

Expected: matches the number of rows in `frame_metrics.csv` excluding the header.

- [ ] **Step 2: Inspect summary**

Run:

```bash
column -s, -t runs/video_model_eval/2026-06-28/video_summary.csv
```

Expected: six video rows with nonzero sampled frame counts.

- [ ] **Step 3: Inspect contact sheet**

Open `runs/video_model_eval/2026-06-28/contact_sheet.jpg` or inspect it with the image viewer.

Expected: shows representative side-by-side original and fused prediction overlays.
