# Jetson Video Model Evaluation Design

Date: 2026-06-28

## Goal

Evaluate the existing fused passable-road segmentation model on all Jetson videos recorded on 2026-06-28.

## Inputs

- Source videos: `jetson-recordings/2026-06-28/**/camera_*.mkv`
- Model pair:
  - Passable/ditch: `runs/passable_ego/passable_ditch_artifact_v3_finetune/best_model.pt`
  - Boundary/wall: `runs/passable_ego/boundary_wall_aux_v2_no_testvideo/best_model.pt`

## Approach

Add a repository-local batch utility that:

1. Discovers the six copied MKV files for the date.
2. Samples frames at 1 FPS by default.
3. Runs the existing fused segmentation helpers on each sampled frame.
4. Writes side-by-side original and prediction overlay images.
5. Records per-frame area ratios for `safe_passable`, `ditch`, `left_barrier`, and `tunnel_wall`.
6. Writes a per-video summary CSV and a contact sheet for quick visual inspection.

This keeps runtime practical while giving enough temporal coverage to judge whether the model generalizes to today's recordings.

## Outputs

Output root: `runs/video_model_eval/2026-06-28/`

- `overlays/<session>/<camera>/<frame>.jpg`
- `frame_metrics.csv`
- `video_summary.csv`
- `contact_sheet.jpg`

## Error Handling

- Fail fast if a checkpoint or video file is missing.
- Skip unreadable sampled frames and record the skipped count in the summary.
- Use CPU only if CUDA is unavailable.

## Verification

- Run the utility against the copied 2026-06-28 videos.
- Confirm `frame_metrics.csv`, `video_summary.csv`, and `contact_sheet.jpg` exist.
- Confirm the sampled overlay count matches the summary count.
- Open or inspect the contact sheet to assess recognition quality.
