# 5TD Work Progress Review Brief

Date: 2026-07-01
Repository: `SolanumLycopersicumX/5TD`

## Executive Summary

The project has moved from a passable-road prototype into a staged semantic-segmentation workflow for tunnel/off-road safety perception. The current best stable perception stack remains the older staged fusion path:

- passable/ditch checkpoint: `runs/passable_ego/passable_ditch_artifact_v3_finetune/best_model.pt`
- boundary/wall checkpoint: `runs/passable_ego/boundary_wall_aux_v2_no_testvideo/best_model.pt`
- fused output: `safe_passable = ego_passable & ~(ditch | left_barrier | tunnel_wall)`

A new video-training branch has now been implemented for an expanded multitask setup:

- passable model: `ego_passable`, `ditch`
- boundary model: `left_barrier`, `right_barrier`, `tunnel_wall`
- obstacle model: `worker`, `construction_vehicle`, `suspended_object`, `debris`
- fused hazard output: `hazard = ditch | left_barrier | right_barrier | tunnel_wall | worker | construction_vehicle | suspended_object | debris`
- fused safe output: `safe_passable = ego_passable & ~hazard`

The new code and tests have been pushed to GitHub in draft PR #2. Raw videos and run outputs were not committed to normal Git history because they are large and contain several files above GitHub's 100MB normal Git limit.

## GitHub Upload Status

Uploaded branches and pull requests:

- PR #1: `codex/lan-file-server -> main`
  - URL: `https://github.com/SolanumLycopersicumX/5TD/pull/1`
  - Current branch also received small 2026-06-28 Jetson evaluation source/docs updates.

- PR #2: `codex/new-video-multitask-training -> codex/lan-file-server`
  - URL: `https://github.com/SolanumLycopersicumX/5TD/pull/2`
  - Status: draft PR
  - Scope: new video keyframe extraction, multitask dataset prep, passable/boundary/obstacle trainers, fused multitask video evaluator, validation gates, and tests.

Not uploaded to ordinary Git:

- `Videos/`: about 2.9G, six `.MOV` files; largest is about 994MiB.
- `jetson-recordings/`: about 1.1G, six `.mkv` camera videos plus logs/metadata.
- `runs/video_model_eval/`: about 64M, generated overlays/CSVs/contact sheet/report.
- `data/annotation_batches/rgb_keyframes_2026-06-29_videos/images/`: extracted annotation images, about 98M for the batch.

Reason: ordinary GitHub pushes reject files over 100MB. Existing LFS tracks `*.mov` and `*.mkv`, but the source phone videos use uppercase `.MOV`, which is not covered by the current LFS rule. Large data should be uploaded only after confirming privacy and storage strategy, preferably through Git LFS, release assets, or external storage.

## Completed Work Timeline

### 2026-06-22 to 2026-06-24

- Repository reorganization and initial GitHub upload were completed.
- First Labelme annotation batches were prepared and validated.
- Passable-road segmentation progressed through:
  - `ego_passable`
  - `ego_passable + ditch`
  - `ego_passable + ditch + surface_artifact_passable`
- Current strongest passable checkpoint recorded in logs:
  - `safe_iou=0.9730`
  - `ditch_as_passable_rate=0.00024`
- Boundary/wall auxiliary model was added for `left_barrier` and `tunnel_wall`.
- Staged fusion and post-processing improved safe-passable output, with recorded `safe_passable_iou=0.9605`.

### 2026-06-25

- Live webcam fused preview was added.
- Vision-gated forward driving and trajectory-following prototypes were added.
- Defaults remain conservative: visualization/dry-run or zero-speed unless driver enable flags are explicit.

### 2026-06-28

- Jetson recordings were evaluated with the existing fused passable/boundary model.
- New utility added locally and uploaded with PR #1 updates:
  - `tools/passable_segmentation/evaluate_recorded_videos.py`
  - `tests/test_evaluate_recorded_videos.py`
  - 2026-06-28 design and plan docs.
- Local evaluation output summary:
  - six camera videos evaluated
  - 1 FPS sampling
  - 1098 sampled overlay frames generated
  - `frame_metrics.csv`, `video_summary.csv`, `contact_sheet.jpg`, and local `report.md` created
- Key 6/28 evaluation result from local report:

| Session | Camera | Samples | Safe passable | Ditch | Left barrier | Tunnel wall |
|---|---:|---:|---:|---:|---:|---:|
| 2026-06-28_14-19-48 | camera_0 | 312 | 28.03% | 0.70% | 0.34% | 0.01% |
| 2026-06-28_14-19-48 | camera_1 | 312 | 26.22% | 0.69% | 0.39% | 0.03% |
| 2026-06-28_14-25-10 | camera_0 | 50 | 0.18% | 0.00% | 0.01% | 0.00% |
| 2026-06-28_14-25-10 | camera_1 | 50 | 0.09% | 0.00% | 0.01% | 0.00% |
| 2026-06-28_14-26-14 | camera_0 | 187 | 4.58% | 0.03% | 0.01% | 0.17% |
| 2026-06-28_14-26-14 | camera_1 | 187 | 4.74% | 0.03% | 0.02% | 0.03% |

Interpretation: the existing model finds stable safe-passable regions in the first session, but two later sessions show near-zero or low safe-passable area. These are likely domain shift, framing, exposure, or limited visible drivable-region cases and motivated the new video annotation/training loop.

### 2026-06-29 to 2026-07-01

- Six new phone videos in `Videos/` were inspected:
  - total size about 2.9G
  - total duration about 24 minutes
  - all 1920x1080
- Keyframes were extracted to:
  - `data/annotation_batches/rgb_keyframes_2026-06-29_videos`
  - 6 videos x 40 frames each = 240 frames
  - extraction now samples across the full video span instead of only the beginning
- Labelme helper files were generated:
  - `labels.txt`
  - `annotation_rules.md`
  - `launch_labelme.sh`
  - `metadata.json`
  - `contact_sheet.jpg`
- Annotation guidance was updated for semantic segmentation quality:
  - all labels should now be annotated with polygons
  - rectangles are no longer recommended because boxed-in background becomes mask noise

## New Multitask Branch Contents

PR #2 adds:

- `tools/passable_segmentation/extract_video_keyframes.py`
  - discovers supported videos
  - extracts stable keyframes
  - writes Labelme batch helper files
  - uses all-label polygon annotation guidance

- `tools/passable_segmentation/prepare_multitask_dataset.py`
  - reads Labelme JSON and source images
  - validates labels and shape types
  - validates JSON image dimensions against actual image size
  - rasterizes masks for all labels
  - writes passable, boundary, and obstacle manifests
  - writes per-label positive image/pixel counts

- `tools/passable_segmentation/train_passable_ditch_artifact_videos.py`
  - wrapper for fine-tuning passable/ditch using the combined dataset

- `tools/passable_segmentation/train_boundary_right_wall.py`
  - trains `left_barrier`, `right_barrier`, `tunnel_wall`
  - preserves left/right semantics during horizontal flips
  - supports label-aware warm start from older boundary checkpoint
  - blocks training if any target class has zero positive pixels unless explicitly overridden

- `tools/passable_segmentation/train_obstacle_semantic.py`
  - trains `worker`, `construction_vehicle`, `suspended_object`, `debris`
  - uses sparse-friendly per-class dice behavior
  - blocks training if any target class has zero positive pixels unless explicitly overridden

- `tools/passable_segmentation/evaluate_multitask_videos.py`
  - loads passable, boundary, and obstacle checkpoints
  - fuses independent masks into `hazard` and `safe_passable`
  - writes per-frame and per-video metrics plus overlays/contact sheet

## Validation Performed

For PR #2:

- Full test suite passed:
  - `PYTHONPATH=. /home/tomato/miniconda3/envs/lerobot/bin/python -m unittest discover -s tests -p 'test_*' -v`
  - result: 157 tests OK
- Focused keyframe test passed:
  - `tests/test_video_keyframe_extraction.py`: 13 tests OK
- Syntax checks passed for edited key scripts.
- Dataset-prep smoke against current existing annotations succeeded.

Important dataset-prep finding from current annotations:

- `right_barrier`: zero positive pixels in existing prepared data
- `suspended_object`: zero positive pixels in existing prepared data
- `worker`: appears only in train split in current smoke
- `construction_vehicle`: appears only in val split in current smoke

The new training gates intentionally block training in these conditions so a checkpoint is not produced from missing or split-broken class supervision.

## Current Blocking Item

The next actual blocker is manual Labelme annotation of the 240 extracted frames.

Current annotation rules:

- All labels use polygon shapes.
- Draw only visible class pixels.
- Do not draw `hazard`; it is generated automatically.
- `hazard = ditch | left_barrier | right_barrier | tunnel_wall | worker | construction_vehicle | suspended_object | debris`.
- `safe_passable = ego_passable & ~hazard`.
- `surface_artifact_passable` is not a hazard; use it only for passable visual artifacts on traversable ground.

Special cases:

- Gantry frame / portal frame:
  - side posts and bases near road edges: `left_barrier` or `right_barrier`
  - overhead crossbeam or hanging part: `suspended_object`
- `debris`: polygon around visible irregular material or obstacle region.
- `worker` and `construction_vehicle`: polygon around visible body/vehicle pixels, not broad boxes.

## Open Questions For ChatGPT Review

1. Is the three-model split optimal, or should obstacle classes use detection/instance segmentation plus tracker instead of semantic segmentation?
2. Should `worker` and `construction_vehicle` be trained as semantic masks, object detectors, or both?
3. Is the proposed validation strategy strong enough: keep at least one new video/session prefix as holdout, and require per-class positive coverage in both train and val?
4. Should `right_barrier` be a separate learned class or derived by geometry/post-processing from generic barrier predictions?
5. Should `surface_artifact_passable` remain an auxiliary passable correction label, or be folded into normal `ego_passable` after enough examples?
6. What is the best data-upload strategy for raw videos: Git LFS, release assets, DVC, object storage, or not uploading raw data at all?
7. What safety metrics should be prioritized beyond IoU: false-safe hazard rate, bottom-center ROI hazard leakage, temporal consistency, and emergency-stop false negative rate?

## Recommended Next Steps

1. Finish polygon Labelme annotation for the 240 extracted frames.
2. Ensure `right_barrier`, `suspended_object`, `worker`, and `construction_vehicle` have positive examples in both train and val splits.
3. Run `prepare_multitask_dataset.py` and inspect `summary.json` positive counts.
4. Train in this order:
   - passable fine-tune
   - boundary right-wall model
   - obstacle semantic model
5. Run `evaluate_multitask_videos.py` on the new videos.
6. Review fused overlays and hazard/safe-passable ratios before considering any live-driving integration.
7. Decide a data-sharing plan before uploading raw videos or extracted frames to GitHub.
