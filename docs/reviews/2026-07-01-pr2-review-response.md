# PR #2 Review Response

Date: 2026-07-01
PR: `codex/new-video-multitask-training`

## Positioning

PR #2 should be reviewed as a new-video multitask semantic-segmentation data loop and tooling workflow. It should not be treated as a production-ready live-driving safety perception stack.

The PR is valuable because it provides the next data iteration path:

- keyframe extraction from the new `Videos/` recordings;
- Labelme batch generation and annotation helpers;
- multitask segmentation dataset preparation;
- passable, boundary, and obstacle training entry points;
- fused offline video evaluation;
- dataset and training validation gates;
- review documentation for the next perception iteration.

It should remain draft while training outputs, obstacle architecture, and live-driving readiness are still experimental.

## Accepted Adjustments

1. **Keep PR #2 draft**
   - The branch can continue as the working place for the new-video data loop.
   - It should not claim autonomous-driving safety readiness.

2. **Use polygon-only new-video annotations**
   - All labels in the new video batch should use Labelme polygons.
   - Legacy batches may keep rectangle compatibility for old annotations.
   - Dataset preparation should expose `--strict-polygons` for new-video validation.

3. **Add an `unsafe` fusion output**
   - Keep `hazard` as the union of explicitly dangerous classes.
   - Keep `safe_passable = ego_passable & ~hazard`.
   - Add `unsafe = ~ego_passable | hazard` for planner keep-out semantics.

4. **Use video/session-level splitting**
   - Train, validation, and test splits should be grouped by complete video or session.
   - Adjacent frames from the same video must not be split across train, validation, and holdout test sets.

5. **Gate training on positive label coverage**
   - `right_barrier`, `suspended_object`, `worker`, `construction_vehicle`, and `debris` must have positive samples in both train and validation before producing trusted checkpoints.
   - A dry-run override may exist, but its outputs must not be described as trusted checkpoints.

6. **Keep obstacle segmentation as a baseline**
   - `debris` is well suited to semantic masks.
   - `worker` and `construction_vehicle` need detector and tracker follow-up before safety use.
   - `suspended_object` needs clearance, ROI, and vehicle-height-envelope rules.

7. **Add safety-oriented metrics**
   - IoU and Dice are not enough for safety acceptance.
   - The next evaluator should track false-safe and near-vehicle miss metrics before any live-driving integration.

8. **Separate tooling mergeability from model readiness**
   - Tooling, tests, validation gates, offline evaluators, and docs can move toward merge.
   - Training outputs, final obstacle architecture, detector/tracker integration, and live-driving integration stay experimental.

## Merge Guidance

The first mergeable slice should include:

- `extract_video_keyframes.py`;
- `prepare_multitask_dataset.py`;
- `evaluate_multitask_videos.py`;
- tests for extraction, dataset validation, positive coverage, fusion, and empty-video behavior;
- review and updated-plan documentation;
- strict polygon validation;
- video/session split support;
- positive label coverage gates;
- offline `unsafe` output.

The following should remain draft or experimental:

- any trained obstacle checkpoint from incomplete labels;
- any claim that the new model is production-ready;
- live-driving integration;
- worker and construction-vehicle safety behavior without detector/tracker support;
- raw videos, extracted frames, and run outputs until storage and privacy policy are confirmed.

## Safety Acceptance Direction

Before live-driving integration, the workflow needs metrics that answer whether dangerous pixels are ever marked as safe. The minimum safety metric set should include:

- `false_safe_hazard_rate`;
- `ditch_as_passable_rate`;
- `worker_false_negative_rate`;
- `bottom_center_safe_leakage`;
- `hazard_near_vehicle_miss_rate`;
- `temporal_flicker_rate`;
- `safe_passable_width_min`.

The core safety question is not whether average mask overlap is high. It is whether a hazardous or unknown region can be falsely exposed as `safe_passable`.
