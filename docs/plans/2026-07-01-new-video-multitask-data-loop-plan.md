# New Video Multitask Data Loop Landing Plan

Date: 2026-07-01
Branch: `codex/new-video-multitask-training`

## Goal

Land PR #2 as a reliable new-video multitask semantic-segmentation data loop and tooling workflow, while keeping training outputs, final obstacle safety behavior, and live-driving integration experimental.

## Current Decision

PR #2 is not the final safety perception stack. It is the toolchain needed to produce, validate, train, and review the next data iteration.

The working target is:

```text
new videos -> polygon Labelme annotations -> strict dataset validation
-> passable/boundary/obstacle training entry points
-> fused offline evaluator -> safety-oriented review
```

## Phase 1: Documentation and Scope Control

- [x] Add the PR review response document at `docs/reviews/2026-07-01-pr2-review-response.md`.
- [x] Add this updated landing plan at `docs/plans/2026-07-01-new-video-multitask-data-loop-plan.md`.
- [x] Mark the original 2026-06-29 design and implementation plan as updated or partially superseded.
- [x] Keep PR #2 draft until the dataset and validation gates below are complete.

## Phase 2: Polygon-Only New Video Validation

- [ ] Add `--strict-polygons` to `tools/passable_segmentation/prepare_multitask_dataset.py`.
- [ ] In strict mode, reject any non-polygon shape for the new video batch labels: `ego_passable`, `ditch`, `left_barrier`, `right_barrier`, `tunnel_wall`, `worker`, `construction_vehicle`, `suspended_object`, `debris`, and `surface_artifact_passable`.
- [ ] Preserve rectangle compatibility for legacy annotation batches when strict mode is not enabled.
- [ ] Add tests that confirm strict mode rejects rectangle annotations while legacy mode still accepts them.

## Phase 3: Video/Session-Level Splits

- [ ] Replace implicit prefix-only validation behavior with explicit video/session split metadata.
- [ ] Record the split policy in `summary.json`:

```json
{
  "split_policy": "video_session_level",
  "train_sessions": [],
  "val_sessions": [],
  "test_sessions": []
}
```

- [ ] Ensure no single video or session contributes adjacent frames to more than one split.
- [ ] Keep at least one complete video or session as a holdout test set when enough labeled coverage exists.

## Phase 4: Positive Coverage Gates

- [ ] Require train and validation positive samples for `right_barrier`, `suspended_object`, `worker`, `construction_vehicle`, and `debris`.
- [ ] Check both positive image counts and positive pixel counts.
- [ ] Block trusted training by default when a required label has zero positives in train or validation.
- [ ] Allow an explicit dry-run override such as `--allow-zero-positive-labels`, but mark outputs as untrusted.

## Phase 5: Fusion Contract Update

Keep the current class-preserving fusion:

```text
hazard =
    ditch
  | left_barrier
  | right_barrier
  | tunnel_wall
  | worker
  | construction_vehicle
  | suspended_object
  | debris

safe_passable = ego_passable & ~hazard
```

Add:

```text
unsafe = ~ego_passable | hazard
```

- [ ] Include `unsafe` in fused evaluator outputs and metrics.
- [ ] Treat `unsafe` as the planner keep-out or high-risk mask.
- [ ] Treat `safe_passable` as the only traversable candidate mask.

## Phase 6: Safety-Oriented Offline Metrics

- [ ] Add evaluator support for `false_safe_hazard_rate`.
- [ ] Preserve and report `ditch_as_passable_rate` for ditch leakage.
- [ ] Add worker-focused false-negative accounting once labeled validation frames exist.
- [ ] Add bottom-center ROI leakage checks for the immediate forward driving region.
- [ ] Add near-vehicle hazard miss checks for high-risk close-range regions.
- [ ] Add temporal flicker checks across consecutive sampled frames.
- [ ] Add a minimum safe-passable corridor-width check before considering navigation use.

## Phase 7: Obstacle Architecture Follow-Up

- [ ] Keep semantic obstacle masks as the baseline.
- [ ] Treat `debris` as primarily mask-based.
- [ ] Add a future worker detector, tracker, temporal memory, and stop-zone rule.
- [ ] Add a future construction-vehicle detector or instance segmentation path with BEV projection and occupancy inflation.
- [ ] Add suspended-object clearance and vehicle-height-envelope rules before using it for final safety decisions.

## Phase 8: Training and Review Order

Training should not start until the annotation and split gates pass.

Recommended order:

1. Complete new-video polygon annotations.
2. Run strict dataset preparation.
3. Inspect `summary.json`, `label_positive_counts`, split sessions, empty masks, and `surface_artifact_outside_ego`.
4. Fine-tune passable/ditch.
5. Train boundary/right-wall.
6. Train obstacle semantic baseline.
7. Run fused offline evaluation with `hazard`, `unsafe`, and `safe_passable`.
8. Review overlays, contact sheets, safety metrics, and holdout performance.
9. Revisit live-driving integration only after safety metrics are credible.

## Live-Driving Gate

Live-driving integration remains out of scope for this PR until:

- video/session-level holdout passes;
- key safety metrics are reported and acceptable;
- worker detector/tracker has at least a baseline;
- right-side ditch and boundary false-safe leakage are controlled;
- `unsafe` is available to the planner;
- testing remains low-speed with manual takeover and emergency stop.
