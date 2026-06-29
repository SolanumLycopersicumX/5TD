# New Video Multitask Training Design

Date: 2026-06-29

## Purpose

The new `Videos/` recordings contain substantially more tunnel-scene variation than the earlier keyframe batches. They also include obstacle classes that were listed in the annotation rules but were not part of the active training path: `worker`, `construction_vehicle`, `suspended_object`, and `debris`.

The goal is to train these new scenes into the RGB perception stack without discarding the current working path. The design keeps the existing passable and boundary models, adds an obstacle model for the newly visible object classes, and exposes one navigation-oriented `hazard` mask while preserving each source class for debugging and future behavior rules.

## Current State

The active evaluated perception path is a two-model fused setup:

- `runs/passable_ego/passable_ditch_artifact_v3_finetune/best_model.pt` predicts `ego_passable` and `ditch`, with `surface_artifact_passable` used as training supervision.
- `runs/passable_ego/boundary_wall_aux_v2_no_testvideo/best_model.pt` predicts `left_barrier` and `tunnel_wall`.
- `tools/passable_segmentation/evaluate_recorded_videos.py` fuses the passable and boundary predictions into video overlays and frame metrics.

The existing Labelme rules and `labels.txt` already include the target obstacle labels:

- `worker`
- `construction_vehicle`
- `suspended_object`
- `debris`

The shared Labelme rasterizer in `tools/passable_segmentation/common.py` already supports both polygon and rectangle shapes, so rectangle object labels can be converted into masks without changing the annotation tool.

## Target Architecture

The perception stack should have three independent model families with a fusion layer on top:

1. **Passable model**
   - Outputs: `ego_passable`, `ditch`
   - Training supervision: `ego_passable`, `ditch`, `surface_artifact_passable`
   - Checkpoint lineage: fine-tune from `runs/passable_ego/passable_ditch_artifact_v3_finetune/best_model.pt`

2. **Boundary model**
   - Outputs: `left_barrier`, `right_barrier`, `tunnel_wall`
   - Training supervision: `left_barrier`, `right_barrier`, `tunnel_wall`
   - Checkpoint lineage: fine-tune compatible weights from `runs/passable_ego/boundary_wall_aux_v2_no_testvideo/best_model.pt`; initialize the new `right_barrier` output head safely when old checkpoints only have two outputs.

3. **Obstacle model**
   - Outputs: `worker`, `construction_vehicle`, `suspended_object`, `debris`
   - Training supervision: rectangle or polygon Labelme shapes rasterized into one binary mask per class
   - Checkpoint lineage: train as a new small segmentation model using the same image preprocessing, augmentation, overlay, and metric conventions as the existing passable models.

4. **Fusion layer**
   - Preserves every individual class mask.
   - Builds one navigation-oriented `hazard` mask:

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
```

The fused safe mask is:

```text
safe_passable = ego_passable & ~hazard
```

`surface_artifact_passable` is never part of `hazard`. It is positive supervision that small drivable stains, shallow cracks, shallow pits, and passable surface artifacts should remain within `ego_passable` and should not become `ditch` or obstacle predictions.

## Annotation Workflow

Create a new annotation batch:

```text
data/annotation_batches/rgb_keyframes_2026-06-29_videos/
  images/
  labels.txt
  metadata.json
  README.md
  annotation_rules.md
  launch_labelme.sh
  launch_labelme.desktop
```

The batch should be created from the six videos under `Videos/`. Because the recordings are about 24 minutes total and have many scene changes, the extraction should balance coverage and labeling cost:

- sample candidate frames across all six videos;
- include uniformly spaced frames so each recording contributes coverage;
- include additional scene-change or model-failure frames when current fused predictions show likely misses;
- avoid adjacent near-duplicate frames unless they contain different obstacle states.

The batch should use the existing label set:

```text
ego_passable
ditch
left_barrier
right_barrier
tunnel_wall
worker
construction_vehicle
suspended_object
debris
surface_artifact_passable
```

Annotation rules:

- Mark `ego_passable` as the drivable ground for this UGV, not all visible floor.
- Mark true ditches, channels, and trench edges as `ditch`.
- Mark left and right hard boundaries separately as `left_barrier` and `right_barrier`.
- Mark walls and wall-base non-drivable regions as `tunnel_wall`.
- Mark `worker`, `construction_vehicle`, `suspended_object`, and `debris` with rectangles by default. Polygon is allowed for irregular debris when the exact occupied area matters.
- Mark drivable visual artifacts with `surface_artifact_passable` only when they lie fully inside `ego_passable`.
- Empty object labels are valid for frames with no visible worker, vehicle, debris, or suspended object.

## Derived Datasets

After annotation, build one combined derived dataset from both the old annotated frames and the new video batch. The combined dataset should keep one mask directory per label:

```text
data/derived/passable_boundary_obstacle_2026-06-29/
  images/
  masks/
    ego_passable/
    ditch/
    left_barrier/
    right_barrier/
    tunnel_wall/
    worker/
    construction_vehicle/
    suspended_object/
    debris/
    surface_artifact_passable/
  manifest.tsv
  train.tsv
  val.tsv
  summary.json
```

Validation splitting should hold out at least one new-video prefix/session and keep the old validation coverage. The split should make it possible to detect whether the new obstacle classes work on new video scenes rather than only on older static tunnel examples.

The dataset summary should report:

- total, train, and validation sample counts;
- empty-mask lists per label;
- source batches included;
- validation prefixes or explicit validation stems;
- frame metadata inherited from extraction.

## Training Workflow

Training should be staged so regressions are easy to isolate.

1. **Prepare annotation batch**
   - Extract keyframes from `Videos/`.
   - Generate Labelme launcher files and metadata.
   - Produce a contact sheet for quick review.

2. **After manual annotation, prepare combined masks**
   - Convert Labelme JSON into one binary mask per label.
   - Create `manifest.tsv`, `train.tsv`, `val.tsv`, and `summary.json`.
   - Validate that expected labels appear and that mask dimensions match source images.

3. **Fine-tune passable model**
   - Inputs: `ego_passable`, `ditch`, `surface_artifact_passable`
   - Initial checkpoint: `runs/passable_ego/passable_ditch_artifact_v3_finetune/best_model.pt`
   - Output run directory: `runs/passable_ego/passable_ditch_artifact_videos_2026-06-29`

4. **Fine-tune boundary model**
   - Inputs: `left_barrier`, `right_barrier`, `tunnel_wall`
   - Initial checkpoint: `runs/passable_ego/boundary_wall_aux_v2_no_testvideo/best_model.pt`
   - Output run directory: `runs/passable_ego/boundary_wall_right_videos_2026-06-29`

5. **Train obstacle model**
   - Inputs: `worker`, `construction_vehicle`, `suspended_object`, `debris`
   - Output run directory: `runs/passable_ego/obstacle_semantic_videos_2026-06-29`
   - Metrics should report per-class IoU/Dice and an aggregate `obstacle_hazard_iou` over the union of all four obstacle classes.

6. **Evaluate fused model on all new videos**
   - Run passable, boundary, and obstacle models over `Videos/`.
   - Write overlays, frame metrics, video summaries, and contact sheets.
   - Include per-class ratios and fused `hazard` / `safe_passable` ratios.

## Fusion and Navigation Contract

The fused output dictionary should include:

- `ego_passable`
- `ditch`
- `left_barrier`
- `right_barrier`
- `tunnel_wall`
- `worker`
- `construction_vehicle`
- `suspended_object`
- `debris`
- `hazard`
- `safe_passable`

For navigation, `hazard` is the union of explicitly dangerous or non-crossable semantics. It is not simply every pixel outside `ego_passable`; unknown background remains non-drivable because it is not inside `safe_passable`.

The existing safety gate in `src/tunnel_nav/vision_autodrive.py` can continue to consume a hazard label list, but the fused prediction source should expose `hazard` directly so downstream code does not need to repeat class-union logic.

## Error Handling and Data Validation

The preparation scripts should fail fast when:

- an image has no matching Labelme JSON after the user marks the batch complete;
- a JSON references an unknown label;
- a required mask path is missing from a manifest;
- mask dimensions do not match the source image dimensions before resizing;
- a training script receives a checkpoint with incompatible labels and cannot partially initialize safely.

Warnings are acceptable when:

- an object class has empty masks in many frames;
- a video contributes fewer frames after deduplication;
- `right_barrier` has no old checkpoint head and must start from random initialization.

## Testing and Verification

The implementation should include focused tests for:

- multi-label dataset preparation with rectangle and polygon shapes;
- manifest reading for passable, boundary, and obstacle datasets;
- partial checkpoint loading when expanding `left_barrier/tunnel_wall` to `left_barrier/right_barrier/tunnel_wall`;
- fusion output keys and `hazard = union(class hazards)`;
- `safe_passable = ego_passable & ~hazard`;
- video discovery for generic `.MOV` files under `Videos/`.

Manual verification should include:

- opening the generated contact sheet before annotation;
- checking a few generated masks after annotation;
- reviewing validation overlays from all three model families;
- reviewing fused video overlays for worker, construction vehicle, suspended object, debris, and safe-passable subtraction.

## Non-Goals

This design does not introduce a YOLO or RT-DETR detector in the first implementation. The current repository already has mask-based dataset, training, visualization, and navigation plumbing, so the fastest reliable path is to rasterize object annotations into class masks. The same Labelme JSON files can later be exported to detection format if object mAP, tracking, or speed-policy behavior requires a detector.

This design also does not change the low-level vehicle controller. It only defines the perception data, training, fusion, and evaluation contract that the controller can consume.

## Acceptance Criteria

- A new Labelme batch exists for the `Videos/` recordings with labels, metadata, launcher files, and review material.
- A combined derived dataset can be generated with masks for passable, boundary, obstacle, and artifact labels.
- Passable, boundary, and obstacle model training can run from explicit run directories without overwriting the previous best runs.
- The fused evaluator produces individual class overlays plus `hazard` and `safe_passable`.
- `worker`, `construction_vehicle`, `suspended_object`, and `debris` are represented both as independent classes and as part of the fused `hazard` mask.
- `surface_artifact_passable` remains passable-correction supervision and never becomes part of `hazard`.
