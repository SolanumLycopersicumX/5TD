# RGB Keyframe Annotation Batch

Date: 2026-06-22

This batch contains low-duplicate keyframes extracted from the existing MP4 files for the RGB-only tunnel navigation route.

## Contents

- `images/`: extracted JPEG keyframes for manual annotation.
- `labels.txt`: fixed Labelme label list for this batch.

## Launch Labelme

From the repository root:

```bash
labelme data/annotation_batches/rgb_keyframes_2026-06-22/images \
  --labels data/annotation_batches/rgb_keyframes_2026-06-22/labels.txt \
  --nodata
```

Save each annotation as a JSON file next to the image.

## Labels

Use polygons for masks:

- `ego_passable`: drivable ground on the vehicle side only.
- `ditch`: drainage trench, central ditch, or deep channel boundary.
- `left_barrier`: left curb, barrier, or no-cross boundary.
- `right_barrier`: right curb, rail edge, trench edge, or no-cross boundary.
- `tunnel_wall`: tunnel wall or wall-base no-cross area.

Use rectangles for detection boxes when objects are present:

- `worker`: any person.
- `construction_vehicle`: excavator, loader, truck, or similar vehicle.
- `suspended_object`: hanging object that may enter the vehicle path.
- `debris`: rocks, tools, boxes, cables, motors, or other obstacles.

## Current Video Assumption

Most current frames have no obstacles. For these frames, annotate `ego_passable` and hard boundaries, and leave object boxes empty. Empty detection labels are valid negative samples.

For hard boundaries, mark the visible no-crossing surface or a narrow band along the edge. Do not mark the ground across a ditch as `ego_passable`, even if it appears flat.
