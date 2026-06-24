# RGB Keyframe Annotation Batch - More Keyframes

Date: 2026-06-24

This batch contains additional non-demo keyframes extracted from the existing tunnel MP4 files. It intentionally excludes `demo_video.mp4`.

## Launch Labelme

From the repository root:

```bash
labelme data/annotation_batches/rgb_keyframes_2026-06-24_more_keyframes/images \
  --labels data/annotation_batches/rgb_keyframes_2026-06-24_more_keyframes/labels.txt \
  --nodata
```

Save each annotation as a JSON file next to the image. Use the same rules as `data/annotation_batches/rgb_keyframes_2026-06-22/annotation_rules.md`.
