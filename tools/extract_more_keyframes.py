#!/usr/bin/env python3
"""Extract additional non-demo keyframes for Labelme annotation."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2


# ==== TUNABLE PARAMETERS ====
REPO = Path(__file__).resolve().parents[1]
OUT_ROOT = REPO / "data/annotation_batches/rgb_keyframes_2026-06-24_more_keyframes"
OUT_IMG = OUT_ROOT / "images"

VIDEOS = [
    REPO
    / "baselines/vision_obstacle_avoidance_legacy/media/samples/11d80f9597761d8b57be06f702d03861.mp4",
    REPO
    / "baselines/vision_obstacle_avoidance_legacy/media/samples/33534475b7346c374d2076f99995cbe0.mp4",
    REPO
    / "baselines/vision_obstacle_avoidance_legacy/media/samples/8bfe843b17120d212e625d2637d7c7c1.mp4",
    REPO
    / "baselines/vision_obstacle_avoidance_legacy/media/samples/b0c37d56037822b2cc6e37cbcb337640.mp4",
    REPO
    / "baselines/vision_obstacle_avoidance_legacy/media/samples/ccb5acbde9da473c291cda3d3d064245.mp4",
    REPO / "baselines/vision_obstacle_avoidance_legacy/src/vision_obstacle_avoidance/test_video.mp4",
]


# ==== CORE ====
def prefix_for(video: Path) -> str:
    """Return the output filename prefix for one source video."""
    return "test_video" if video.stem == "test_video" else video.stem[:6]


def extract_video(video: Path) -> list[dict]:
    """Extract JPEG frames from one video without resizing."""
    prefix = prefix_for(video)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    first_offset = max(1, int(round(fps)))
    step = max(1, int(round(fps * 2)))
    frame_indices = list(range(first_offset, total, step))

    records = []
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        out_name = f"{prefix}_f{frame_idx:06d}.jpg"
        out_path = OUT_IMG / out_name
        cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
        height, width = frame.shape[:2]
        records.append(
            {
                "file": out_name,
                "source_video": str(video.relative_to(REPO)),
                "source_prefix": prefix,
                "frame_idx": int(frame_idx),
                "timestamp_sec": round(frame_idx / fps, 3),
                "width": int(width),
                "height": int(height),
            }
        )
    cap.release()
    print(f"[OK] {prefix}: extracted {len(records)} frames from {video.name}")
    return records


def write_batch_files(metadata: dict) -> None:
    """Write Labelme helper files for the extracted batch."""
    src_batch = REPO / "data/annotation_batches/rgb_keyframes_2026-06-22"
    shutil.copy2(src_batch / "labels.txt", OUT_ROOT / "labels.txt")

    readme = """# RGB Keyframe Annotation Batch - More Keyframes

Date: 2026-06-24

This batch contains additional non-demo keyframes extracted from the existing tunnel MP4 files. It intentionally excludes `demo_video.mp4`.

## Launch Labelme

From the repository root:

```bash
labelme data/annotation_batches/rgb_keyframes_2026-06-24_more_keyframes/images \\
  --labels data/annotation_batches/rgb_keyframes_2026-06-24_more_keyframes/labels.txt \\
  --nodata
```

Save each annotation as a JSON file next to the image. Use the same rules as `data/annotation_batches/rgb_keyframes_2026-06-22/annotation_rules.md`.
"""
    (OUT_ROOT / "README.md").write_text(readme, encoding="utf-8")

    launch = """#!/usr/bin/env bash
set -euo pipefail
cd /home/tomato/5TD
labelme data/annotation_batches/rgb_keyframes_2026-06-24_more_keyframes/images \\
  --labels data/annotation_batches/rgb_keyframes_2026-06-24_more_keyframes/labels.txt \\
  --nodata
"""
    launch_path = OUT_ROOT / "launch_labelme.sh"
    launch_path.write_text(launch, encoding="utf-8")
    launch_path.chmod(0o755)

    desktop = """[Desktop Entry]
Type=Application
Name=Labelme More RGB Keyframes
Exec=/home/tomato/5TD/data/annotation_batches/rgb_keyframes_2026-06-24_more_keyframes/launch_labelme.sh
Terminal=true
Categories=Development;
"""
    desktop_path = OUT_ROOT / "launch_labelme.desktop"
    desktop_path.write_text(desktop, encoding="utf-8")
    desktop_path.chmod(0o755)

    (OUT_ROOT / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> None:
    """Run keyframe extraction with the tunable parameters above."""
    OUT_IMG.mkdir(parents=True, exist_ok=True)
    metadata = {
        "batch": OUT_ROOT.name,
        "purpose": "Additional non-demo keyframes for manual Labelme annotation.",
        "sampling": "Frames at odd seconds, approximately frame 30, 90, 150... for 30fps videos.",
        "format": "High-quality JPEG, decoded at source video resolution without resizing.",
        "excluded": ["demo_video.mp4"],
        "frames": [],
    }

    for video in VIDEOS:
        if not video.exists():
            raise FileNotFoundError(video)
        metadata["frames"].extend(extract_video(video))

    write_batch_files(metadata)
    print(f"[OK] Total extracted: {len(metadata['frames'])}")
    print(f"[OK] Output batch: {OUT_ROOT}")


# ==== TEST ====
if __name__ == "__main__":
    main()
