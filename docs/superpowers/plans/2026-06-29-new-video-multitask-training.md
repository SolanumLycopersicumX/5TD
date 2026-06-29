# New Video Multitask Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the six `Videos/` recordings into a Labelme batch, combined derived masks, three trainable model families, and a fused evaluator that outputs individual classes plus `hazard` and `safe_passable`.

**Architecture:** Keep the current passable model path intact, add focused scripts around it, and create new boundary-right-wall and obstacle semantic trainers instead of rewriting the existing experiments. The combined dataset writer owns label validation and writes model-specific TSV views so each trainer can stay small and explicit.

**Tech Stack:** Python 3.10, OpenCV, Pillow, NumPy, PyTorch, existing `tools.passable_segmentation` helpers, `unittest` tests, Labelme JSON.

---

## File Map

- Create `tools/passable_segmentation/extract_video_keyframes.py`: discover `.MOV` videos, sample keyframes, write Labelme helper files, metadata, and a contact sheet.
- Create `tools/passable_segmentation/prepare_multitask_dataset.py`: combine old and new Labelme batches, validate labels, rasterize masks, and write full plus model-view manifests.
- Create `tools/passable_segmentation/train_passable_ditch_artifact_videos.py`: call the existing passable/artifact trainer with the new model-view dataset and a new run directory.
- Create `tools/passable_segmentation/train_boundary_right_wall.py`: train `left_barrier`, `right_barrier`, `tunnel_wall` with partial initialization from the existing two-output boundary checkpoint.
- Create `tools/passable_segmentation/train_obstacle_semantic.py`: train `worker`, `construction_vehicle`, `suspended_object`, `debris` as semantic masks.
- Create `tools/passable_segmentation/evaluate_multitask_videos.py`: run passable, boundary, and obstacle models on `Videos/`, then write overlays and CSV metrics including `hazard` and `safe_passable`.
- Create `tests/test_video_keyframe_extraction.py`: test video discovery, sampling, and deterministic filename prefixes.
- Create `tests/test_prepare_multitask_dataset.py`: test label validation and rectangle/polygon mask output.
- Create `tests/test_boundary_right_wall_training.py`: test manifest reading and partial checkpoint loading.
- Create `tests/test_obstacle_semantic_training.py`: test obstacle manifest reading and union metrics.
- Create `tests/test_multitask_fusion.py`: test `hazard` union, `safe_passable` subtraction, and metrics labels.

---

### Task 1: Add New Video Keyframe Extraction Utility

**Files:**
- Create: `tools/passable_segmentation/extract_video_keyframes.py`
- Create: `tests/test_video_keyframe_extraction.py`

- [ ] **Step 1: Write failing tests for discovery and sampling**

Add this test file:

```python
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.passable_segmentation.extract_video_keyframes import (
    discover_videos,
    sample_frame_indices,
    video_prefix,
)


class VideoKeyframeExtractionTest(unittest.TestCase):
    def test_discover_videos_finds_mov_mp4_and_mkv_in_stable_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "IMG_3197.MOV").write_bytes(b"video")
            (root / "b.mp4").write_bytes(b"video")
            (root / "a.mkv").write_bytes(b"video")
            (root / "notes.txt").write_text("ignore", encoding="utf-8")

            self.assertEqual(
                [p.name for p in discover_videos(root)],
                ["IMG_3197.MOV", "a.mkv", "b.mp4"],
            )

    def test_video_prefix_preserves_camera_filename_stem(self) -> None:
        self.assertEqual(video_prefix(Path("Videos/IMG_3161.MOV")), "IMG_3161")

    def test_sample_frame_indices_limits_uniform_samples(self) -> None:
        self.assertEqual(
            sample_frame_indices(total_frames=300, source_fps=30.0, sample_seconds=2.0, max_frames=4),
            [0, 60, 120, 180],
        )

    def test_sample_frame_indices_handles_unknown_fps(self) -> None:
        self.assertEqual(
            sample_frame_indices(total_frames=90, source_fps=0.0, sample_seconds=1.0, max_frames=10),
            [0, 30, 60],
        )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests and verify they fail because the module does not exist**

Run:

```bash
python -m unittest tests.test_video_keyframe_extraction -v
```

Expected: `ModuleNotFoundError: No module named 'tools.passable_segmentation.extract_video_keyframes'`.

- [ ] **Step 3: Implement the keyframe utility**

Create `tools/passable_segmentation/extract_video_keyframes.py` with these public functions and CLI:

```python
#!/usr/bin/env python3
"""Extract Labelme keyframes from newly collected tunnel videos."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


DEFAULT_VIDEO_ROOT = Path("Videos")
DEFAULT_OUTPUT_ROOT = Path("data/annotation_batches/rgb_keyframes_2026-06-29_videos")
DEFAULT_LABELS_PATH = Path("data/annotation_batches/rgb_keyframes_2026-06-22/labels.txt")
VIDEO_SUFFIXES = {".mov", ".mp4", ".mkv", ".avi", ".h264", ".mjpeg"}


def discover_videos(video_root: Path | str) -> list[Path]:
    root = Path(video_root)
    return sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES)


def video_prefix(video_path: Path | str) -> str:
    return Path(video_path).stem


def sample_frame_indices(
    total_frames: int,
    source_fps: float,
    sample_seconds: float,
    max_frames: int,
) -> list[int]:
    if total_frames <= 0 or max_frames <= 0:
        return []
    fps = source_fps if source_fps > 0 else 30.0
    step = max(1, int(round(fps * sample_seconds)))
    indices = list(range(0, total_frames, step))
    return indices[:max_frames]


def extract_keyframes(
    *,
    video_root: Path | str = DEFAULT_VIDEO_ROOT,
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
    labels_path: Path | str = DEFAULT_LABELS_PATH,
    sample_seconds: float = 2.0,
    max_frames_per_video: int = 90,
    jpeg_quality: int = 95,
) -> dict[str, Any]:
    video_root = Path(video_root)
    output_root = Path(output_root)
    image_dir = output_root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    labels_path = Path(labels_path)
    shutil.copy2(labels_path, output_root / "labels.txt")

    metadata: dict[str, Any] = {
        "batch": output_root.name,
        "source_video_root": str(video_root),
        "sample_seconds": sample_seconds,
        "max_frames_per_video": max_frames_per_video,
        "frames": [],
    }

    for video in discover_videos(video_root):
        records = extract_video(
            video_path=video,
            image_dir=image_dir,
            sample_seconds=sample_seconds,
            max_frames=max_frames_per_video,
            jpeg_quality=jpeg_quality,
        )
        metadata["frames"].extend(records)

    write_batch_files(output_root)
    write_contact_sheet(output_root / "contact_sheet.jpg", [image_dir / row["file"] for row in metadata["frames"]])
    (output_root / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return metadata


def extract_video(
    *,
    video_path: Path,
    image_dir: Path,
    sample_seconds: float,
    max_frames: int,
    jpeg_quality: int,
) -> list[dict[str, Any]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    indices = sample_frame_indices(total, fps, sample_seconds, max_frames)
    prefix = video_prefix(video_path)
    rows: list[dict[str, Any]] = []
    for frame_idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        out_name = f"{prefix}_f{frame_idx:06d}.jpg"
        out_path = image_dir / out_name
        cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)])
        height, width = frame.shape[:2]
        rows.append(
            {
                "file": out_name,
                "source_video": str(video_path),
                "frame_idx": int(frame_idx),
                "timestamp_sec": round(frame_idx / fps, 3) if fps > 0 else 0.0,
                "width": int(width),
                "height": int(height),
            }
        )
    cap.release()
    return rows


def write_batch_files(output_root: Path) -> None:
    readme = """# RGB Keyframe Annotation Batch - 2026-06-29 Videos

Launch Labelme from the repository root:

```bash
labelme data/annotation_batches/rgb_keyframes_2026-06-29_videos/images \\
  --labels data/annotation_batches/rgb_keyframes_2026-06-29_videos/labels.txt \\
  --nodata
```

Save each annotation JSON next to its image.
"""
    rules = """# Annotation Rules

Use `ego_passable`, `ditch`, `left_barrier`, `right_barrier`, `tunnel_wall`, `worker`, `construction_vehicle`, `suspended_object`, `debris`, and `surface_artifact_passable`.

`worker`, `construction_vehicle`, `suspended_object`, and `debris` may use rectangles. Irregular debris may use polygons.

`surface_artifact_passable` must lie inside `ego_passable` and is not a hazard label.
"""
    launch = """#!/usr/bin/env bash
set -euo pipefail
cd /home/tomato/5TD
labelme data/annotation_batches/rgb_keyframes_2026-06-29_videos/images \\
  --labels data/annotation_batches/rgb_keyframes_2026-06-29_videos/labels.txt \\
  --nodata
"""
    desktop = """[Desktop Entry]
Type=Application
Name=Labelme 2026-06-29 Videos
Exec=/home/tomato/5TD/data/annotation_batches/rgb_keyframes_2026-06-29_videos/launch_labelme.sh
Terminal=true
Categories=Development;
"""
    (output_root / "README.md").write_text(readme, encoding="utf-8")
    (output_root / "annotation_rules.md").write_text(rules, encoding="utf-8")
    launch_path = output_root / "launch_labelme.sh"
    launch_path.write_text(launch, encoding="utf-8")
    launch_path.chmod(0o755)
    desktop_path = output_root / "launch_labelme.desktop"
    desktop_path.write_text(desktop, encoding="utf-8")
    desktop_path.chmod(0o755)


def write_contact_sheet(path: Path, image_paths: list[Path], *, thumb_width: int = 320, columns: int = 4) -> None:
    thumbs = []
    for image_path in image_paths[: min(len(image_paths), 48)]:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        height, width = image.shape[:2]
        scale = thumb_width / float(width)
        thumbs.append(cv2.resize(image, (thumb_width, max(1, int(round(height * scale))))))
    if not thumbs:
        return
    cell_height = max(thumb.shape[0] for thumb in thumbs)
    rows = int(np.ceil(len(thumbs) / float(columns)))
    sheet = np.full((rows * cell_height, columns * thumb_width, 3), 255, dtype=np.uint8)
    for idx, thumb in enumerate(thumbs):
        row, col = divmod(idx, columns)
        y = row * cell_height
        x = col * thumb_width
        sheet[y : y + thumb.shape[0], x : x + thumb.shape[1]] = thumb
    cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS_PATH)
    parser.add_argument("--sample-seconds", type=float, default=2.0)
    parser.add_argument("--max-frames-per-video", type=int, default=90)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    metadata = extract_keyframes(
        video_root=args.video_root,
        output_root=args.output_root,
        labels_path=args.labels,
        sample_seconds=args.sample_seconds,
        max_frames_per_video=args.max_frames_per_video,
    )
    print(f"[OK] Extracted {len(metadata['frames'])} frames to {args.output_root}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests and commit**

Run:

```bash
python -m unittest tests.test_video_keyframe_extraction -v
```

Expected: all 4 tests pass.

Commit:

```bash
git add tools/passable_segmentation/extract_video_keyframes.py tests/test_video_keyframe_extraction.py
git commit -m "Add new video keyframe extraction"
```

---

### Task 2: Add Combined Multitask Dataset Preparation

**Files:**
- Create: `tools/passable_segmentation/prepare_multitask_dataset.py`
- Create: `tests/test_prepare_multitask_dataset.py`

- [ ] **Step 1: Write failing tests for label validation and mask output**

Add tests covering one polygon and one rectangle:

```python
from pathlib import Path
import json
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image

from tools.passable_segmentation.prepare_multitask_dataset import (
    LABELS,
    prepare_multitask_dataset,
    validate_annotation_labels,
)


class PrepareMultitaskDatasetTest(unittest.TestCase):
    def test_validate_annotation_labels_rejects_unknown_label(self) -> None:
        annotation = {"shapes": [{"label": "unknown_label"}]}
        with self.assertRaisesRegex(ValueError, "unknown Labelme labels"):
            validate_annotation_labels(annotation, allowed_labels=LABELS)

    def test_prepare_multitask_dataset_writes_full_and_view_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            batch = root / "batch" / "images"
            batch.mkdir(parents=True)
            Image.new("RGB", (8, 8), (0, 0, 0)).save(batch / "IMG_1_f000000.jpg")
            annotation = {
                "imageWidth": 8,
                "imageHeight": 8,
                "shapes": [
                    {"label": "ego_passable", "shape_type": "polygon", "points": [[0, 4], [7, 4], [7, 7], [0, 7]]},
                    {"label": "worker", "shape_type": "rectangle", "points": [[1, 1], [3, 3]]},
                ],
            }
            (batch / "IMG_1_f000000.json").write_text(json.dumps(annotation), encoding="utf-8")

            summary = prepare_multitask_dataset(
                image_dirs=[batch],
                output_dir=root / "derived",
                val_prefixes=("IMG_1",),
            )

            self.assertEqual(summary["total"], 1)
            self.assertTrue((root / "derived" / "masks" / "ego_passable" / "IMG_1_f000000.png").exists())
            self.assertTrue((root / "derived" / "masks" / "worker" / "IMG_1_f000000.png").exists())
            self.assertTrue((root / "derived" / "passable" / "val.tsv").exists())
            self.assertTrue((root / "derived" / "boundary" / "val.tsv").exists())
            self.assertTrue((root / "derived" / "obstacle" / "val.tsv").exists())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
python -m unittest tests.test_prepare_multitask_dataset -v
```

Expected: `ModuleNotFoundError: No module named 'tools.passable_segmentation.prepare_multitask_dataset'`.

- [ ] **Step 3: Implement dataset preparation**

Create `tools/passable_segmentation/prepare_multitask_dataset.py` with:

```python
#!/usr/bin/env python3
"""Prepare a combined multitask segmentation dataset from Labelme batches."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Sequence

from PIL import Image

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.passable_segmentation.common import (
    label_prefix,
    load_labelme_json,
    rasterize_labelme_mask,
)


LABELS = (
    "ego_passable",
    "ditch",
    "left_barrier",
    "right_barrier",
    "tunnel_wall",
    "worker",
    "construction_vehicle",
    "suspended_object",
    "debris",
    "surface_artifact_passable",
)
PASSABLE_LABELS = ("ego_passable", "ditch", "surface_artifact_passable")
BOUNDARY_LABELS = ("left_barrier", "right_barrier", "tunnel_wall")
OBSTACLE_LABELS = ("worker", "construction_vehicle", "suspended_object", "debris")
DEFAULT_IMAGE_DIRS = (
    Path("data/annotation_batches/rgb_keyframes_2026-06-22/images"),
    Path("data/annotation_batches/rgb_keyframes_2026-06-24_more_keyframes/images"),
    Path("data/annotation_batches/rgb_keyframes_2026-06-29_videos/images"),
)
DEFAULT_OUTPUT_DIR = Path("data/derived/passable_boundary_obstacle_2026-06-29")


def validate_annotation_labels(annotation: dict, *, allowed_labels: Sequence[str] = LABELS) -> None:
    allowed = set(allowed_labels)
    unknown = sorted({shape.get("label", "") for shape in annotation.get("shapes", [])} - allowed)
    if unknown:
        raise ValueError(f"unknown Labelme labels: {unknown}")


def discover_records(image_dirs: Sequence[Path | str]) -> list[tuple[str, Path, Path, str]]:
    rows: list[tuple[str, Path, Path, str]] = []
    seen: set[str] = set()
    for image_dir in image_dirs:
        root = Path(image_dir)
        if not root.exists():
            continue
        for image_path in sorted(list(root.glob("*.jpg")) + list(root.glob("*.jpeg"))):
            annotation_path = image_path.with_suffix(".json")
            if not annotation_path.exists():
                continue
            stem = image_path.stem
            unique_stem = stem
            if unique_stem in seen:
                unique_stem = f"{root.parent.name}_{stem}"
            seen.add(unique_stem)
            rows.append((unique_stem, image_path, annotation_path, root.parent.name))
    return rows


def prepare_multitask_dataset(
    *,
    image_dirs: Sequence[Path | str] = DEFAULT_IMAGE_DIRS,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    val_prefixes: Sequence[str] = ("IMG_3197", "b0c37d"),
    labels: Sequence[str] = LABELS,
) -> dict:
    output_dir = Path(output_dir)
    out_images = output_dir / "images"
    out_masks = output_dir / "masks"
    out_images.mkdir(parents=True, exist_ok=True)
    for label in labels:
        (out_masks / label).mkdir(parents=True, exist_ok=True)

    records = discover_records(image_dirs)
    rows: list[tuple[str, Path, dict[str, Path], str]] = []
    mask_pixels: dict[str, dict[str, int]] = {label: {} for label in labels}
    for stem, image_path, annotation_path, source_batch in records:
        annotation = load_labelme_json(annotation_path)
        validate_annotation_labels(annotation, allowed_labels=labels)
        image_out = out_images / f"{stem}{image_path.suffix.lower()}"
        shutil.copy2(image_path, image_out)
        masks: dict[str, Path] = {}
        for label in labels:
            mask = rasterize_labelme_mask(annotation, label=label)
            mask_out = out_masks / label / f"{stem}.png"
            Image.fromarray(mask).save(mask_out)
            masks[label] = mask_out
            mask_pixels[label][stem] = int((mask > 0).sum())
        rows.append((stem, image_out, masks, source_batch))

    val_set = set(val_prefixes)
    train_rows = [row for row in rows if label_prefix(row[0]) not in val_set]
    val_rows = [row for row in rows if label_prefix(row[0]) in val_set]

    write_full_manifest(output_dir / "manifest.tsv", rows, output_dir, labels)
    write_full_manifest(output_dir / "train.tsv", train_rows, output_dir, labels)
    write_full_manifest(output_dir / "val.tsv", val_rows, output_dir, labels)
    write_view_manifests(output_dir, train_rows, val_rows)

    summary = {
        "source_image_dirs": [str(Path(p)) for p in image_dirs],
        "output_dir": str(output_dir),
        "labels": list(labels),
        "val_prefixes": list(val_prefixes),
        "total": len(rows),
        "train": len(train_rows),
        "val": len(val_rows),
        "empty_masks": {
            label: sorted(stem for stem, pixels in label_pixels.items() if pixels == 0)
            for label, label_pixels in mask_pixels.items()
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def write_full_manifest(path: Path, rows: Sequence[tuple[str, Path, dict[str, Path], str]], root: Path, labels: Sequence[str]) -> None:
    lines = []
    for stem, image_path, masks, _source_batch in rows:
        parts = [stem, image_path.resolve().relative_to(root.resolve()).as_posix()]
        parts.extend(masks[label].resolve().relative_to(root.resolve()).as_posix() for label in labels)
        lines.append("\t".join(parts) + "\n")
    path.write_text("".join(lines), encoding="utf-8")


def write_view_manifests(
    output_dir: Path,
    train_rows: Sequence[tuple[str, Path, dict[str, Path], str]],
    val_rows: Sequence[tuple[str, Path, dict[str, Path], str]],
) -> None:
    specs = {
        "passable": PASSABLE_LABELS,
        "boundary": BOUNDARY_LABELS,
        "obstacle": OBSTACLE_LABELS,
    }
    for view, labels in specs.items():
        view_dir = output_dir / view
        view_dir.mkdir(parents=True, exist_ok=True)
        write_view_manifest(view_dir / "train.tsv", train_rows, view_dir, labels)
        write_view_manifest(view_dir / "val.tsv", val_rows, view_dir, labels)


def write_view_manifest(path: Path, rows: Sequence[tuple[str, Path, dict[str, Path], str]], root: Path, labels: Sequence[str]) -> None:
    lines = []
    for stem, image_path, masks, _source_batch in rows:
        parts = [stem, relpath(image_path, root)]
        parts.extend(relpath(masks[label], root) for label in labels)
        lines.append("\t".join(parts) + "\n")
    path.write_text("".join(lines), encoding="utf-8")


def relpath(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix() if path.resolve().is_relative_to(root.resolve()) else Path("../", path.resolve().relative_to(root.parent.resolve())).as_posix()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, action="append", dest="image_dirs")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--val-prefix", action="append", dest="val_prefixes", default=[])
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    summary = prepare_multitask_dataset(
        image_dirs=args.image_dirs or DEFAULT_IMAGE_DIRS,
        output_dir=args.output_dir,
        val_prefixes=tuple(args.val_prefixes) or ("IMG_3197", "b0c37d"),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests and commit**

Run:

```bash
python -m unittest tests.test_prepare_multitask_dataset -v
```

Expected: both tests pass.

Commit:

```bash
git add tools/passable_segmentation/prepare_multitask_dataset.py tests/test_prepare_multitask_dataset.py
git commit -m "Add multitask dataset preparation"
```

---

### Task 3: Add Passable Fine-Tune Entry Point for New Videos

**Files:**
- Create: `tools/passable_segmentation/train_passable_ditch_artifact_videos.py`

- [ ] **Step 1: Create wrapper that reuses existing passable trainer**

Create:

```python
#!/usr/bin/env python3
"""Fine-tune passable/ditch/artifact segmentation on the 2026-06-29 video dataset."""
from __future__ import annotations

from tools.passable_segmentation.train_passable_ditch_artifact import build_train_config, run_training


def build_video_train_config() -> dict:
    config = build_train_config()
    config.update(
        {
            "dataset_dir": "data/derived/passable_boundary_obstacle_2026-06-29/passable",
            "run_dir": "runs/passable_ego/passable_ditch_artifact_videos_2026-06-29",
            "init_checkpoint": "runs/passable_ego/passable_ditch_artifact_v3_finetune/best_model.pt",
            "epochs": 50,
            "seed": 41,
        }
    )
    return config


def main() -> None:
    run_training(build_video_train_config())


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run syntax check**

Run:

```bash
python -m py_compile tools/passable_segmentation/train_passable_ditch_artifact_videos.py
```

Expected: exit code 0.

- [ ] **Step 3: Commit**

```bash
git add tools/passable_segmentation/train_passable_ditch_artifact_videos.py
git commit -m "Add passable video fine-tune entrypoint"
```

---

### Task 4: Add Boundary Model With Right Barrier Output

**Files:**
- Create: `tools/passable_segmentation/train_boundary_right_wall.py`
- Create: `tests/test_boundary_right_wall_training.py`

- [ ] **Step 1: Write failing manifest and partial-load tests**

Add:

```python
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from tools.passable_segmentation.train_boundary_right_wall import (
    LABELS,
    copy_compatible_state,
    read_boundary_right_wall_manifest,
)
from tools.passable_segmentation.train_passable import SmallPassableUNet


class BoundaryRightWallTrainingTest(unittest.TestCase):
    def test_read_boundary_right_wall_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "train.tsv"
            manifest.write_text(
                "stem\t../images/a.jpg\t../masks/left/a.png\t../masks/right/a.png\t../masks/wall/a.png\n",
                encoding="utf-8",
            )
            rows = read_boundary_right_wall_manifest(manifest)
            self.assertEqual(rows[0][0], "stem")
            self.assertEqual(len(rows[0][2]), 3)

    def test_copy_compatible_state_expands_two_outputs_to_three(self) -> None:
        source = SmallPassableUNet(base_channels=4, out_channels=2)
        target = SmallPassableUNet(base_channels=4, out_channels=3)
        with torch.no_grad():
            source.out.bias.fill_(0.25)
        copy_compatible_state(target, source.state_dict())
        self.assertTrue(torch.allclose(target.out.bias[:2], torch.tensor([0.25, 0.25])))
        self.assertEqual(LABELS, ("left_barrier", "right_barrier", "tunnel_wall"))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
python -m unittest tests.test_boundary_right_wall_training -v
```

Expected: `ModuleNotFoundError: No module named 'tools.passable_segmentation.train_boundary_right_wall'`.

- [ ] **Step 3: Implement new trainer by adapting the existing boundary pattern**

Create `train_boundary_right_wall.py` with these public definitions:

```python
#!/usr/bin/env python3
"""Train left/right barrier and tunnel-wall segmentation."""
from __future__ import annotations

import csv
import json
import math
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.passable_segmentation.train_passable import (
    IMAGE_SIZE,
    MEAN,
    STD,
    SmallPassableUNet,
    augment_image_and_mask,
    dice_loss,
    resize_pair,
    seed_everything,
)

DATASET_DIR = "data/derived/passable_boundary_obstacle_2026-06-29/boundary"
RUN_DIR = "runs/passable_ego/boundary_wall_right_videos_2026-06-29"
INIT_CHECKPOINT = "runs/passable_ego/boundary_wall_aux_v2_no_testvideo/best_model.pt"
LABELS = ("left_barrier", "right_barrier", "tunnel_wall")
EPOCHS = 70
BATCH_SIZE = 8
LR = 5e-4
WEIGHT_DECAY = 1e-4
BASE_CHANNELS = 16
OVERLAP_WEIGHT = 1.5
CONFUSION_WEIGHT = 2.5
SEED = 43
NUM_WORKERS = 2
OVERLAY_COUNT = 12


def build_train_config() -> dict:
    return {
        "dataset_dir": DATASET_DIR,
        "run_dir": RUN_DIR,
        "init_checkpoint": INIT_CHECKPOINT,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "base_channels": BASE_CHANNELS,
        "overlap_weight": OVERLAP_WEIGHT,
        "confusion_weight": CONFUSION_WEIGHT,
        "seed": SEED,
        "num_workers": NUM_WORKERS,
        "overlay_count": OVERLAY_COUNT,
    }


def read_boundary_right_wall_manifest(path: Path | str) -> list[tuple[str, Path, tuple[Path, Path, Path]]]:
    path = Path(path)
    root = path.parent
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) != 5:
                raise ValueError(f"{path} must have 5 columns: stem, image, left, right, wall")
            stem, image_rel, left_rel, right_rel, wall_rel = row
            rows.append((stem, root / image_rel, (root / left_rel, root / right_rel, root / wall_rel)))
    return rows


class BoundaryRightWallDataset(Dataset):
    def __init__(self, manifest_path: Path | str, *, image_size: tuple[int, int] = IMAGE_SIZE, augment: bool = False, seed: int = 0):
        self.rows = read_boundary_right_wall_manifest(manifest_path)
        self.image_size = image_size
        self.augment = augment
        self.seed = seed

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        stem, image_path, mask_paths = self.rows[idx]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masks = []
        for mask_path in mask_paths:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(mask_path)
            masks.append(mask)
        mask_stack = np.stack(masks, axis=-1)
        rng = random.Random(self.seed + idx * 1009 + random.randint(0, 1_000_000))
        if self.augment:
            image, mask_stack = augment_image_and_mask(image, mask_stack, rng)
        image, mask_stack = resize_pair(image, mask_stack, self.image_size)
        image_f = (image.astype(np.float32) / 255.0 - MEAN) / STD
        mask_f = (mask_stack > 127).astype(np.float32).transpose(2, 0, 1)
        return {"stem": stem, "image": torch.from_numpy(image_f.transpose(2, 0, 1)).float(), "mask": torch.from_numpy(mask_f).float()}


def boundary_right_wall_loss(logits: torch.Tensor, targets: torch.Tensor, *, overlap_weight: float, confusion_weight: float) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    d_loss = dice_loss(logits, targets)
    probs = torch.sigmoid(logits)
    overlap = (probs[:, 0:1] * probs[:, 1:2]).mean() + (probs[:, 0:1] * probs[:, 2:3]).mean() + (probs[:, 1:2] * probs[:, 2:3]).mean()
    confusion = logits.sum() * 0.0
    for pred_idx in range(3):
        for target_idx in range(3):
            if pred_idx == target_idx:
                continue
            confusion = confusion + masked_bce(logits[:, pred_idx : pred_idx + 1], torch.zeros_like(targets[:, target_idx : target_idx + 1]), targets[:, target_idx : target_idx + 1])
    return bce + d_loss + overlap_weight * overlap + confusion_weight * confusion


def masked_bce(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.sum().item() <= 0:
        return logits.sum() * 0.0
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def copy_compatible_state(model: nn.Module, source_state: dict[str, torch.Tensor]) -> None:
    target = model.state_dict()
    for key, value in source_state.items():
        if key in ("out.weight", "out.bias") and key in target and target[key].shape[0] >= value.shape[0]:
            target[key][: value.shape[0]].copy_(value)
        elif key in target and target[key].shape == value.shape:
            target[key].copy_(value)
    model.load_state_dict(target)
```

Complete the file with these function names and responsibilities:

- `_loss_from_config(logits, masks, config) -> torch.Tensor`: calls `boundary_right_wall_loss` with `overlap_weight` and `confusion_weight` from `config`.
- `boundary_right_wall_metrics(logits, targets) -> dict[str, float]`: returns per-label IoU/Dice for all labels plus pairwise confusion rates for each label predicted on another label target.
- `train_one_epoch(model, loader, optimizer, device, config) -> dict[str, float]`: iterates batches, computes `_loss_from_config`, backpropagates, and returns average loss.
- `evaluate(model, loader, device, config) -> dict[str, float]`: computes average loss and averaged `boundary_right_wall_metrics`.
- `make_boundary_right_wall_overlay(image, probs, target=None) -> np.ndarray`: overlays left barrier in cyan, right barrier in blue, and wall in gray.
- `save_overlays(model, loader, device, out_dir, limit) -> None`: writes validation overlays.
- `run_training(config=None) -> dict`: builds loaders from `dataset_dir/train.tsv` and `dataset_dir/val.tsv`, loads compatible checkpoint weights through `copy_compatible_state`, trains, writes `best_model.pt`, `last_model.pt`, `history.json`, and `summary.json`.
- `main() -> None`: calls `run_training()`.
- `_print_json(prefix, data) -> None`: prints JSON summaries line by line.

Use these exact substitutions relative to the existing two-label boundary trainer:

- dataset class with `BoundaryRightWallDataset`
- loss with `boundary_right_wall_loss`
- output channels with `len(LABELS)`
- checkpoint loading with `copy_compatible_state(model, checkpoint.get("model", checkpoint))`

- [ ] **Step 4: Run focused tests and syntax check**

Run:

```bash
python -m unittest tests.test_boundary_right_wall_training -v
python -m py_compile tools/passable_segmentation/train_boundary_right_wall.py
```

Expected: tests pass and syntax check exits 0.

- [ ] **Step 5: Commit**

```bash
git add tools/passable_segmentation/train_boundary_right_wall.py tests/test_boundary_right_wall_training.py
git commit -m "Add right-barrier boundary training"
```

---

### Task 5: Add Obstacle Semantic Trainer

**Files:**
- Create: `tools/passable_segmentation/train_obstacle_semantic.py`
- Create: `tests/test_obstacle_semantic_training.py`

- [ ] **Step 1: Write failing tests for manifest reading and hazard metrics**

Add:

```python
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from tools.passable_segmentation.train_obstacle_semantic import (
    LABELS,
    obstacle_metrics,
    read_obstacle_manifest,
)


class ObstacleSemanticTrainingTest(unittest.TestCase):
    def test_read_obstacle_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "train.tsv"
            manifest.write_text(
                "stem\t../images/a.jpg\t../masks/worker/a.png\t../masks/vehicle/a.png\t../masks/suspended/a.png\t../masks/debris/a.png\n",
                encoding="utf-8",
            )
            rows = read_obstacle_manifest(manifest)
            self.assertEqual(rows[0][0], "stem")
            self.assertEqual(len(rows[0][2]), 4)
            self.assertEqual(LABELS, ("worker", "construction_vehicle", "suspended_object", "debris"))

    def test_obstacle_metrics_reports_union_iou(self) -> None:
        logits = torch.full((1, 4, 2, 2), -8.0)
        logits[:, 0, 0, 0] = 8.0
        targets = torch.zeros((1, 4, 2, 2))
        targets[:, 0, 0, 0] = 1.0
        metrics = obstacle_metrics(logits, targets)
        self.assertAlmostEqual(metrics["obstacle_hazard_iou"], 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
python -m unittest tests.test_obstacle_semantic_training -v
```

Expected: `ModuleNotFoundError: No module named 'tools.passable_segmentation.train_obstacle_semantic'`.

- [ ] **Step 3: Implement obstacle trainer**

Create `train_obstacle_semantic.py` with these constants:

```python
LABELS = ("worker", "construction_vehicle", "suspended_object", "debris")
DATASET_DIR = "data/derived/passable_boundary_obstacle_2026-06-29/obstacle"
RUN_DIR = "runs/passable_ego/obstacle_semantic_videos_2026-06-29"
EPOCHS = 80
BATCH_SIZE = 8
LR = 1e-3
WEIGHT_DECAY = 1e-4
BASE_CHANNELS = 16
OVERLAP_WEIGHT = 1.0
SEED = 47
NUM_WORKERS = 2
OVERLAY_COUNT = 12
```

Use this manifest reader and metrics implementation:

```python
def read_obstacle_manifest(path: Path | str) -> list[tuple[str, Path, tuple[Path, Path, Path, Path]]]:
    path = Path(path)
    root = path.parent
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) != 6:
                raise ValueError(f"{path} must have 6 columns: stem, image, worker, construction_vehicle, suspended_object, debris")
            stem, image_rel, worker_rel, vehicle_rel, suspended_rel, debris_rel = row
            rows.append((stem, root / image_rel, (root / worker_rel, root / vehicle_rel, root / suspended_rel, root / debris_rel)))
    return rows


@torch.no_grad()
def obstacle_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    preds = (torch.sigmoid(logits) > 0.5).float()
    targets = (targets > 0.5).float()
    metrics = {}
    for idx, name in enumerate(LABELS):
        pred = preds[:, idx : idx + 1]
        tgt = targets[:, idx : idx + 1]
        intersection = (pred * tgt).sum().item()
        union = ((pred + tgt) > 0).float().sum().item()
        pred_sum = pred.sum().item()
        tgt_sum = tgt.sum().item()
        metrics[f"{name}_iou"] = float((intersection + 1e-6) / (union + 1e-6))
        metrics[f"{name}_dice"] = float((2 * intersection + 1e-6) / (pred_sum + tgt_sum + 1e-6))
    pred_union = (preds.sum(dim=1, keepdim=True) > 0).float()
    target_union = (targets.sum(dim=1, keepdim=True) > 0).float()
    intersection = (pred_union * target_union).sum().item()
    union = ((pred_union + target_union) > 0).float().sum().item()
    metrics["obstacle_hazard_iou"] = float((intersection + 1e-6) / (union + 1e-6))
    return metrics
```

For dataset loading, training, evaluation, overlays, and `run_training`, use the same structure as `train_boundary_right_wall.py`, with output channels set to `len(LABELS)`.

- [ ] **Step 4: Run tests and syntax check**

Run:

```bash
python -m unittest tests.test_obstacle_semantic_training -v
python -m py_compile tools/passable_segmentation/train_obstacle_semantic.py
```

Expected: tests pass and syntax check exits 0.

- [ ] **Step 5: Commit**

```bash
git add tools/passable_segmentation/train_obstacle_semantic.py tests/test_obstacle_semantic_training.py
git commit -m "Add obstacle semantic training"
```

---

### Task 6: Add Multitask Fusion and Video Evaluation

**Files:**
- Create: `tools/passable_segmentation/evaluate_multitask_videos.py`
- Create: `tests/test_multitask_fusion.py`

- [ ] **Step 1: Write failing fusion tests**

Add:

```python
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from tools.passable_segmentation.evaluate_multitask_videos import (
    FUSED_LABELS,
    fuse_multitask_predictions,
    mask_ratios,
)


class MultitaskFusionTest(unittest.TestCase):
    def test_fuse_multitask_predictions_builds_hazard_and_safe_passable(self) -> None:
        passable_probs = np.zeros((2, 2, 2), dtype=np.float32)
        passable_probs[0] = np.array([[1.0, 1.0], [1.0, 0.0]])
        passable_probs[1] = np.array([[0.0, 1.0], [0.0, 0.0]])
        boundary_probs = np.zeros((3, 2, 2), dtype=np.float32)
        boundary_probs[1] = np.array([[0.0, 0.0], [1.0, 0.0]])
        obstacle_probs = np.zeros((4, 2, 2), dtype=np.float32)
        obstacle_probs[0] = np.array([[1.0, 0.0], [0.0, 0.0]])

        fused = fuse_multitask_predictions(passable_probs, boundary_probs, obstacle_probs)

        self.assertTrue(fused["hazard"][0, 0])
        self.assertTrue(fused["hazard"][0, 1])
        self.assertTrue(fused["hazard"][1, 0])
        self.assertFalse(fused["safe_passable"].any())
        self.assertIn("right_barrier", fused)

    def test_mask_ratios_reports_all_fused_labels(self) -> None:
        fused = {label: np.zeros((2, 2), dtype=bool) for label in FUSED_LABELS}
        ratios = mask_ratios(fused)
        self.assertIn("hazard_ratio", ratios)
        self.assertIn("safe_passable_ratio", ratios)
        self.assertIn("construction_vehicle_ratio", ratios)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
python -m unittest tests.test_multitask_fusion -v
```

Expected: `ModuleNotFoundError: No module named 'tools.passable_segmentation.evaluate_multitask_videos'`.

- [ ] **Step 3: Implement fusion evaluator**

Create `evaluate_multitask_videos.py` by reusing `evaluate_recorded_videos.py` structure. The fusion functions must include:

```python
PASSABLE_LABELS = ("ego_passable", "ditch")
BOUNDARY_LABELS = ("left_barrier", "right_barrier", "tunnel_wall")
OBSTACLE_LABELS = ("worker", "construction_vehicle", "suspended_object", "debris")
FUSED_LABELS = (
    "ego_passable",
    "ditch",
    "left_barrier",
    "right_barrier",
    "tunnel_wall",
    "worker",
    "construction_vehicle",
    "suspended_object",
    "debris",
    "hazard",
    "safe_passable",
)


def fuse_multitask_predictions(
    passable_probs: np.ndarray,
    boundary_probs: np.ndarray,
    obstacle_probs: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, np.ndarray]:
    ego_passable = passable_probs[0] > threshold
    ditch = passable_probs[1] > threshold
    left_barrier = boundary_probs[0] > threshold
    right_barrier = boundary_probs[1] > threshold
    tunnel_wall = boundary_probs[2] > threshold
    worker = obstacle_probs[0] > threshold
    construction_vehicle = obstacle_probs[1] > threshold
    suspended_object = obstacle_probs[2] > threshold
    debris = obstacle_probs[3] > threshold
    hazard = (
        ditch
        | left_barrier
        | right_barrier
        | tunnel_wall
        | worker
        | construction_vehicle
        | suspended_object
        | debris
    )
    safe_passable = ego_passable & ~hazard
    return {
        "ego_passable": ego_passable,
        "ditch": ditch,
        "left_barrier": left_barrier,
        "right_barrier": right_barrier,
        "tunnel_wall": tunnel_wall,
        "worker": worker,
        "construction_vehicle": construction_vehicle,
        "suspended_object": suspended_object,
        "debris": debris,
        "hazard": hazard,
        "safe_passable": safe_passable,
    }


def mask_ratios(fused: dict[str, np.ndarray]) -> dict[str, float]:
    return {
        f"{label}_ratio": float(np.count_nonzero(fused[label]) / fused[label].size) if fused[label].size else 0.0
        for label in FUSED_LABELS
    }
```

The CLI defaults should be:

```python
DEFAULT_VIDEO_ROOT = Path("Videos")
DEFAULT_OUTPUT_DIR = Path("runs/video_model_eval/2026-06-29_multitask")
DEFAULT_PASSABLE_CHECKPOINT = Path("runs/passable_ego/passable_ditch_artifact_videos_2026-06-29/best_model.pt")
DEFAULT_BOUNDARY_CHECKPOINT = Path("runs/passable_ego/boundary_wall_right_videos_2026-06-29/best_model.pt")
DEFAULT_OBSTACLE_CHECKPOINT = Path("runs/passable_ego/obstacle_semantic_videos_2026-06-29/best_model.pt")
VIDEO_SUFFIXES = {".mkv", ".mp4", ".avi", ".mov", ".MOV", ".h264", ".mjpeg"}
```

- [ ] **Step 4: Run tests and syntax check**

Run:

```bash
python -m unittest tests.test_multitask_fusion -v
python -m py_compile tools/passable_segmentation/evaluate_multitask_videos.py
```

Expected: tests pass and syntax check exits 0.

- [ ] **Step 5: Commit**

```bash
git add tools/passable_segmentation/evaluate_multitask_videos.py tests/test_multitask_fusion.py
git commit -m "Add multitask video fusion evaluation"
```

---

### Task 7: Execute the Data and Training Workflow

**Files:**
- Write: `data/annotation_batches/rgb_keyframes_2026-06-29_videos/`
- Write: `data/derived/passable_boundary_obstacle_2026-06-29/`
- Write: `runs/passable_ego/passable_ditch_artifact_videos_2026-06-29/`
- Write: `runs/passable_ego/boundary_wall_right_videos_2026-06-29/`
- Write: `runs/passable_ego/obstacle_semantic_videos_2026-06-29/`
- Write: `runs/video_model_eval/2026-06-29_multitask/`

- [ ] **Step 1: Extract Labelme keyframes**

Run:

```bash
python tools/passable_segmentation/extract_video_keyframes.py \
  --video-root Videos \
  --output-root data/annotation_batches/rgb_keyframes_2026-06-29_videos \
  --sample-seconds 2 \
  --max-frames-per-video 90
```

Expected: `data/annotation_batches/rgb_keyframes_2026-06-29_videos/images/` contains up to 540 JPEGs and `metadata.json` records every extracted frame.

- [ ] **Step 2: User completes Labelme annotations**

Run Labelme:

```bash
labelme data/annotation_batches/rgb_keyframes_2026-06-29_videos/images \
  --labels data/annotation_batches/rgb_keyframes_2026-06-29_videos/labels.txt \
  --nodata
```

Expected: every selected `.jpg` has a sibling `.json` after annotation is complete.

- [ ] **Step 3: Build combined derived dataset**

Run:

```bash
python tools/passable_segmentation/prepare_multitask_dataset.py \
  --image-dir data/annotation_batches/rgb_keyframes_2026-06-22/images \
  --image-dir data/annotation_batches/rgb_keyframes_2026-06-24_more_keyframes/images \
  --image-dir data/annotation_batches/rgb_keyframes_2026-06-29_videos/images \
  --output-dir data/derived/passable_boundary_obstacle_2026-06-29 \
  --val-prefix b0c37d \
  --val-prefix IMG_3197
```

Expected: `summary.json` reports nonzero `train`, nonzero `val`, and mask directories for all ten labels.

- [ ] **Step 4: Train passable model**

Run:

```bash
python tools/passable_segmentation/train_passable_ditch_artifact_videos.py
```

Expected: `runs/passable_ego/passable_ditch_artifact_videos_2026-06-29/best_model.pt` exists and `summary.json` reports `labels` as `["ego_passable", "ditch"]`.

- [ ] **Step 5: Train boundary model**

Run:

```bash
python tools/passable_segmentation/train_boundary_right_wall.py
```

Expected: `runs/passable_ego/boundary_wall_right_videos_2026-06-29/best_model.pt` exists and `summary.json` reports `labels` as `["left_barrier", "right_barrier", "tunnel_wall"]`.

- [ ] **Step 6: Train obstacle model**

Run:

```bash
python tools/passable_segmentation/train_obstacle_semantic.py
```

Expected: `runs/passable_ego/obstacle_semantic_videos_2026-06-29/best_model.pt` exists and `summary.json` includes `obstacle_hazard_iou`.

- [ ] **Step 7: Evaluate fused multitask models on all new videos**

Run:

```bash
python tools/passable_segmentation/evaluate_multitask_videos.py \
  --video-root Videos \
  --output-dir runs/video_model_eval/2026-06-29_multitask \
  --sample-fps 1
```

Expected: `frame_metrics.csv`, `video_summary.csv`, `contact_sheet.jpg`, and overlay images exist. CSV columns include `hazard_ratio`, `safe_passable_ratio`, `worker_ratio`, `construction_vehicle_ratio`, `suspended_object_ratio`, and `debris_ratio`.

- [ ] **Step 8: Run focused regression tests**

Run:

```bash
python -m unittest \
  tests.test_video_keyframe_extraction \
  tests.test_prepare_multitask_dataset \
  tests.test_boundary_right_wall_training \
  tests.test_obstacle_semantic_training \
  tests.test_multitask_fusion \
  tests.test_evaluate_recorded_videos \
  -v
```

Expected: all listed tests pass.

- [ ] **Step 9: Commit generated scripts and tests**

Commit source changes only:

```bash
git add tools/passable_segmentation/extract_video_keyframes.py \
  tools/passable_segmentation/prepare_multitask_dataset.py \
  tools/passable_segmentation/train_passable_ditch_artifact_videos.py \
  tools/passable_segmentation/train_boundary_right_wall.py \
  tools/passable_segmentation/train_obstacle_semantic.py \
  tools/passable_segmentation/evaluate_multitask_videos.py \
  tests/test_video_keyframe_extraction.py \
  tests/test_prepare_multitask_dataset.py \
  tests/test_boundary_right_wall_training.py \
  tests/test_obstacle_semantic_training.py \
  tests/test_multitask_fusion.py
git commit -m "Add multitask video training pipeline"
```

Expected: source and tests are committed separately from large generated data, video frames, and training runs.

## Self-Review

- Spec coverage: the plan covers keyframe extraction, Labelme batch creation, combined derived masks, passable fine-tuning, boundary fine-tuning with `right_barrier`, obstacle training, fused `hazard`, fused `safe_passable`, and video evaluation.
- Scope: the plan does not introduce a detector or change low-level vehicle control.
- Type consistency: the label tuples match the design document: passable uses `ego_passable/ditch`, boundary uses `left_barrier/right_barrier/tunnel_wall`, obstacle uses `worker/construction_vehicle/suspended_object/debris`, and fusion includes every individual class plus `hazard` and `safe_passable`.
