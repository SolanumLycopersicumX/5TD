#!/usr/bin/env python3
"""Extract uniformly sampled video frames for Labelme annotation."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import stat
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageOps


DEFAULT_VIDEO_ROOT = Path("Videos")
DEFAULT_OUTPUT_ROOT = Path("data/annotation_batches/rgb_keyframes_2026-06-29_videos")
DEFAULT_LABELS_PATH = Path("data/annotation_batches/rgb_keyframes_2026-06-22/labels.txt")
VIDEO_SUFFIXES = {".mov", ".mp4", ".mkv", ".avi", ".h264", ".mjpeg"}

ANNOTATION_LABELS = (
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


def discover_videos(video_root: Path | str) -> list[Path]:
    """Return supported video files that are direct children of video_root."""
    root = Path(video_root)
    if not root.exists():
        return []
    return sorted(
        path for path in root.iterdir() if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    )


def video_prefix(video_path: Path | str) -> str:
    """Return the filename stem used as the frame output prefix."""
    return Path(video_path).stem


def video_output_prefix(video_path: Path | str, video_root: Path | str | None = None) -> str:
    """Build a filesystem-safe batch output prefix that avoids same-stem collisions."""
    path = Path(video_path)
    if video_root is None:
        rel = Path(path.name)
    else:
        root = Path(video_root)
        try:
            rel = path.relative_to(root)
        except ValueError:
            rel = Path(path.name)
    rel_text = rel.as_posix()
    raw = "_".join(rel.parts)
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("._-")
    slug = re.sub(r"[._-]+", "_", slug).strip("_")
    digest = hashlib.sha1(rel_text.encode("utf-8")).hexdigest()[:8]
    return f"{slug or path.stem or 'video'}_{digest}"


def sample_frame_indices(
    total_frames: int,
    source_fps: float,
    sample_seconds: float,
    max_frames: int,
) -> list[int]:
    """Compute frame indices sampled every sample_seconds, capped by max_frames."""
    if sample_seconds <= 0:
        raise ValueError("sample_seconds must be positive")
    if total_frames <= 0 or max_frames <= 0:
        return []
    fps = source_fps if source_fps > 0 else 30.0
    step = max(1, round(fps * sample_seconds))
    candidates = list(range(0, total_frames, step))
    if len(candidates) <= max_frames:
        return candidates
    if max_frames == 1:
        return [candidates[0]]

    last_index = len(candidates) - 1
    selected_positions = [
        int((idx * last_index / (max_frames - 1)) + 0.5) for idx in range(max_frames)
    ]
    return [candidates[position] for position in selected_positions]


def should_save_sequential_frame(frame_idx: int, source_fps: float, sample_seconds: float) -> bool:
    """Return True when a sequentially read frame falls on the sample interval."""
    if sample_seconds <= 0:
        raise ValueError("sample_seconds must be positive")
    fps = source_fps if source_fps > 0 else 30.0
    step = max(1, round(fps * sample_seconds))
    return frame_idx % step == 0


def extract_keyframes(
    *,
    video_root: Path | str = DEFAULT_VIDEO_ROOT,
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
    labels_path: Path | str = DEFAULT_LABELS_PATH,
    sample_seconds: float = 2.0,
    max_frames_per_video: int = 40,
    allow_empty: bool = False,
) -> dict[str, Any]:
    """Extract sampled frames from discovered videos and write batch helper files."""
    if sample_seconds <= 0:
        raise ValueError("sample_seconds must be positive")
    if max_frames_per_video <= 0:
        raise ValueError("max_frames_per_video must be positive")

    video_root = Path(video_root)
    output_root = Path(output_root)
    labels_path = Path(labels_path)
    images_dir = output_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    if labels_path.exists():
        shutil.copy2(labels_path, output_root / "labels.txt")
    else:
        (output_root / "labels.txt").write_text("\n".join(ANNOTATION_LABELS) + "\n", encoding="utf-8")

    videos = discover_videos(video_root)
    if not videos and not allow_empty:
        raise RuntimeError(f"No supported video files were discovered in {video_root}")

    video_summaries: list[dict[str, Any]] = []
    all_frames: list[dict[str, Any]] = []
    for video_path in videos:
        prefix = video_output_prefix(video_path, video_root=video_root)
        frames = extract_video(
            video_path=video_path,
            images_dir=images_dir,
            sample_seconds=sample_seconds,
            max_frames=max_frames_per_video,
            prefix=prefix,
        )
        all_frames.extend(frames)
        video_summaries.append(
            {
                "source_video": str(video_path),
                "prefix": prefix,
                "frames": len(frames),
            }
        )

    write_batch_files(output_root)
    write_contact_sheet(output_root / "contact_sheet.jpg", [Path(frame["file"]) for frame in all_frames])

    metadata: dict[str, Any] = {
        "video_root": str(video_root),
        "output_root": str(output_root),
        "labels_path": str(labels_path),
        "sample_seconds": sample_seconds,
        "max_frames_per_video": max_frames_per_video,
        "allow_empty": allow_empty,
        "video_count": len(videos),
        "frame_count": len(all_frames),
        "videos": video_summaries,
        "frames": all_frames,
    }
    (output_root / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return metadata


def extract_video(
    video_path: Path | str,
    images_dir: Path | str,
    sample_seconds: float = 2.0,
    max_frames: int = 40,
    prefix: str | None = None,
) -> list[dict[str, Any]]:
    """Extract sampled JPEG frames from one video into images_dir."""
    import cv2

    if sample_seconds <= 0:
        raise ValueError("sample_seconds must be positive")
    if max_frames <= 0:
        raise ValueError("max_frames must be positive")

    video_path = Path(video_path)
    images_dir = Path(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open video: {video_path}")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps_for_timestamp = source_fps if source_fps > 0 else 30.0
    records: list[dict[str, Any]] = []
    prefix = video_prefix(video_path) if prefix is None else prefix

    def save_frame(frame_idx: int, frame: Any) -> None:
        height, width = frame.shape[:2]
        image_path = images_dir / f"{prefix}_f{frame_idx:06d}.jpg"
        if not cv2.imwrite(str(image_path), frame):
            raise RuntimeError(f"Could not write frame image: {image_path}")

        records.append(
            {
                "file": str(image_path),
                "source_video": str(video_path),
                "prefix": prefix,
                "frame_idx": frame_idx,
                "timestamp_sec": frame_idx / fps_for_timestamp,
                "width": int(width),
                "height": int(height),
            }
        )

    try:
        if total_frames <= 0:
            frame_idx = 0
            while len(records) < max_frames:
                ok, frame = capture.read()
                if not ok or frame is None:
                    break
                if should_save_sequential_frame(frame_idx, source_fps, sample_seconds):
                    save_frame(frame_idx, frame)
                frame_idx += 1
        else:
            frame_indices = sample_frame_indices(
                total_frames=total_frames,
                source_fps=source_fps,
                sample_seconds=sample_seconds,
                max_frames=max_frames,
            )
            for frame_idx in frame_indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok, frame = capture.read()
                if not ok or frame is None:
                    continue
                save_frame(frame_idx, frame)

    finally:
        capture.release()

    if not records and max_frames > 0:
        raise RuntimeError(f"Could not extract readable frames from video: {video_path}")
    return records


def write_batch_files(output_root: Path) -> None:
    """Write README, Labelme rules, and launch helpers for an annotation batch."""
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    image_dir = output_root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    readme = f"""# Video keyframe annotation batch

Images are in `{image_dir.as_posix()}`.

Run `./launch_labelme.sh` from this directory to start Labelme with `--nodata`:

```bash
labelme images --labels labels.txt --nodata
```
"""
    (output_root / "README.md").write_text(readme, encoding="utf-8")

    all_labels = "\n".join(f"- {label}" for label in ANNOTATION_LABELS)
    polygon_labels = "\n".join(
        f"- {label}"
        for label in (
            "ego_passable",
            "ditch",
            "left_barrier",
            "right_barrier",
            "tunnel_wall",
            "suspended_object",
            "debris",
            "surface_artifact_passable",
        )
    )
    rectangle_labels = "\n".join(
        f"- {label}"
        for label in ("worker", "construction_vehicle")
    )
    rules = f"""# Annotation rules

Available labels:

{all_labels}

Use polygons for:

{polygon_labels}

Use rectangles by default for:

{rectangle_labels}

Mark suspended objects with polygons around the visible overhead or hanging
object boundary. Mark debris with polygons around the visible obstacle boundary,
especially for irregular piles, scattered material, cables, boards, and broken
ground.

`surface_artifact_passable` is not a hazard label. Use it only for passable surface
artifacts such as texture, staining, mats, shallow plates, or similar features that
should remain traversable.
"""
    (output_root / "annotation_rules.md").write_text(rules, encoding="utf-8")

    launch_script = f"""#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
labelme images --labels labels.txt --nodata
"""
    launch_path = output_root / "launch_labelme.sh"
    launch_path.write_text(launch_script, encoding="utf-8")
    launch_path.chmod(launch_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    desktop = f"""[Desktop Entry]
Type=Application
Name=Labelme Video Keyframes
Comment=Annotate video keyframes for passable segmentation
Exec={launch_path.resolve()}
Path={output_root.resolve()}
Terminal=true
Categories=Graphics;
"""
    (output_root / "launch_labelme.desktop").write_text(desktop, encoding="utf-8")


def write_contact_sheet(
    path: Path,
    image_paths: list[Path],
    thumb_width: int = 320,
    columns: int = 4,
) -> None:
    """Write a JPEG contact sheet for quick review of extracted frames."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    selected = image_paths[:48]
    columns = max(1, columns)
    thumb_width = max(1, thumb_width)

    if not selected:
        Image.new("RGB", (thumb_width * columns, thumb_width), "white").save(
            path, "JPEG", quality=90
        )
        return

    thumbs: list[Image.Image] = []
    for image_path in selected:
        with Image.open(image_path) as image:
            rgb = ImageOps.exif_transpose(image).convert("RGB")
            height = max(1, round(rgb.height * (thumb_width / rgb.width)))
            rgb.thumbnail((thumb_width, height), Image.Resampling.LANCZOS)
            thumbs.append(rgb.copy())

    cell_height = max(thumb.height for thumb in thumbs)
    rows = (len(thumbs) + columns - 1) // columns
    sheet = Image.new("RGB", (thumb_width * columns, cell_height * rows), "white")
    draw = ImageDraw.Draw(sheet)
    for index, thumb in enumerate(thumbs):
        row, col = divmod(index, columns)
        x = col * thumb_width
        y = row * cell_height
        sheet.paste(thumb, (x, y))
        draw.text((x + 6, y + 6), Path(selected[index]).name, fill=(0, 0, 0))

    sheet.save(path, "JPEG", quality=90)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS_PATH)
    parser.add_argument("--sample-seconds", type=float, default=2.0)
    parser.add_argument("--max-frames-per-video", type=int, default=40)
    parser.add_argument("--allow-empty", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    metadata = extract_keyframes(
        video_root=args.video_root,
        output_root=args.output_root,
        labels_path=args.labels,
        sample_seconds=args.sample_seconds,
        max_frames_per_video=args.max_frames_per_video,
        allow_empty=args.allow_empty,
    )
    print(f"[OK] Extracted {metadata['frame_count']} frames from {metadata['video_count']} videos")
    print(f"[OUT] {metadata['output_root']}")


if __name__ == "__main__":
    main()
