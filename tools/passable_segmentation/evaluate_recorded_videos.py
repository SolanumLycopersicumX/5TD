#!/usr/bin/env python3
"""Evaluate fused passable-road segmentation on recorded video files."""
from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.passable_segmentation.train_boundary_wall import LABELS as BOUNDARY_LABELS
from tools.passable_segmentation.train_passable_ditch import LABELS as PASSABLE_LABELS
from tools.passable_segmentation.visualize_fused_passable_boundary import (
    fuse_passable_boundary_predictions,
    load_model,
    make_fused_overlay,
    predict_probabilities,
)


DEFAULT_VIDEO_ROOT = Path("jetson-recordings/2026-06-28")
DEFAULT_OUTPUT_DIR = Path("runs/video_model_eval/2026-06-28")
DEFAULT_PASSABLE_CHECKPOINT = Path("runs/passable_ego/passable_ditch_artifact_v3_finetune/best_model.pt")
DEFAULT_BOUNDARY_CHECKPOINT = Path("runs/passable_ego/boundary_wall_aux_v2_no_testvideo/best_model.pt")
VIDEO_SUFFIXES = {".mkv", ".mp4", ".avi", ".mov", ".h264", ".mjpeg"}
METRIC_LABELS = ("safe_passable", "ditch", "left_barrier", "tunnel_wall")


@dataclass
class VideoSummary:
    video: Path
    session: str
    camera: str
    fps: float
    total_frames: int
    sampled_frames: int
    skipped_frames: int
    duration_sec: float
    elapsed_sec: float
    safe_passable_mean: float
    ditch_mean: float
    left_barrier_mean: float
    tunnel_wall_mean: float


def discover_videos(video_root: Path | str) -> list[Path]:
    """Return recorded camera video files in stable session/camera order."""
    root = Path(video_root)
    return sorted(
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES and p.name.startswith("camera_")
    )


def sample_frame_indices(total_frames: int, source_fps: float, sample_fps: float) -> list[int]:
    """Return frame indices sampled at approximately sample_fps."""
    if total_frames <= 0:
        return []
    if sample_fps <= 0:
        raise ValueError("sample_fps must be positive")
    fps = source_fps if source_fps > 0 else 30.0
    step = max(1, int(round(fps / sample_fps)))
    return list(range(0, total_frames, step))


def mask_ratios(fused: dict[str, np.ndarray]) -> dict[str, float]:
    """Calculate area ratios for each fused semantic mask."""
    ratios: dict[str, float] = {}
    for label in METRIC_LABELS:
        mask = fused[label]
        ratios[f"{label}_ratio"] = float(np.count_nonzero(mask) / mask.size) if mask.size else 0.0
    return ratios


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument("--passable-checkpoint", type=Path, default=DEFAULT_PASSABLE_CHECKPOINT)
    parser.add_argument("--boundary-checkpoint", type=Path, default=DEFAULT_BOUNDARY_CHECKPOINT)
    parser.add_argument("--cpu", action="store_true", help="force CPU inference")
    parser.add_argument("--max-contact-per-video", type=int, default=6)
    return parser


def load_fused_models(args: argparse.Namespace, device: torch.device) -> tuple[Any, Any]:
    if not args.passable_checkpoint.exists():
        raise FileNotFoundError(args.passable_checkpoint)
    if not args.boundary_checkpoint.exists():
        raise FileNotFoundError(args.boundary_checkpoint)
    passable_model = load_model(args.passable_checkpoint, expected_labels=PASSABLE_LABELS, device=device)
    boundary_model = load_model(args.boundary_checkpoint, expected_labels=BOUNDARY_LABELS, device=device)
    return passable_model, boundary_model


def evaluate_video(
    *,
    video_path: Path,
    video_root: Path,
    output_dir: Path,
    sample_fps: float,
    passable_model: Any,
    boundary_model: Any,
    device: torch.device,
) -> tuple[VideoSummary, list[dict[str, Any]], list[Path]]:
    """Evaluate one video and write sampled overlay images."""
    start = time.monotonic()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    indices = sample_frame_indices(total_frames, source_fps, sample_fps)
    rel = video_path.relative_to(video_root)
    session = rel.parts[0] if len(rel.parts) > 1 else video_path.parent.name
    camera = video_path.stem
    overlay_dir = output_dir / "overlays" / session / camera
    overlay_dir.mkdir(parents=True, exist_ok=True)

    frame_rows: list[dict[str, Any]] = []
    overlay_paths: list[Path] = []
    skipped = 0
    ratio_sums = {f"{label}_ratio": 0.0 for label in METRIC_LABELS}

    for frame_idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            skipped += 1
            continue

        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        passable_probs = predict_probabilities(passable_model, image_rgb, device)
        boundary_probs = predict_probabilities(boundary_model, image_rgb, device)
        height, width = passable_probs.shape[1:]
        image_resized = cv2.resize(image_rgb, (width, height), interpolation=cv2.INTER_AREA)
        fused = fuse_passable_boundary_predictions(passable_probs, boundary_probs)
        ratios = mask_ratios(fused)
        canvas = make_fused_overlay(image_resized, fused)

        timestamp_sec = frame_idx / source_fps if source_fps > 0 else 0.0
        overlay_path = overlay_dir / f"frame_{frame_idx:06d}_t{timestamp_sec:07.2f}.jpg"
        cv2.imwrite(str(overlay_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
        overlay_paths.append(overlay_path)

        for key, value in ratios.items():
            ratio_sums[key] += value
        frame_rows.append(
            {
                "session": session,
                "camera": camera,
                "video": str(video_path),
                "frame_idx": frame_idx,
                "timestamp_sec": round(timestamp_sec, 3),
                "overlay": str(overlay_path),
                **{key: round(value, 6) for key, value in ratios.items()},
            }
        )

    cap.release()
    sampled = len(frame_rows)
    duration_sec = total_frames / source_fps if source_fps > 0 else 0.0
    denom = sampled if sampled else 1
    summary = VideoSummary(
        video=video_path,
        session=session,
        camera=camera,
        fps=source_fps,
        total_frames=total_frames,
        sampled_frames=sampled,
        skipped_frames=skipped,
        duration_sec=duration_sec,
        elapsed_sec=time.monotonic() - start,
        safe_passable_mean=ratio_sums["safe_passable_ratio"] / denom,
        ditch_mean=ratio_sums["ditch_ratio"] / denom,
        left_barrier_mean=ratio_sums["left_barrier_ratio"] / denom,
        tunnel_wall_mean=ratio_sums["tunnel_wall_ratio"] / denom,
    )
    return summary, frame_rows, overlay_paths


def write_frame_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "session",
        "camera",
        "video",
        "frame_idx",
        "timestamp_sec",
        "overlay",
        "safe_passable_ratio",
        "ditch_ratio",
        "left_barrier_ratio",
        "tunnel_wall_ratio",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_video_summary(path: Path, summaries: list[VideoSummary]) -> None:
    fieldnames = [
        "session",
        "camera",
        "video",
        "fps",
        "total_frames",
        "sampled_frames",
        "skipped_frames",
        "duration_sec",
        "elapsed_sec",
        "safe_passable_mean",
        "ditch_mean",
        "left_barrier_mean",
        "tunnel_wall_mean",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in summaries:
            writer.writerow(
                {
                    "session": item.session,
                    "camera": item.camera,
                    "video": str(item.video),
                    "fps": round(item.fps, 3),
                    "total_frames": item.total_frames,
                    "sampled_frames": item.sampled_frames,
                    "skipped_frames": item.skipped_frames,
                    "duration_sec": round(item.duration_sec, 3),
                    "elapsed_sec": round(item.elapsed_sec, 3),
                    "safe_passable_mean": round(item.safe_passable_mean, 6),
                    "ditch_mean": round(item.ditch_mean, 6),
                    "left_barrier_mean": round(item.left_barrier_mean, 6),
                    "tunnel_wall_mean": round(item.tunnel_wall_mean, 6),
                }
            )


def select_contact_images(groups: dict[str, list[Path]], max_per_video: int) -> list[Path]:
    chosen: list[Path] = []
    per_video = max(1, max_per_video)
    for key in sorted(groups):
        paths = groups[key]
        if len(paths) <= per_video:
            chosen.extend(paths)
            continue
        positions = np.linspace(0, len(paths) - 1, num=per_video, dtype=int)
        chosen.extend(paths[int(pos)] for pos in positions)
    return chosen


def write_contact_sheet(path: Path, image_paths: list[Path], *, thumb_width: int = 480, columns: int = 3) -> None:
    if not image_paths:
        return
    thumbs = []
    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        height, width = image.shape[:2]
        scale = thumb_width / float(width)
        thumb = cv2.resize(image, (thumb_width, max(1, int(round(height * scale)))), interpolation=cv2.INTER_AREA)
        thumbs.append(thumb)
    if not thumbs:
        return

    cell_height = max(img.shape[0] for img in thumbs)
    rows = int(np.ceil(len(thumbs) / float(columns)))
    sheet = np.full((rows * cell_height, columns * thumb_width, 3), 255, dtype=np.uint8)
    for idx, thumb in enumerate(thumbs):
        row, col = divmod(idx, columns)
        y = row * cell_height
        x = col * thumb_width
        sheet[y : y + thumb.shape[0], x : x + thumb.shape[1]] = thumb
    cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])


def main() -> None:
    args = build_arg_parser().parse_args()
    video_root = args.video_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = discover_videos(video_root)
    if not videos:
        raise SystemExit(f"No camera videos found under {video_root}")

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Videos: {len(videos)}")
    passable_model, boundary_model = load_fused_models(args, device)

    summaries: list[VideoSummary] = []
    rows: list[dict[str, Any]] = []
    contact_groups: dict[str, list[Path]] = {}
    for video in videos:
        print(f"[INFO] Evaluating {video}")
        summary, frame_rows, overlays = evaluate_video(
            video_path=video,
            video_root=video_root,
            output_dir=output_dir,
            sample_fps=args.sample_fps,
            passable_model=passable_model,
            boundary_model=boundary_model,
            device=device,
        )
        summaries.append(summary)
        rows.extend(frame_rows)
        contact_groups[f"{summary.session}/{summary.camera}"] = overlays
        print(
            "[OK] "
            f"{summary.session}/{summary.camera}: {summary.sampled_frames} sampled, "
            f"{summary.skipped_frames} skipped, {summary.elapsed_sec:.1f}s"
        )

    write_frame_metrics(output_dir / "frame_metrics.csv", rows)
    write_video_summary(output_dir / "video_summary.csv", summaries)
    contact_images = select_contact_images(contact_groups, args.max_contact_per_video)
    write_contact_sheet(output_dir / "contact_sheet.jpg", contact_images)
    print(f"[OK] Wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
