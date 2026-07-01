#!/usr/bin/env python3
"""Evaluate passable, boundary, and obstacle segmentation checkpoints on videos."""
from __future__ import annotations

import argparse
import csv
import hashlib
import math
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.passable_segmentation.visualize_fused_passable_boundary import load_model, predict_probabilities


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

DEFAULT_VIDEO_ROOT = Path("Videos")
DEFAULT_OUTPUT_DIR = Path("runs/video_model_eval/2026-06-29_multitask")
DEFAULT_PASSABLE_CHECKPOINT = Path("runs/passable_ego/passable_ditch_artifact_videos_2026-06-29/best_model.pt")
DEFAULT_BOUNDARY_CHECKPOINT = Path("runs/passable_ego/boundary_wall_right_videos_2026-06-29/best_model.pt")
DEFAULT_OBSTACLE_CHECKPOINT = Path("runs/passable_ego/obstacle_semantic_videos_2026-06-29/best_model.pt")
VIDEO_SUFFIXES = (".mkv", ".mp4", ".avi", ".mov", ".MOV", ".h264", ".mjpeg")

OVERLAY_COLORS = {
    "safe_passable": np.array([0, 220, 80], dtype=np.float32),
    "ditch": np.array([230, 30, 30], dtype=np.float32),
    "left_barrier": np.array([255, 170, 0], dtype=np.float32),
    "right_barrier": np.array([255, 230, 0], dtype=np.float32),
    "tunnel_wall": np.array([130, 130, 130], dtype=np.float32),
    "worker": np.array([255, 0, 255], dtype=np.float32),
    "construction_vehicle": np.array([255, 120, 0], dtype=np.float32),
    "suspended_object": np.array([0, 210, 255], dtype=np.float32),
    "debris": np.array([150, 80, 30], dtype=np.float32),
}
RATIO_COLUMNS = [f"{label}_ratio" for label in FUSED_LABELS]
MEAN_RATIO_COLUMNS = [f"mean_{label}_ratio" for label in FUSED_LABELS]


def fuse_multitask_predictions(
    passable_probs: np.ndarray,
    boundary_probs: np.ndarray,
    obstacle_probs: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, np.ndarray]:
    """Fuse three task probability tensors into binary safety masks."""
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
    """Return foreground ratios for every fused label."""
    ratios: dict[str, float] = {}
    for label in FUSED_LABELS:
        mask = np.asarray(fused[label], dtype=bool)
        ratios[f"{label}_ratio"] = float(mask.mean()) if mask.size else 0.0
    return ratios


def discover_videos(video_root: Path | str) -> list[Path]:
    """Find supported video files recursively under a root."""
    root = Path(video_root)
    suffixes = {suffix.lower() for suffix in VIDEO_SUFFIXES}
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in suffixes)


def frame_sample_step(source_fps: float, sample_fps: float) -> int:
    """Return the frame interval for a requested sample rate."""
    if sample_fps <= 0:
        raise ValueError("sample_fps must be positive")
    fps = source_fps if source_fps > 0 and math.isfinite(source_fps) else 30.0
    return max(1, round(fps / sample_fps))


def should_sample_frame(frame_idx: int, source_fps: float, sample_fps: float) -> bool:
    """Return whether a sequential frame counter should be evaluated."""
    return frame_idx >= 0 and frame_idx % frame_sample_step(source_fps, sample_fps) == 0


def sample_frame_indices(total_frames: int, source_fps: float, sample_fps: float) -> list[int]:
    """Return frame indices sampled at approximately sample_fps."""
    if total_frames <= 0:
        return []
    return [idx for idx in range(total_frames) if should_sample_frame(idx, source_fps, sample_fps)]


def video_output_slug(video_path: Path | str, video_root: Path | str) -> str:
    """Build a filesystem-safe slug from a video's path relative to the video root."""
    path = Path(video_path)
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


def make_fused_overlay(image_rgb: np.ndarray, fused: dict[str, np.ndarray]) -> np.ndarray:
    """Build side-by-side original and fused-prediction panels."""
    pred = image_rgb.copy()
    for label in (
        "safe_passable",
        "ditch",
        "left_barrier",
        "right_barrier",
        "tunnel_wall",
        "worker",
        "construction_vehicle",
        "suspended_object",
        "debris",
    ):
        mask = fused[label]
        if not mask.any():
            continue
        color = OVERLAY_COLORS[label]
        pred[mask] = (pred[mask].astype(np.float32) * 0.45 + color * 0.55).astype(np.uint8)
    return np.concatenate([image_rgb, pred], axis=1)


def write_contact_sheet(image_paths: list[Path], output_path: Path, *, max_width: int = 1200) -> bool:
    """Write a simple contact sheet from representative overlay files."""
    images = []
    for path in image_paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is not None:
            images.append(image)
    if not images:
        return False

    thumb_width = min(360, max(1, images[0].shape[1]))
    thumbs = []
    for image in images:
        scale = thumb_width / image.shape[1]
        thumb_height = max(1, round(image.shape[0] * scale))
        thumbs.append(cv2.resize(image, (thumb_width, thumb_height), interpolation=cv2.INTER_AREA))

    cols = max(1, min(len(thumbs), max_width // thumb_width))
    rows = math.ceil(len(thumbs) / cols)
    height = max(thumb.shape[0] for thumb in thumbs)
    sheet = np.zeros((rows * height, cols * thumb_width, 3), dtype=np.uint8)
    for idx, thumb in enumerate(thumbs):
        row = idx // cols
        col = idx % cols
        y = row * height
        x = col * thumb_width
        sheet[y : y + thumb.shape[0], x : x + thumb.shape[1]] = thumb

    output_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(output_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92]))


def write_overlay_image(path: Path, overlay_rgb: np.ndarray) -> None:
    """Write one overlay image and raise when OpenCV reports failure."""
    ok = cv2.imwrite(str(path), cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise RuntimeError(f"Could not write overlay image: {path}")


def write_metric_csvs(output_dir: Path, frame_rows: list[dict[str, object]], summary_rows: list[dict[str, object]]) -> None:
    """Write per-frame and per-video metric CSV files."""
    frame_fields = ["video", "frame_idx", "timestamp_sec", "overlay", *RATIO_COLUMNS]
    with (output_dir / "frame_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=frame_fields)
        writer.writeheader()
        writer.writerows(frame_rows)

    summary_fields = [
        "video",
        "fps",
        "total_frames",
        "sampled_frames",
        "skipped_frames",
        "duration_sec",
        "elapsed_sec",
        *MEAN_RATIO_COLUMNS,
    ]
    with (output_dir / "video_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)


def evaluate_video(
    video_path: Path,
    *,
    video_root: Path,
    output_dir: Path,
    sample_fps: float,
    models: tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module],
    device: torch.device,
    max_contact_per_video: int,
) -> tuple[list[dict[str, object]], dict[str, object], list[Path]]:
    """Evaluate one video and return per-frame rows, summary row, and contact candidates."""
    start = time.perf_counter()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = source_fps if source_fps > 0 and math.isfinite(source_fps) else 30.0
    reported_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    known_total_frames = reported_total_frames if reported_total_frames > 0 else None
    slug = video_output_slug(video_path, video_root)
    overlay_dir = output_dir / "overlays" / slug
    overlay_dir.mkdir(parents=True, exist_ok=True)

    passable_model, boundary_model, obstacle_model = models
    frame_rows: list[dict[str, object]] = []
    contact_paths: list[Path] = []
    frames_read = 0
    frame_idx = 0
    while known_total_frames is None or frame_idx < known_total_frames:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frames_read += 1
        if not should_sample_frame(frame_idx, fps, sample_fps):
            frame_idx += 1
            continue

        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        passable_probs = predict_probabilities(passable_model, image_rgb, device)
        boundary_probs = predict_probabilities(boundary_model, image_rgb, device)
        obstacle_probs = predict_probabilities(obstacle_model, image_rgb, device)
        height, width = passable_probs.shape[1:]
        image_resized = cv2.resize(image_rgb, (width, height), interpolation=cv2.INTER_AREA)
        fused = fuse_multitask_predictions(passable_probs, boundary_probs, obstacle_probs)
        ratios = mask_ratios(fused)
        overlay = make_fused_overlay(image_resized, fused)
        overlay_path = overlay_dir / f"{slug}_frame_{frame_idx:06d}.jpg"
        write_overlay_image(overlay_path, overlay)
        if len(contact_paths) < max_contact_per_video:
            contact_paths.append(overlay_path)

        row: dict[str, object] = {
            "video": str(video_path),
            "frame_idx": frame_idx,
            "timestamp_sec": frame_idx / fps,
            "overlay": str(overlay_path),
        }
        row.update(ratios)
        frame_rows.append(row)
        frame_idx += 1

    cap.release()
    if frames_read == 0:
        raise RuntimeError(f"Could not read any frames from video: {video_path}")
    if not frame_rows:
        raise RuntimeError(f"Could not sample any frames from video: {video_path}")

    elapsed = time.perf_counter() - start
    total_frames = known_total_frames if known_total_frames is not None else frames_read
    skipped_frames = max(0, frames_read - len(frame_rows))
    if known_total_frames is not None and frames_read < known_total_frames:
        skipped_frames += known_total_frames - frames_read
    summary: dict[str, object] = {
        "video": str(video_path),
        "fps": fps,
        "total_frames": total_frames,
        "sampled_frames": len(frame_rows),
        "skipped_frames": skipped_frames,
        "duration_sec": total_frames / fps if fps > 0 else 0.0,
        "elapsed_sec": elapsed,
    }
    for ratio_col, mean_col in zip(RATIO_COLUMNS, MEAN_RATIO_COLUMNS):
        values = [float(row[ratio_col]) for row in frame_rows]
        summary[mean_col] = float(np.mean(values)) if values else 0.0
    return frame_rows, summary, contact_paths


def evaluate_videos(
    *,
    video_root: Path,
    output_dir: Path,
    sample_fps: float,
    passable_checkpoint: Path,
    boundary_checkpoint: Path,
    obstacle_checkpoint: Path,
    cpu: bool = False,
    max_contact_per_video: int = 4,
    allow_empty: bool = False,
) -> tuple[int, int]:
    """Evaluate all discovered videos and write metrics artifacts."""
    if sample_fps <= 0:
        raise ValueError("sample_fps must be positive")

    videos = discover_videos(video_root)
    if not videos and not allow_empty:
        raise RuntimeError(f"No supported video files were discovered in {video_root}")

    output_dir.mkdir(parents=True, exist_ok=True)
    if not videos:
        write_metric_csvs(output_dir, [], [])
        write_contact_sheet([], output_dir / "contact_sheet.jpg")
        return 0, 0

    device = torch.device("cpu" if cpu or not torch.cuda.is_available() else "cuda")
    models = (
        load_model(passable_checkpoint, expected_labels=PASSABLE_LABELS, device=device),
        load_model(boundary_checkpoint, expected_labels=BOUNDARY_LABELS, device=device),
        load_model(obstacle_checkpoint, expected_labels=OBSTACLE_LABELS, device=device),
    )

    frame_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    contact_paths: list[Path] = []
    for video_path in videos:
        rows, summary, contacts = evaluate_video(
            video_path,
            video_root=video_root,
            output_dir=output_dir,
            sample_fps=sample_fps,
            models=models,
            device=device,
            max_contact_per_video=max_contact_per_video,
        )
        frame_rows.extend(rows)
        summary_rows.append(summary)
        contact_paths.extend(contacts)

    write_metric_csvs(output_dir, frame_rows, summary_rows)
    write_contact_sheet(contact_paths, output_dir / "contact_sheet.jpg")
    return len(videos), len(frame_rows)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument("--passable-checkpoint", type=Path, default=DEFAULT_PASSABLE_CHECKPOINT)
    parser.add_argument("--boundary-checkpoint", type=Path, default=DEFAULT_BOUNDARY_CHECKPOINT)
    parser.add_argument("--obstacle-checkpoint", type=Path, default=DEFAULT_OBSTACLE_CHECKPOINT)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--max-contact-per-video", type=int, default=4)
    parser.add_argument("--allow-empty", action="store_true")
    return parser


def main() -> None:
    """Run multitask video evaluation from CLI args."""
    args = build_arg_parser().parse_args()
    videos, frames = evaluate_videos(
        video_root=args.video_root,
        output_dir=args.output_dir,
        sample_fps=args.sample_fps,
        passable_checkpoint=args.passable_checkpoint,
        boundary_checkpoint=args.boundary_checkpoint,
        obstacle_checkpoint=args.obstacle_checkpoint,
        cpu=args.cpu,
        max_contact_per_video=max(0, args.max_contact_per_video),
        allow_empty=args.allow_empty,
    )
    print(f"[OK] Evaluated {videos} videos / {frames} sampled frames into {args.output_dir}")


if __name__ == "__main__":
    main()
