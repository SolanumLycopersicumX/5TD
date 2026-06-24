#!/usr/bin/env python3
"""Visualize fused v3 passable/ditch and auxiliary boundary/wall predictions."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.passable_segmentation.train_boundary_wall import LABELS as BOUNDARY_LABELS
from tools.passable_segmentation.train_passable import IMAGE_SIZE, MEAN, STD, SmallPassableUNet
from tools.passable_segmentation.train_passable_ditch import (
    LABELS as PASSABLE_LABELS,
    filter_small_binary_components,
)
from tools.passable_segmentation.visualize_passable import collect_image_paths


# ==== TUNABLE PARAMETERS ====
PASSABLE_CHECKPOINT_PATH = Path("runs/passable_ego/passable_ditch_artifact_v3_finetune/best_model.pt")
BOUNDARY_CHECKPOINT_PATH = Path("runs/passable_ego/boundary_wall_aux_v2_no_testvideo/best_model.pt")
OUTPUT_DIR = Path("runs/passable_ego/fused_passable_boundary_v2_no_testvideo/overlays_more_keyframes")
IMAGE_DIR = Path("data/annotation_batches/rgb_keyframes_2026-06-24_more_keyframes/images")
MANIFEST_PATH = Path("data/derived/passable_ditch_left_barrier_wall_aux_no_testvideo_2026-06-24/manifest.tsv")
MIN_DITCH_COMPONENT_AREA = 2000
MIN_LEFT_BARRIER_COMPONENT_AREA = 500
MIN_WALL_COMPONENT_AREA = 2500
MAX_PASSABLE_HOLE_AREA = 10000


# ==== CORE ====
def load_model(checkpoint_path: Path | str, *, expected_labels: tuple[str, ...], device: torch.device) -> SmallPassableUNet:
    """Load a segmentation checkpoint with the expected output labels."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    labels = tuple(checkpoint.get("labels", expected_labels))
    if labels != expected_labels:
        raise ValueError(f"Expected checkpoint labels {expected_labels}, got {labels}")
    model = SmallPassableUNet(base_channels=int(config.get("base_channels", 16)), out_channels=len(expected_labels))
    model.load_state_dict(checkpoint.get("model", checkpoint))
    model.to(device)
    model.eval()
    return model


def preprocess_image(image_rgb: np.ndarray) -> torch.Tensor:
    """Resize and normalize one RGB image."""
    height, width = IMAGE_SIZE
    resized = cv2.resize(image_rgb, (width, height), interpolation=cv2.INTER_AREA)
    image_f = resized.astype(np.float32) / 255.0
    image_f = (image_f - MEAN) / STD
    return torch.from_numpy(image_f.transpose(2, 0, 1)).float().unsqueeze(0)


@torch.no_grad()
def predict_probabilities(model: SmallPassableUNet, image_rgb: np.ndarray, device: torch.device) -> np.ndarray:
    """Predict probability maps for one model."""
    input_tensor = preprocess_image(image_rgb).to(device)
    logits = model(input_tensor)
    return torch.sigmoid(logits)[0].detach().cpu().numpy().astype(np.float32)


def fuse_passable_boundary_predictions(
    passable_probs: np.ndarray,
    boundary_probs: np.ndarray,
    *,
    min_ditch_area: int = MIN_DITCH_COMPONENT_AREA,
    min_left_barrier_area: int = MIN_LEFT_BARRIER_COMPONENT_AREA,
    min_wall_area: int = MIN_WALL_COMPONENT_AREA,
    max_passable_hole_area: int = MAX_PASSABLE_HOLE_AREA,
) -> dict[str, np.ndarray]:
    """Fuse main passable/ditch output with boundary/wall output by fixed safety rules."""
    passable = passable_probs[0] > 0.5
    passable = keep_bottom_connected_passable(passable)
    ditch = passable_probs[1] > 0.5
    if min_ditch_area > 1:
        ditch = filter_small_binary_components(ditch, min_area=min_ditch_area)

    wall = boundary_probs[1] > 0.5
    left = boundary_probs[0] > 0.5
    if min_left_barrier_area > 1:
        left = filter_small_binary_components(left, min_area=min_left_barrier_area)
    if min_wall_area > 1:
        wall = filter_small_binary_components(wall, min_area=min_wall_area)
    if max_passable_hole_area > 0:
        passable = fill_small_passable_holes(passable, protected=ditch | wall, max_area=max_passable_hole_area)
    left = left & ~ditch & ~wall
    safe_passable = passable & ~ditch & ~wall
    return {
        "passable": passable,
        "ditch": ditch,
        "left_barrier": left,
        "tunnel_wall": wall,
        "safe_passable": safe_passable,
    }


def fill_small_passable_holes(passable: np.ndarray, *, protected: np.ndarray, max_area: int) -> np.ndarray:
    """Fill small enclosed non-passable holes that are not ditch or wall."""
    filled = passable.copy()
    candidates = (~passable) & ~protected
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidates.astype(np.uint8), connectivity=8)
    height, width = candidates.shape
    for label_idx in range(1, num_labels):
        x, y, w, h, area = stats[label_idx]
        touches_border = x == 0 or y == 0 or x + w >= width or y + h >= height
        if touches_border or area > max_area:
            continue
        filled[labels == label_idx] = True
    return filled


def keep_bottom_connected_passable(passable: np.ndarray) -> np.ndarray:
    """Remove floating passable islands that are not connected to the vehicle-side road region."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(passable.astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return passable.copy()

    height = passable.shape[0]
    kept = np.zeros_like(passable, dtype=bool)
    largest_label = 1
    largest_area = int(stats[1, cv2.CC_STAT_AREA])
    for label_idx in range(1, num_labels):
        y = int(stats[label_idx, cv2.CC_STAT_TOP])
        h = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area > largest_area:
            largest_label = label_idx
            largest_area = area
        if y + h >= height:
            kept[labels == label_idx] = True

    if kept.any():
        return kept
    return labels == largest_label


def make_fused_overlay(image: np.ndarray, fused: dict[str, np.ndarray]) -> np.ndarray:
    """Build original and fused prediction panels."""
    pred_overlay = image.copy()
    safe = fused["safe_passable"]
    ditch = fused["ditch"]
    left = fused["left_barrier"]
    wall = fused["tunnel_wall"]
    pred_overlay[safe] = (pred_overlay[safe] * 0.45 + np.array([0, 220, 80]) * 0.55).astype(np.uint8)
    pred_overlay[left] = (pred_overlay[left] * 0.35 + np.array([0, 170, 255]) * 0.65).astype(np.uint8)
    pred_overlay[wall] = (pred_overlay[wall] * 0.45 + np.array([120, 120, 120]) * 0.55).astype(np.uint8)
    pred_overlay[ditch] = (pred_overlay[ditch] * 0.35 + np.array([230, 30, 30]) * 0.65).astype(np.uint8)
    return np.concatenate([image, pred_overlay], axis=1)


def read_manifest_images(path: Path | str) -> list[tuple[str, Path]]:
    """Read image records from any current derived segmentation manifest."""
    path = Path(path)
    root = path.parent
    records = []
    for row in path.read_text(encoding="utf-8").splitlines():
        stem, image_rel, *_ = row.split("\t")
        records.append((stem, root / image_rel))
    return records


def write_fused_visualizations(
    *,
    passable_checkpoint_path: Path | str,
    boundary_checkpoint_path: Path | str,
    output_dir: Path | str,
    image_dir: Path | str | None = None,
    manifest_path: Path | str | None = None,
    min_ditch_area: int = MIN_DITCH_COMPONENT_AREA,
    min_left_barrier_area: int = MIN_LEFT_BARRIER_COMPONENT_AREA,
    min_wall_area: int = MIN_WALL_COMPONENT_AREA,
    max_passable_hole_area: int = MAX_PASSABLE_HOLE_AREA,
) -> int:
    """Write fused prediction overlays for an image directory or manifest."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    passable_model = load_model(passable_checkpoint_path, expected_labels=PASSABLE_LABELS, device=device)
    boundary_model = load_model(boundary_checkpoint_path, expected_labels=BOUNDARY_LABELS, device=device)

    if manifest_path is not None:
        records = read_manifest_images(manifest_path)
    elif image_dir is not None:
        records = [(p.stem, p) for p in collect_image_paths(image_dir)]
    else:
        raise ValueError("Provide either image_dir or manifest_path.")

    for stem, image_path in records:
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(image_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        passable_probs = predict_probabilities(passable_model, image_rgb, device)
        boundary_probs = predict_probabilities(boundary_model, image_rgb, device)
        height, width = passable_probs.shape[1:]
        image_resized = cv2.resize(image_rgb, (width, height), interpolation=cv2.INTER_AREA)
        fused = fuse_passable_boundary_predictions(
            passable_probs,
            boundary_probs,
            min_ditch_area=min_ditch_area,
            min_left_barrier_area=min_left_barrier_area,
            min_wall_area=min_wall_area,
            max_passable_hole_area=max_passable_hole_area,
        )
        canvas = make_fused_overlay(image_resized, fused)
        out_path = output_dir / f"{stem}_fused_overlay.jpg"
        cv2.imwrite(str(out_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
    return len(records)


def main() -> None:
    """Run fused visualization with the tunable parameters above."""
    count = write_fused_visualizations(
        passable_checkpoint_path=PASSABLE_CHECKPOINT_PATH,
        boundary_checkpoint_path=BOUNDARY_CHECKPOINT_PATH,
        output_dir=OUTPUT_DIR,
        image_dir=IMAGE_DIR,
        min_ditch_area=MIN_DITCH_COMPONENT_AREA,
        min_left_barrier_area=MIN_LEFT_BARRIER_COMPONENT_AREA,
        min_wall_area=MIN_WALL_COMPONENT_AREA,
        max_passable_hole_area=MAX_PASSABLE_HOLE_AREA,
    )
    # count = write_fused_visualizations(
    #     passable_checkpoint_path=PASSABLE_CHECKPOINT_PATH,
    #     boundary_checkpoint_path=BOUNDARY_CHECKPOINT_PATH,
    #     output_dir=OUTPUT_DIR,
    #     manifest_path=MANIFEST_PATH,
    #     min_ditch_area=MIN_DITCH_COMPONENT_AREA,
    #     min_left_barrier_area=MIN_LEFT_BARRIER_COMPONENT_AREA,
    #     min_wall_area=MIN_WALL_COMPONENT_AREA,
    #     max_passable_hole_area=MAX_PASSABLE_HOLE_AREA,
    # )
    print(f"[OK] Wrote {count} fused overlays to {OUTPUT_DIR}")


# ==== TEST ====
if __name__ == "__main__":
    main()
