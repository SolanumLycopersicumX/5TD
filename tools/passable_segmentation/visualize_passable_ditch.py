#!/usr/bin/env python3
"""Visualize dual-output passable-road and ditch predictions."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.passable_segmentation.train_passable import IMAGE_SIZE, MEAN, STD, SmallPassableUNet
from tools.passable_segmentation.train_passable_ditch import (
    LABELS,
    make_dual_overlay,
    read_multilabel_manifest,
)
from tools.passable_segmentation.visualize_passable import collect_image_paths


# ==== TUNABLE PARAMETERS ====
CHECKPOINT_PATH = Path("runs/passable_ego/passable_ditch_augmented/best_model.pt")
OUTPUT_DIR = Path("runs/passable_ego/passable_ditch_augmented/overlays_more_keyframes")
IMAGE_DIR = Path("data/annotation_batches/rgb_keyframes_2026-06-24_more_keyframes/images")
MANIFEST_PATH = Path("data/derived/passable_ditch_2026-06-24/manifest.tsv")


# ==== CORE ====
def load_dual_model(checkpoint_path: Path | str, device: torch.device) -> SmallPassableUNet:
    """Load a dual-output passable-road and ditch checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    labels = tuple(checkpoint.get("labels", LABELS))
    if labels != LABELS:
        raise ValueError(f"Expected checkpoint labels {LABELS}, got {labels}")

    model = SmallPassableUNet(base_channels=int(config.get("base_channels", 16)), out_channels=2)
    state = checkpoint.get("model", checkpoint)
    model.load_state_dict(state)
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
    """Predict passable-road and ditch probability maps."""
    input_tensor = preprocess_image(image_rgb).to(device)
    logits = model(input_tensor)
    probs = torch.sigmoid(logits)[0].detach().cpu().numpy()
    return probs.astype(np.float32)


def read_target_stack(mask_paths: tuple[Path, Path], size_hw: tuple[int, int]) -> np.ndarray:
    """Read and resize passable-road and ditch target masks."""
    height, width = size_hw
    masks = []
    for mask_path in mask_paths:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(mask_path)
        resized = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        masks.append((resized > 127).astype(np.float32))
    return np.stack(masks, axis=0)


def write_visualizations(
    *,
    checkpoint_path: Path | str,
    output_dir: Path | str,
    image_dir: Path | str | None = None,
    manifest_path: Path | str | None = None,
) -> int:
    """Write dual-output visualization JPEGs for a manifest or image directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_dual_model(checkpoint_path, device)

    if manifest_path is not None:
        records = read_multilabel_manifest(manifest_path)
    elif image_dir is not None:
        records = [(p.stem, p, None) for p in collect_image_paths(image_dir)]
    else:
        raise ValueError("Provide either image_dir or manifest_path.")

    for stem, image_path, mask_paths in records:
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(image_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        probs = predict_probabilities(model, image_rgb, device)
        height, width = probs.shape[1:]
        image_resized = cv2.resize(image_rgb, (width, height), interpolation=cv2.INTER_AREA)
        target = read_target_stack(mask_paths, (height, width)) if mask_paths is not None else None
        canvas = make_dual_overlay(image_resized, probs, target)
        out_path = output_dir / f"{stem}_passable_ditch_overlay.jpg"
        cv2.imwrite(str(out_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])

    return len(records)


def main() -> None:
    """Run dual-output visualization with the tunable parameters above."""
    count = write_visualizations(
        checkpoint_path=CHECKPOINT_PATH,
        output_dir=OUTPUT_DIR,
        image_dir=IMAGE_DIR,
    )
    # count = write_visualizations(checkpoint_path=CHECKPOINT_PATH, output_dir=OUTPUT_DIR, manifest_path=MANIFEST_PATH)
    print(f"[OK] Wrote {count} overlays to {OUTPUT_DIR}")


# ==== TEST ====
if __name__ == "__main__":
    main()
