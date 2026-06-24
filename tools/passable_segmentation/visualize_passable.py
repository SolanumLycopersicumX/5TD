#!/usr/bin/env python3
"""Visualize binary ego-passable segmentation predictions."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.passable_segmentation.train_passable import IMAGE_SIZE, MEAN, STD, SmallPassableUNet


# ==== TUNABLE PARAMETERS ====
CHECKPOINT_PATH = Path("runs/passable_ego/first_augmented/best_model.pt")
OUTPUT_DIR = Path("runs/passable_ego/first_augmented/overlays_all_labeled")
IMAGE_DIR = Path("data/annotation_batches/rgb_keyframes_2026-06-24_more_keyframes/images")
MANIFEST_PATH = Path("data/derived/passable_ego_2026-06-24/manifest.tsv")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


# ==== CORE ====
def collect_image_paths(image_dir: Path | str) -> list[Path]:
    """Collect supported image files from a directory."""
    image_dir = Path(image_dir)
    return sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def read_manifest_images(manifest_path: Path | str) -> list[tuple[str, Path, Path | None]]:
    """Read single-mask visualization records from a TSV manifest."""
    manifest_path = Path(manifest_path)
    root = manifest_path.parent
    rows: list[tuple[str, Path, Path | None]] = []
    with manifest_path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for stem, image_rel, mask_rel in reader:
            rows.append((stem, root / image_rel, root / mask_rel))
    return rows


def load_model(checkpoint_path: Path | str, device: torch.device) -> SmallPassableUNet:
    """Load a binary passable-road checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    model = SmallPassableUNet(base_channels=int(config.get("base_channels", 16)))
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
def predict_probability(model: SmallPassableUNet, image_rgb: np.ndarray, device: torch.device) -> np.ndarray:
    """Predict one passable-road probability map."""
    input_tensor = preprocess_image(image_rgb).to(device)
    logits = model(input_tensor)
    prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
    return prob.astype(np.float32)


def make_overlay_canvas(
    image_rgb: np.ndarray, probability: np.ndarray, target_mask: np.ndarray | None = None
) -> np.ndarray:
    """Build side-by-side original, prediction, and optional target panels."""
    height, width = probability.shape
    image_resized = cv2.resize(image_rgb, (width, height), interpolation=cv2.INTER_AREA)
    pred_mask = probability > 0.5

    pred_overlay = image_resized.copy()
    pred_overlay[pred_mask] = (
        pred_overlay[pred_mask] * 0.45 + np.array([0, 220, 80]) * 0.55
    ).astype(np.uint8)

    panels = [image_resized, pred_overlay]
    if target_mask is not None:
        target_resized = cv2.resize(target_mask, (width, height), interpolation=cv2.INTER_NEAREST) > 127
        target_overlay = image_resized.copy()
        target_overlay[target_resized] = (
            target_overlay[target_resized] * 0.45 + np.array([255, 210, 0]) * 0.55
        ).astype(np.uint8)
        panels.append(target_overlay)
    return np.concatenate(panels, axis=1)


def write_visualizations(
    *,
    checkpoint_path: Path | str,
    output_dir: Path | str,
    image_dir: Path | str | None = None,
    manifest_path: Path | str | None = None,
) -> int:
    """Write visualization JPEGs for a manifest or image directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint_path, device)

    if manifest_path is not None:
        records = read_manifest_images(manifest_path)
    elif image_dir is not None:
        records = [(p.stem, p, None) for p in collect_image_paths(image_dir)]
    else:
        raise ValueError("Provide either image_dir or manifest_path.")

    for stem, image_path, mask_path in records:
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(image_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        target = None
        if mask_path is not None and mask_path.exists():
            target = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        probability = predict_probability(model, image_rgb, device)
        canvas = make_overlay_canvas(image_rgb, probability, target)
        out_path = output_dir / f"{stem}_passable_overlay.jpg"
        cv2.imwrite(str(out_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])

    return len(records)


def main() -> None:
    """Run visualization with the tunable parameters above."""
    count = write_visualizations(
        checkpoint_path=CHECKPOINT_PATH,
        output_dir=OUTPUT_DIR,
        manifest_path=MANIFEST_PATH,
    )
    # count = write_visualizations(checkpoint_path=CHECKPOINT_PATH, output_dir=OUTPUT_DIR, image_dir=IMAGE_DIR)
    print(f"[OK] Wrote {count} overlays to {OUTPUT_DIR}")


# ==== TEST ====
if __name__ == "__main__":
    main()
