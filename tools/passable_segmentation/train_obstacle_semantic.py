#!/usr/bin/env python3
"""Train semantic obstacle segmentation from video-derived masks."""
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
    resize_pair,
    seed_everything,
)


# ==== TUNABLE PARAMETERS ====
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


# ==== CORE ====
def build_train_config() -> dict:
    """Build the default obstacle-semantic training config."""
    return {
        "dataset_dir": DATASET_DIR,
        "run_dir": RUN_DIR,
        "labels": list(LABELS),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "base_channels": BASE_CHANNELS,
        "overlap_weight": OVERLAP_WEIGHT,
        "seed": SEED,
        "num_workers": NUM_WORKERS,
        "overlay_count": OVERLAY_COUNT,
    }


def read_obstacle_manifest(path: Path | str) -> list[tuple[str, Path, tuple[Path, Path, Path, Path]]]:
    """Read stem/image/worker/vehicle/suspended/debris TSV rows relative to the manifest parent."""
    path = Path(path)
    root = path.parent
    rows: list[tuple[str, Path, tuple[Path, Path, Path, Path]]] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for line_no, row in enumerate(reader, start=1):
            if len(row) != 6:
                raise ValueError(
                    f"{path}:{line_no} must have exactly 6 columns: "
                    "stem, image, worker, construction_vehicle, suspended_object, debris"
                )
            stem, image_rel, worker_rel, vehicle_rel, suspended_rel, debris_rel = row
            rows.append(
                (
                    stem,
                    root / image_rel,
                    (root / worker_rel, root / vehicle_rel, root / suspended_rel, root / debris_rel),
                )
            )
    return rows


class ObstacleSemanticDataset(Dataset):
    """PyTorch dataset for four-class obstacle hazard masks."""

    def __init__(
        self,
        manifest_path: Path | str,
        *,
        image_size: tuple[int, int] = IMAGE_SIZE,
        augment: bool = False,
        seed: int = 0,
    ):
        self.rows = read_obstacle_manifest(manifest_path)
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

        image_f = image.astype(np.float32) / 255.0
        image_f = (image_f - MEAN) / STD
        mask_f = (mask_stack > 127).astype(np.float32).transpose(2, 0, 1)
        image_t = torch.from_numpy(image_f.transpose(2, 0, 1)).float()
        mask_t = torch.from_numpy(mask_f).float()
        return {"stem": stem, "image": image_t, "mask": mask_t}


def obstacle_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    overlap_weight: float = OVERLAP_WEIGHT,
) -> torch.Tensor:
    """Combine BCE, Dice, and a small predicted-class overlap penalty."""
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    d_loss = per_class_dice_loss(logits, targets)
    probs = torch.sigmoid(logits)

    overlap = logits.sum() * 0.0
    pair_count = 0
    for left_idx in range(len(LABELS)):
        for right_idx in range(left_idx + 1, len(LABELS)):
            overlap = overlap + (probs[:, left_idx : left_idx + 1] * probs[:, right_idx : right_idx + 1]).mean()
            pair_count += 1
    overlap = overlap / max(1, pair_count)
    return bce + d_loss + overlap_weight * overlap


def per_class_dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Average soft Dice loss across obstacle channels with neutral absent classes."""
    probs = torch.sigmoid(logits)
    targets = targets.float()
    dims = (0, 2, 3)
    intersection = (probs * targets).sum(dim=dims)
    target_sum = targets.sum(dim=dims)
    denom = probs.sum(dim=dims) + target_sum
    dice = (2.0 * intersection + eps) / (denom + eps)
    dice = torch.where(target_sum > 0, dice, torch.ones_like(dice))
    return 1.0 - dice.mean()


def _loss_from_config(logits: torch.Tensor, masks: torch.Tensor, config: dict) -> torch.Tensor:
    return obstacle_loss(logits, masks, overlap_weight=float(config["overlap_weight"]))


def new_obstacle_metric_totals() -> dict[str, torch.Tensor]:
    """Create corpus-level metric accumulators for obstacle validation."""
    return {
        "intersection": torch.zeros(len(LABELS), dtype=torch.float64),
        "union": torch.zeros(len(LABELS), dtype=torch.float64),
        "pred_sum": torch.zeros(len(LABELS), dtype=torch.float64),
        "target_sum": torch.zeros(len(LABELS), dtype=torch.float64),
        "hazard_intersection": torch.zeros((), dtype=torch.float64),
        "hazard_union": torch.zeros((), dtype=torch.float64),
    }


@torch.no_grad()
def update_obstacle_metric_totals(
    totals: dict[str, torch.Tensor], logits: torch.Tensor, targets: torch.Tensor
) -> None:
    """Accumulate obstacle intersections and denominators over a batch."""
    preds = (torch.sigmoid(logits) > 0.5).float()
    targets = (targets > 0.5).float()

    totals["intersection"] += (preds * targets).sum(dim=(0, 2, 3)).detach().cpu().double()
    totals["union"] += ((preds + targets) > 0).float().sum(dim=(0, 2, 3)).detach().cpu().double()
    totals["pred_sum"] += preds.sum(dim=(0, 2, 3)).detach().cpu().double()
    totals["target_sum"] += targets.sum(dim=(0, 2, 3)).detach().cpu().double()

    hazard_pred = (preds.sum(dim=1, keepdim=True) > 0).float()
    hazard_target = (targets.sum(dim=1, keepdim=True) > 0).float()
    totals["hazard_intersection"] += (hazard_pred * hazard_target).sum().detach().cpu().double()
    totals["hazard_union"] += ((hazard_pred + hazard_target) > 0).float().sum().detach().cpu().double()


def finalize_obstacle_metrics(totals: dict[str, torch.Tensor]) -> dict[str, float]:
    """Convert corpus-level obstacle totals into IoU and Dice metrics."""
    metrics = {}
    for idx, label in enumerate(LABELS):
        intersection = totals["intersection"][idx].item()
        union = totals["union"][idx].item()
        pred_sum = totals["pred_sum"][idx].item()
        tgt_sum = totals["target_sum"][idx].item()
        metrics[f"{label}_iou"] = float(intersection / union) if union > 0 else 0.0
        dice_denom = pred_sum + tgt_sum
        metrics[f"{label}_dice"] = float((2 * intersection) / dice_denom) if dice_denom > 0 else 0.0

    hazard_intersection = totals["hazard_intersection"].item()
    hazard_union = totals["hazard_union"].item()
    metrics["obstacle_hazard_iou"] = float(hazard_intersection / hazard_union) if hazard_union > 0 else 0.0
    return metrics


@torch.no_grad()
def obstacle_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    """Compute per-obstacle IoU/Dice and aggregate hazard-union IoU for one batch."""
    totals = new_obstacle_metric_totals()
    update_obstacle_metric_totals(totals, logits, targets)
    return finalize_obstacle_metrics(totals)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: dict,
) -> dict[str, float]:
    """Run one obstacle-semantic training epoch."""
    model.train()
    total_loss = 0.0
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        logits = model(images)
        loss = _loss_from_config(logits, masks, config)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return {"loss": total_loss / max(1, len(loader.dataset))}


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, config: dict) -> dict[str, float]:
    """Evaluate one obstacle-semantic model."""
    model.eval()
    total_loss = 0.0
    metric_totals = new_obstacle_metric_totals()
    n = 0
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        logits = model(images)
        loss = _loss_from_config(logits, masks, config)
        batch_n = images.size(0)
        total_loss += loss.item() * batch_n
        update_obstacle_metric_totals(metric_totals, logits, masks)
        n += batch_n
    out = {"loss": total_loss / max(1, n)}
    out.update(finalize_obstacle_metrics(metric_totals))
    return out


def make_obstacle_overlay(image: np.ndarray, probs: np.ndarray, target: np.ndarray | None = None) -> np.ndarray:
    """Build original, prediction, and target panels for obstacle outputs."""
    colors = (
        np.array([255, 70, 70]),
        np.array([255, 160, 0]),
        np.array([150, 70, 255]),
        np.array([30, 210, 160]),
    )
    pred_overlay = _apply_obstacle_overlay(image, probs > 0.5, colors)
    panels = [image, pred_overlay]

    if target is not None:
        target_overlay = _apply_obstacle_overlay(image, target > 0.5, colors)
        panels.append(target_overlay)
    return np.concatenate(panels, axis=1)


@torch.no_grad()
def save_overlays(model: nn.Module, loader: DataLoader, device: torch.device, out_dir: Path, limit: int) -> None:
    """Save obstacle validation overlays."""
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    saved = 0
    for batch in loader:
        images = batch["image"].to(device)
        targets = batch["mask"].to(device)
        probs = torch.sigmoid(model(images)).detach().cpu().numpy()
        for i, stem in enumerate(batch["stem"]):
            image = _denormalize_image(batch["image"][i])
            canvas = make_obstacle_overlay(image, probs[i], targets[i].detach().cpu().numpy())
            cv2.imwrite(str(out_dir / f"{stem}_overlay.jpg"), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
            saved += 1
            if saved >= limit:
                return


def run_training(config: dict | None = None) -> dict:
    """Train the four-output obstacle-semantic model."""
    config = build_train_config() if config is None else config
    seed_everything(int(config["seed"]))
    dataset_dir = Path(str(config["dataset_dir"]))
    run_dir = Path(str(config["run_dir"]))
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    train_ds = ObstacleSemanticDataset(dataset_dir / "train.tsv", augment=True, seed=int(config["seed"]))
    val_ds = ObstacleSemanticDataset(dataset_dir / "val.tsv", augment=False, seed=int(config["seed"]))
    train_loader = DataLoader(
        train_ds,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=int(config["num_workers"]),
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=max(0, min(int(config["num_workers"]), 2)),
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallPassableUNet(base_channels=int(config["base_channels"]), out_channels=len(LABELS)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(config["epochs"])))

    best_score = -math.inf
    best_path = run_dir / "best_model.pt"
    history = []
    for epoch in range(1, int(config["epochs"]) + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, config)
        val_metrics = evaluate(model, val_loader, device, config)
        scheduler.step()
        score = val_metrics["obstacle_hazard_iou"] + sum(val_metrics[f"{label}_iou"] for label in LABELS)
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "score": score,
            "lr": scheduler.get_last_lr()[0],
            **{f"val_{key}": value for key, value in val_metrics.items() if key != "loss"},
        }
        history.append(row)
        print(
            f"[TRAIN] epoch {epoch:03d}/{int(config['epochs'])} "
            f"train_loss={row['train_loss']:.4f} val_loss={row['val_loss']:.4f} "
            f"hazard_iou={val_metrics['obstacle_hazard_iou']:.4f} "
            f"worker_iou={val_metrics['worker_iou']:.4f} "
            f"vehicle_iou={val_metrics['construction_vehicle_iou']:.4f} "
            f"suspended_iou={val_metrics['suspended_object_iou']:.4f} "
            f"debris_iou={val_metrics['debris_iou']:.4f}"
        )
        if score > best_score:
            best_score = score
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": config,
                    "labels": LABELS,
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                },
                best_path,
            )

    (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    final_metrics = evaluate(model, val_loader, device, config)
    save_overlays(model, val_loader, device, run_dir / "overlays_val", int(config["overlay_count"]))
    torch.save({"model": model.state_dict(), "config": config, "labels": LABELS}, run_dir / "last_model.pt")
    summary = {
        "device": str(device),
        "labels": list(LABELS),
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "best_checkpoint": str(best_path),
        "final_val": final_metrics,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[OK] Obstacle-semantic training summary")
    _print_json("[TRAIN]", summary)
    return summary


def main() -> None:
    """Run obstacle-semantic training with the tunable parameters above."""
    run_training()


# ==== HELPERS ====
def _apply_obstacle_overlay(image: np.ndarray, masks: np.ndarray, colors: tuple[np.ndarray, ...]) -> np.ndarray:
    overlay = image.copy()
    for idx, color in enumerate(colors):
        mask = masks[idx]
        overlay[mask] = (overlay[mask] * 0.35 + color * 0.65).astype(np.uint8)
    return overlay


def _denormalize_image(image: torch.Tensor) -> np.ndarray:
    arr = image.detach().cpu().numpy().transpose(1, 2, 0)
    arr = np.clip((arr * STD + MEAN) * 255.0, 0, 255).astype(np.uint8)
    return arr


def _print_json(prefix: str, data: dict) -> None:
    for line in json.dumps(data, indent=2).splitlines():
        print(f"{prefix} {line}")


# ==== TEST ====
if __name__ == "__main__":
    main()
