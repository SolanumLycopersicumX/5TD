#!/usr/bin/env python3
"""Train an auxiliary left-boundary and tunnel-wall segmentation model."""
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


# ==== TUNABLE PARAMETERS ====
DATASET_DIR = "data/derived/passable_ditch_left_barrier_wall_aux_no_testvideo_2026-06-24"
RUN_DIR = "runs/passable_ego/boundary_wall_aux_v2_no_testvideo"
EPOCHS = 80
BATCH_SIZE = 8
LR = 1e-3
WEIGHT_DECAY = 1e-4
BASE_CHANNELS = 16
OVERLAP_WEIGHT = 1.5
CONFUSION_WEIGHT = 2.5
SEED = 31
NUM_WORKERS = 2
OVERLAY_COUNT = 12
LABELS = ("left_barrier", "tunnel_wall")


# ==== CORE ====
def build_train_config() -> dict:
    """Build the default auxiliary boundary-wall training config."""
    return {
        "dataset_dir": DATASET_DIR,
        "run_dir": RUN_DIR,
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


def read_boundary_wall_manifest(path: Path | str) -> list[tuple[str, Path, tuple[Path, Path]]]:
    """Read a full v4 manifest and select left-boundary and wall masks."""
    path = Path(path)
    root = path.parent
    rows: list[tuple[str, Path, tuple[Path, Path]]] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 7:
                raise ValueError(
                    f"{path} must have at least 7 columns: stem, image, passable, ditch, left, artifact, wall"
                )
            stem, image_rel = row[:2]
            left_rel = row[4]
            wall_rel = row[6]
            rows.append((stem, root / image_rel, (root / left_rel, root / wall_rel)))
    return rows


class BoundaryWallDataset(Dataset):
    """PyTorch dataset for left-boundary and tunnel-wall masks."""

    def __init__(
        self,
        manifest_path: Path | str,
        *,
        image_size: tuple[int, int] = IMAGE_SIZE,
        augment: bool = False,
        seed: int = 0,
    ):
        self.rows = read_boundary_wall_manifest(manifest_path)
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


def boundary_wall_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    overlap_weight: float = OVERLAP_WEIGHT,
    confusion_weight: float = CONFUSION_WEIGHT,
) -> torch.Tensor:
    """Penalize left-boundary/wall segmentation and class confusion."""
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    d_loss = dice_loss(logits, targets)
    probs = torch.sigmoid(logits)
    left = targets[:, 0:1]
    wall = targets[:, 1:2]
    overlap = (probs[:, 0:1] * probs[:, 1:2]).mean()
    confusion = _masked_bce(logits[:, 0:1], torch.zeros_like(wall), wall)
    confusion = confusion + _masked_bce(logits[:, 1:2], torch.zeros_like(left), left)
    return bce + d_loss + overlap_weight * overlap + confusion_weight * confusion


@torch.no_grad()
def boundary_wall_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    """Compute boundary-wall IoU and confusion rates."""
    preds = (torch.sigmoid(logits) > 0.5).float()
    targets = (targets > 0.5).float()
    left_pred = preds[:, 0:1]
    wall_pred = preds[:, 1:2]
    left_tgt = targets[:, 0:1]
    wall_tgt = targets[:, 1:2]

    metrics = {}
    for name, pred, tgt in [("left_barrier", left_pred, left_tgt), ("tunnel_wall", wall_pred, wall_tgt)]:
        intersection = (pred * tgt).sum().item()
        union = ((pred + tgt) > 0).float().sum().item()
        pred_sum = pred.sum().item()
        tgt_sum = tgt.sum().item()
        metrics[f"{name}_iou"] = float((intersection + 1e-6) / (union + 1e-6))
        metrics[f"{name}_dice"] = float((2 * intersection + 1e-6) / (pred_sum + tgt_sum + 1e-6))

    left_pixels = left_tgt.sum().item()
    wall_pixels = wall_tgt.sum().item()
    metrics["left_barrier_as_wall_rate"] = float(((wall_pred * left_tgt).sum().item() + 1e-6) / (left_pixels + 1e-6))
    metrics["wall_as_left_barrier_rate"] = float(((left_pred * wall_tgt).sum().item() + 1e-6) / (wall_pixels + 1e-6))
    metrics["left_wall_overlap_rate"] = float(
        ((left_pred * wall_pred).sum().item() + 1e-6) / (left_pred.sum().item() + wall_pred.sum().item() + 1e-6)
    )
    return metrics


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: dict,
) -> dict[str, float]:
    """Run one boundary-wall training epoch."""
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
    """Evaluate one boundary-wall model."""
    model.eval()
    total_loss = 0.0
    totals: dict[str, float] = {}
    n = 0
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        logits = model(images)
        loss = _loss_from_config(logits, masks, config)
        metrics = boundary_wall_metrics(logits, masks)
        batch_n = images.size(0)
        total_loss += loss.item() * batch_n
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value * batch_n
        n += batch_n
    out = {"loss": total_loss / max(1, n)}
    out.update({key: value / max(1, n) for key, value in totals.items()})
    return out


def make_boundary_wall_overlay(image: np.ndarray, probs: np.ndarray, target: np.ndarray | None = None) -> np.ndarray:
    """Build original, prediction, and target panels for boundary-wall outputs."""
    left = probs[0] > 0.5
    wall = probs[1] > 0.5
    pred_overlay = image.copy()
    pred_overlay[left] = (pred_overlay[left] * 0.35 + np.array([0, 170, 255]) * 0.65).astype(np.uint8)
    pred_overlay[wall] = (pred_overlay[wall] * 0.45 + np.array([120, 120, 120]) * 0.55).astype(np.uint8)
    panels = [image, pred_overlay]

    if target is not None:
        target_overlay = image.copy()
        left_tgt = target[0] > 0.5
        wall_tgt = target[1] > 0.5
        target_overlay[left_tgt] = (target_overlay[left_tgt] * 0.35 + np.array([0, 255, 255]) * 0.65).astype(
            np.uint8
        )
        target_overlay[wall_tgt] = (target_overlay[wall_tgt] * 0.45 + np.array([80, 80, 80]) * 0.55).astype(
            np.uint8
        )
        panels.append(target_overlay)
    return np.concatenate(panels, axis=1)


@torch.no_grad()
def save_overlays(model: nn.Module, loader: DataLoader, device: torch.device, out_dir: Path, limit: int) -> None:
    """Save boundary-wall validation overlays."""
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    saved = 0
    for batch in loader:
        images = batch["image"].to(device)
        targets = batch["mask"].to(device)
        probs = torch.sigmoid(model(images)).detach().cpu().numpy()
        for i, stem in enumerate(batch["stem"]):
            image = _denormalize_image(batch["image"][i])
            canvas = make_boundary_wall_overlay(image, probs[i], targets[i].detach().cpu().numpy())
            cv2.imwrite(str(out_dir / f"{stem}_overlay.jpg"), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
            saved += 1
            if saved >= limit:
                return


def run_training(config: dict | None = None) -> dict:
    """Train the auxiliary boundary-wall model."""
    config = build_train_config() if config is None else config
    seed_everything(int(config["seed"]))
    dataset_dir = Path(str(config["dataset_dir"]))
    run_dir = Path(str(config["run_dir"]))
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    train_ds = BoundaryWallDataset(dataset_dir / "train.tsv", augment=True, seed=int(config["seed"]))
    val_ds = BoundaryWallDataset(dataset_dir / "val.tsv", augment=False, seed=int(config["seed"]))
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
    model = SmallPassableUNet(base_channels=int(config["base_channels"]), out_channels=2).to(device)
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
        score = (
            val_metrics["left_barrier_iou"]
            + val_metrics["tunnel_wall_iou"]
            - 2.0 * val_metrics["wall_as_left_barrier_rate"]
            - 1.0 * val_metrics["left_barrier_as_wall_rate"]
        )
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
            f"left_iou={val_metrics['left_barrier_iou']:.4f} wall_iou={val_metrics['tunnel_wall_iou']:.4f} "
            f"wall_as_left={val_metrics['wall_as_left_barrier_rate']:.4f}"
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
    print("[OK] Boundary-wall training summary")
    _print_json("[TRAIN]", summary)
    return summary


def main() -> None:
    """Run auxiliary boundary-wall training with the tunable parameters above."""
    run_training()


# ==== HELPERS ====
def _loss_from_config(logits: torch.Tensor, masks: torch.Tensor, config: dict) -> torch.Tensor:
    return boundary_wall_loss(
        logits,
        masks,
        overlap_weight=float(config["overlap_weight"]),
        confusion_weight=float(config["confusion_weight"]),
    )


def _masked_bce(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.sum().item() <= 0:
        return logits.sum() * 0.0
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


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
