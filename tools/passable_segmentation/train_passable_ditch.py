#!/usr/bin/env python3
"""Train dual-output passable-road and ditch segmentation."""
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
DATASET_DIR = "data/derived/passable_ditch_2026-06-24"
RUN_DIR = "runs/passable_ego/passable_ditch_augmented"
EPOCHS = 90
BATCH_SIZE = 8
LR = 1e-3
WEIGHT_DECAY = 1e-4
BASE_CHANNELS = 16
OVERLAP_WEIGHT = 1.5
SEED = 11
NUM_WORKERS = 2
OVERLAY_COUNT = 12
LABELS = ("ego_passable", "ditch")


# ==== CORE ====
def build_train_config() -> dict:
    """Build the default dual-output training config from module-level tunables."""
    return {
        "dataset_dir": DATASET_DIR,
        "run_dir": RUN_DIR,
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


def read_multilabel_manifest(path: Path | str) -> list[tuple[str, Path, tuple[Path, Path]]]:
    """Read a four-column image/passable/ditch TSV manifest."""
    path = Path(path)
    root = path.parent
    rows: list[tuple[str, Path, tuple[Path, Path]]] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) != 4:
                raise ValueError(f"{path} must have 4 columns: stem, image, ego_passable_mask, ditch_mask")
            stem, image_rel, ego_rel, ditch_rel = row
            rows.append((stem, root / image_rel, (root / ego_rel, root / ditch_rel)))
    return rows


class PassableDitchDataset(Dataset):
    """PyTorch dataset for passable-road and ditch segmentation."""

    def __init__(
        self,
        manifest_path: Path | str,
        *,
        image_size: tuple[int, int] = IMAGE_SIZE,
        augment: bool = False,
        seed: int = 0,
    ):
        self.rows = read_multilabel_manifest(manifest_path)
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


def passable_ditch_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    overlap_weight: float = 1.5,
) -> torch.Tensor:
    """Penalize segmentation error and passable/ditch overlap."""
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    d_loss = dice_loss(logits, targets)
    probs = torch.sigmoid(logits)
    predicted_overlap = (probs[:, 0:1] * probs[:, 1:2]).mean()
    passable_on_true_ditch = (probs[:, 0:1] * targets[:, 1:2]).mean()
    return bce + d_loss + overlap_weight * (predicted_overlap + passable_on_true_ditch)


@torch.no_grad()
def safe_passable_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    """Compute passable, ditch, and safe-passable metrics."""
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    targets = (targets > 0.5).float()

    pass_pred = preds[:, 0:1]
    ditch_pred = preds[:, 1:2]
    pass_tgt = targets[:, 0:1]
    ditch_tgt = targets[:, 1:2]
    safe_pred = pass_pred * (1.0 - ditch_pred)
    safe_tgt = pass_tgt * (1.0 - ditch_tgt)

    metrics = {}
    for name, pred, tgt in [
        ("passable", pass_pred, pass_tgt),
        ("ditch", ditch_pred, ditch_tgt),
        ("safe", safe_pred, safe_tgt),
    ]:
        intersection = (pred * tgt).sum().item()
        union = ((pred + tgt) > 0).float().sum().item()
        pred_sum = pred.sum().item()
        tgt_sum = tgt.sum().item()
        metrics[f"{name}_iou"] = float((intersection + 1e-6) / (union + 1e-6))
        metrics[f"{name}_dice"] = float((2 * intersection + 1e-6) / (pred_sum + tgt_sum + 1e-6))

    ditch_pixels = ditch_tgt.sum().item()
    ditch_as_passable = (pass_pred * ditch_tgt).sum().item()
    metrics["ditch_as_passable_rate"] = float((ditch_as_passable + 1e-6) / (ditch_pixels + 1e-6))
    metrics["passable_ditch_overlap_rate"] = float(
        ((pass_pred * ditch_pred).sum().item() + 1e-6) / (pass_pred.sum().item() + ditch_pred.sum().item() + 1e-6)
    )
    return metrics


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    overlap_weight: float,
) -> dict[str, float]:
    """Run one dual-output training epoch."""
    model.train()
    total_loss = 0.0
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        logits = model(images)
        loss = passable_ditch_loss(logits, masks, overlap_weight=overlap_weight)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return {"loss": total_loss / max(1, len(loader.dataset))}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    overlap_weight: float,
) -> dict[str, float]:
    """Evaluate dual-output segmentation on one loader."""
    model.eval()
    total_loss = 0.0
    totals: dict[str, float] = {}
    n = 0
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        logits = model(images)
        loss = passable_ditch_loss(logits, masks, overlap_weight=overlap_weight)
        metrics = safe_passable_metrics(logits, masks)
        batch_n = images.size(0)
        total_loss += loss.item() * batch_n
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value * batch_n
        n += batch_n
    out = {"loss": total_loss / max(1, n)}
    out.update({key: value / max(1, n) for key, value in totals.items()})
    return out


def denormalize_image(image: torch.Tensor) -> np.ndarray:
    """Convert a normalized CHW tensor back to an RGB image."""
    arr = image.detach().cpu().numpy().transpose(1, 2, 0)
    arr = np.clip((arr * STD + MEAN) * 255.0, 0, 255).astype(np.uint8)
    return arr


def make_dual_overlay(image: np.ndarray, probs: np.ndarray, target: np.ndarray | None = None) -> np.ndarray:
    """Build original, prediction, and optional target panels."""
    passable = probs[0] > 0.5
    ditch = probs[1] > 0.5
    safe = passable & ~ditch

    pred_overlay = image.copy()
    pred_overlay[safe] = (pred_overlay[safe] * 0.45 + np.array([0, 220, 80]) * 0.55).astype(np.uint8)
    pred_overlay[ditch] = (pred_overlay[ditch] * 0.35 + np.array([230, 30, 30]) * 0.65).astype(np.uint8)

    panels = [image, pred_overlay]
    if target is not None:
        tgt_overlay = image.copy()
        pass_tgt = target[0] > 0.5
        ditch_tgt = target[1] > 0.5
        tgt_overlay[pass_tgt] = (tgt_overlay[pass_tgt] * 0.45 + np.array([255, 210, 0]) * 0.55).astype(np.uint8)
        tgt_overlay[ditch_tgt] = (tgt_overlay[ditch_tgt] * 0.35 + np.array([0, 120, 255]) * 0.65).astype(np.uint8)
        panels.append(tgt_overlay)
    return np.concatenate(panels, axis=1)


@torch.no_grad()
def save_overlays(model: nn.Module, loader: DataLoader, device: torch.device, out_dir: Path, limit: int) -> None:
    """Save dual-output validation overlays."""
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    saved = 0
    for batch in loader:
        images = batch["image"].to(device)
        targets = batch["mask"].to(device)
        probs = torch.sigmoid(model(images)).detach().cpu().numpy()
        for i, stem in enumerate(batch["stem"]):
            image = denormalize_image(batch["image"][i])
            canvas = make_dual_overlay(image, probs[i], targets[i].detach().cpu().numpy())
            cv2.imwrite(str(out_dir / f"{stem}_overlay.jpg"), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
            saved += 1
            if saved >= limit:
                return


def run_training(config: dict | None = None) -> dict:
    """Train the dual-output passable-road and ditch model."""
    config = build_train_config() if config is None else config
    seed_everything(int(config["seed"]))
    dataset_dir = Path(str(config["dataset_dir"]))
    run_dir = Path(str(config["run_dir"]))
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    train_ds = PassableDitchDataset(dataset_dir / "train.tsv", augment=True, seed=int(config["seed"]))
    val_ds = PassableDitchDataset(dataset_dir / "val.tsv", augment=False, seed=int(config["seed"]))
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
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, float(config["overlap_weight"]))
        val_metrics = evaluate(model, val_loader, device, float(config["overlap_weight"]))
        scheduler.step()
        score = (
            val_metrics["safe_iou"]
            + 0.5 * val_metrics["ditch_iou"]
            - 2.0 * val_metrics["ditch_as_passable_rate"]
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
            f"safe_iou={val_metrics['safe_iou']:.4f} ditch_iou={val_metrics['ditch_iou']:.4f} "
            f"ditch_as_passable={val_metrics['ditch_as_passable_rate']:.4f}"
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
    final_metrics = evaluate(model, val_loader, device, float(config["overlap_weight"]))
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
    print("[OK] Dual-output training summary")
    print_json("[TRAIN]", summary)
    return summary


def main() -> None:
    """Run training with the tunable parameters above."""
    run_training()


# ==== HELPERS ====
def print_json(prefix: str, data: dict) -> None:
    """Print JSON with a consistent log prefix."""
    for line in json.dumps(data, indent=2).splitlines():
        print(f"{prefix} {line}")


# ==== TEST ====
if __name__ == "__main__":
    main()
