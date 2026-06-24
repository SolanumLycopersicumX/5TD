#!/usr/bin/env python3
"""Train passable-road, ditch, and left-boundary segmentation."""
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
from tools.passable_segmentation.train_passable_ditch import filter_small_binary_components


# ==== TUNABLE PARAMETERS ====
DATASET_DIR = "data/derived/passable_ditch_left_barrier_wall_aux_2026-06-24"
RUN_DIR = "runs/passable_ego/passable_ditch_left_barrier_v4_wall_aux"
INIT_CHECKPOINT = "runs/passable_ego/passable_ditch_artifact_v3_finetune/best_model.pt"
EPOCHS = 70
BATCH_SIZE = 8
LR = 2e-4
WEIGHT_DECAY = 1e-4
BASE_CHANNELS = 16
OVERLAP_WEIGHT = 1.5
DITCH_SAFETY_WEIGHT = 2.0
LEFT_BARRIER_WEIGHT = 4.0
BOUNDARY_CONFUSION_WEIGHT = 2.5
WALL_NEGATIVE_WEIGHT = 3.0
ARTIFACT_WEIGHT = 1.5
SEED = 23
NUM_WORKERS = 2
OVERLAY_COUNT = 12
MIN_DITCH_COMPONENT_AREA = 500
LABELS = ("ego_passable", "ditch", "left_barrier")
TARGET_LABELS = ("ego_passable", "ditch", "left_barrier", "surface_artifact_passable", "tunnel_wall")


# ==== CORE ====
def build_train_config() -> dict:
    """Build the default v4 left-boundary training config."""
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
        "ditch_safety_weight": DITCH_SAFETY_WEIGHT,
        "left_barrier_weight": LEFT_BARRIER_WEIGHT,
        "boundary_confusion_weight": BOUNDARY_CONFUSION_WEIGHT,
        "wall_negative_weight": WALL_NEGATIVE_WEIGHT,
        "artifact_weight": ARTIFACT_WEIGHT,
        "seed": SEED,
        "num_workers": NUM_WORKERS,
        "overlay_count": OVERLAY_COUNT,
        "min_ditch_component_area": MIN_DITCH_COMPONENT_AREA,
    }


def read_left_barrier_manifest(path: Path | str) -> list[tuple[str, Path, tuple[Path, ...]]]:
    """Read image/passable/ditch/left/artifact/wall TSV manifests."""
    path = Path(path)
    root = path.parent
    rows: list[tuple[str, Path, tuple[Path, ...]]] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) not in (6, 7):
                raise ValueError(
                    f"{path} must have 6 or 7 columns: stem, image, passable_mask, ditch_mask, left_mask, artifact_mask, optional_wall_mask"
                )
            stem, image_rel, *mask_rels = row
            rows.append(
                (
                    stem,
                    root / image_rel,
                    tuple(root / mask_rel for mask_rel in mask_rels),
                )
            )
    return rows


class PassableDitchLeftBarrierDataset(Dataset):
    """PyTorch dataset with passable, ditch, left-boundary, and artifact masks."""

    def __init__(
        self,
        manifest_path: Path | str,
        *,
        image_size: tuple[int, int] = IMAGE_SIZE,
        augment: bool = False,
        seed: int = 0,
    ):
        self.rows = read_left_barrier_manifest(manifest_path)
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


def passable_ditch_left_barrier_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    overlap_weight: float = OVERLAP_WEIGHT,
    ditch_safety_weight: float = DITCH_SAFETY_WEIGHT,
    left_barrier_weight: float = LEFT_BARRIER_WEIGHT,
    boundary_confusion_weight: float = BOUNDARY_CONFUSION_WEIGHT,
    wall_negative_weight: float = WALL_NEGATIVE_WEIGHT,
    artifact_weight: float = ARTIFACT_WEIGHT,
) -> torch.Tensor:
    """Penalize passable, ditch, left-boundary, and artifact mistakes."""
    segmentation_targets = targets[:, :3]
    passable = targets[:, 0:1]
    ditch = targets[:, 1:2]
    left = targets[:, 2:3]
    artifact = targets[:, 3:4]
    wall = targets[:, 4:5] if targets.shape[1] > 4 else torch.zeros_like(left)

    bce = F.binary_cross_entropy_with_logits(logits, segmentation_targets)
    d_loss = dice_loss(logits, segmentation_targets)
    probs = torch.sigmoid(logits)
    overlap = (
        (probs[:, 0:1] * probs[:, 1:2]).mean()
        + (probs[:, 0:1] * probs[:, 2:3]).mean()
        + (probs[:, 1:2] * probs[:, 2:3]).mean()
    )

    ditch_safety = _masked_bce(logits[:, 0:1], torch.zeros_like(ditch), ditch)
    ditch_safety = ditch_safety + _masked_bce(logits[:, 1:2], torch.ones_like(ditch), ditch)
    left_safety = _masked_bce(logits[:, 0:1], torch.zeros_like(left), left)
    left_safety = left_safety + _masked_bce(logits[:, 2:3], torch.ones_like(left), left)
    boundary_confusion = _masked_bce(logits[:, 1:2], torch.zeros_like(left), left)
    boundary_confusion = boundary_confusion + _masked_bce(logits[:, 2:3], torch.zeros_like(ditch), ditch)
    artifact_loss = _masked_bce(logits[:, 0:1], torch.ones_like(artifact), artifact)
    artifact_loss = artifact_loss + _masked_bce(logits[:, 1:2], torch.zeros_like(artifact), artifact)
    artifact_loss = artifact_loss + _masked_bce(logits[:, 2:3], torch.zeros_like(artifact), artifact)
    wall_negative = _masked_bce(logits[:, 0:1], torch.zeros_like(wall), wall)
    wall_negative = wall_negative + _masked_bce(logits[:, 1:2], torch.zeros_like(wall), wall)
    wall_negative = wall_negative + _masked_bce(logits[:, 2:3], torch.zeros_like(wall), wall)

    return (
        bce
        + d_loss
        + overlap_weight * overlap
        + ditch_safety_weight * ditch_safety
        + left_barrier_weight * left_safety
        + boundary_confusion_weight * boundary_confusion
        + wall_negative_weight * wall_negative
        + artifact_weight * artifact_loss
        + (1.0 - passable).sum() * 0.0
    )


@torch.no_grad()
def left_barrier_safe_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    """Compute safe-passable metrics plus left-boundary confusion rates."""
    preds = (torch.sigmoid(logits) > 0.5).float()
    targets = (targets > 0.5).float()

    pass_pred = preds[:, 0:1]
    ditch_pred = preds[:, 1:2]
    left_pred = preds[:, 2:3]
    pass_tgt = targets[:, 0:1]
    ditch_tgt = targets[:, 1:2]
    left_tgt = targets[:, 2:3]
    artifact_tgt = targets[:, 3:4]
    wall_tgt = targets[:, 4:5] if targets.shape[1] > 4 else torch.zeros_like(left_tgt)
    safe_pred = pass_pred * (1.0 - ditch_pred) * (1.0 - left_pred)
    safe_tgt = pass_tgt * (1.0 - ditch_tgt) * (1.0 - left_tgt)

    metrics = {}
    for name, pred, tgt in [
        ("passable", pass_pred, pass_tgt),
        ("ditch", ditch_pred, ditch_tgt),
        ("left_barrier", left_pred, left_tgt),
        ("safe", safe_pred, safe_tgt),
    ]:
        intersection = (pred * tgt).sum().item()
        union = ((pred + tgt) > 0).float().sum().item()
        pred_sum = pred.sum().item()
        tgt_sum = tgt.sum().item()
        metrics[f"{name}_iou"] = float((intersection + 1e-6) / (union + 1e-6))
        metrics[f"{name}_dice"] = float((2 * intersection + 1e-6) / (pred_sum + tgt_sum + 1e-6))

    ditch_pixels = ditch_tgt.sum().item()
    left_pixels = left_tgt.sum().item()
    artifact_pixels = artifact_tgt.sum().item()
    wall_pixels = wall_tgt.sum().item()
    metrics["ditch_as_passable_rate"] = float(((pass_pred * ditch_tgt).sum().item() + 1e-6) / (ditch_pixels + 1e-6))
    metrics["left_barrier_as_passable_rate"] = float(
        ((pass_pred * left_tgt).sum().item() + 1e-6) / (left_pixels + 1e-6)
    )
    metrics["left_barrier_as_ditch_rate"] = float(
        ((ditch_pred * left_tgt).sum().item() + 1e-6) / (left_pixels + 1e-6)
    )
    metrics["ditch_as_left_barrier_rate"] = float(
        ((left_pred * ditch_tgt).sum().item() + 1e-6) / (ditch_pixels + 1e-6)
    )
    metrics["passable_ditch_overlap_rate"] = float(
        ((pass_pred * ditch_pred).sum().item() + 1e-6)
        / (pass_pred.sum().item() + ditch_pred.sum().item() + 1e-6)
    )
    metrics["passable_left_barrier_overlap_rate"] = float(
        ((pass_pred * left_pred).sum().item() + 1e-6) / (pass_pred.sum().item() + left_pred.sum().item() + 1e-6)
    )

    if artifact_pixels <= 0:
        metrics["artifact_ditch_false_positive_rate"] = 0.0
        metrics["artifact_left_barrier_false_positive_rate"] = 0.0
        metrics["artifact_passable_false_negative_rate"] = 0.0
    else:
        metrics["artifact_ditch_false_positive_rate"] = float(
            ((ditch_pred * artifact_tgt).sum().item() + 1e-6) / (artifact_pixels + 1e-6)
        )
        metrics["artifact_left_barrier_false_positive_rate"] = float(
            ((left_pred * artifact_tgt).sum().item() + 1e-6) / (artifact_pixels + 1e-6)
        )
        metrics["artifact_passable_false_negative_rate"] = float(
            (((1.0 - pass_pred) * artifact_tgt).sum().item() + 1e-6) / (artifact_pixels + 1e-6)
        )
    if wall_pixels <= 0:
        metrics["wall_passable_false_positive_rate"] = 0.0
        metrics["wall_ditch_false_positive_rate"] = 0.0
        metrics["wall_left_barrier_false_positive_rate"] = 0.0
    else:
        metrics["wall_passable_false_positive_rate"] = float(
            ((pass_pred * wall_tgt).sum().item() + 1e-6) / (wall_pixels + 1e-6)
        )
        metrics["wall_ditch_false_positive_rate"] = float(
            ((ditch_pred * wall_tgt).sum().item() + 1e-6) / (wall_pixels + 1e-6)
        )
        metrics["wall_left_barrier_false_positive_rate"] = float(
            ((left_pred * wall_tgt).sum().item() + 1e-6) / (wall_pixels + 1e-6)
        )
    return metrics


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: dict,
) -> dict[str, float]:
    """Run one v4 training epoch."""
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
    """Evaluate v4 segmentation on one loader."""
    model.eval()
    total_loss = 0.0
    totals: dict[str, float] = {}
    n = 0
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        logits = model(images)
        loss = _loss_from_config(logits, masks, config)
        metrics = left_barrier_safe_metrics(logits, masks)
        batch_n = images.size(0)
        total_loss += loss.item() * batch_n
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value * batch_n
        n += batch_n
    out = {"loss": total_loss / max(1, n)}
    out.update({key: value / max(1, n) for key, value in totals.items()})
    return out


def make_left_barrier_overlay(
    image: np.ndarray,
    probs: np.ndarray,
    target: np.ndarray | None = None,
    *,
    min_ditch_area: int = 0,
) -> np.ndarray:
    """Build original, prediction, and target panels for v4 outputs."""
    probs = _postprocess_v4_probabilities(probs, min_ditch_area=min_ditch_area)
    passable = probs[0] > 0.5
    ditch = probs[1] > 0.5
    left = probs[2] > 0.5
    safe = passable & ~ditch & ~left

    pred_overlay = image.copy()
    pred_overlay[safe] = (pred_overlay[safe] * 0.45 + np.array([0, 220, 80]) * 0.55).astype(np.uint8)
    pred_overlay[ditch] = (pred_overlay[ditch] * 0.35 + np.array([230, 30, 30]) * 0.65).astype(np.uint8)
    pred_overlay[left] = (pred_overlay[left] * 0.35 + np.array([0, 170, 255]) * 0.65).astype(np.uint8)
    panels = [image, pred_overlay]

    if target is not None:
        target_overlay = image.copy()
        pass_tgt = target[0] > 0.5
        ditch_tgt = target[1] > 0.5
        left_tgt = target[2] > 0.5
        artifact_tgt = target[3] > 0.5
        wall_tgt = target[4] > 0.5 if target.shape[0] > 4 else np.zeros_like(artifact_tgt)
        target_overlay[pass_tgt] = (target_overlay[pass_tgt] * 0.45 + np.array([255, 210, 0]) * 0.55).astype(
            np.uint8
        )
        target_overlay[ditch_tgt] = (target_overlay[ditch_tgt] * 0.35 + np.array([0, 120, 255]) * 0.65).astype(
            np.uint8
        )
        target_overlay[left_tgt] = (target_overlay[left_tgt] * 0.35 + np.array([0, 255, 255]) * 0.65).astype(
            np.uint8
        )
        target_overlay[wall_tgt] = (target_overlay[wall_tgt] * 0.55 + np.array([80, 80, 80]) * 0.45).astype(
            np.uint8
        )
        target_overlay[artifact_tgt] = (
            target_overlay[artifact_tgt] * 0.25 + np.array([255, 0, 255]) * 0.75
        ).astype(np.uint8)
        panels.append(target_overlay)
    return np.concatenate(panels, axis=1)


@torch.no_grad()
def save_overlays(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    out_dir: Path,
    limit: int,
    *,
    min_ditch_area: int = 0,
) -> None:
    """Save v4 validation overlays."""
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    saved = 0
    for batch in loader:
        images = batch["image"].to(device)
        targets = batch["mask"].to(device)
        probs = torch.sigmoid(model(images)).detach().cpu().numpy()
        for i, stem in enumerate(batch["stem"]):
            image = _denormalize_image(batch["image"][i])
            canvas = make_left_barrier_overlay(
                image,
                probs[i],
                targets[i].detach().cpu().numpy(),
                min_ditch_area=min_ditch_area,
            )
            cv2.imwrite(str(out_dir / f"{stem}_overlay.jpg"), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
            saved += 1
            if saved >= limit:
                return


def run_training(config: dict | None = None) -> dict:
    """Train the v4 passable-road, ditch, and left-boundary model."""
    config = build_train_config() if config is None else config
    seed_everything(int(config["seed"]))
    dataset_dir = Path(str(config["dataset_dir"]))
    run_dir = Path(str(config["run_dir"]))
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    train_ds = PassableDitchLeftBarrierDataset(dataset_dir / "train.tsv", augment=True, seed=int(config["seed"]))
    val_ds = PassableDitchLeftBarrierDataset(dataset_dir / "val.tsv", augment=False, seed=int(config["seed"]))
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
    model = SmallPassableUNet(base_channels=int(config["base_channels"]), out_channels=3).to(device)
    init_checkpoint = str(config.get("init_checkpoint", "") or "")
    if init_checkpoint:
        _load_partial_checkpoint(model, init_checkpoint, device)
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
        score = _score_metrics(val_metrics)
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
            f"left_iou={val_metrics['left_barrier_iou']:.4f} "
            f"left_as_ditch={val_metrics['left_barrier_as_ditch_rate']:.4f}"
        )
        if score > best_score:
            best_score = score
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": config,
                    "labels": LABELS,
                    "target_labels": TARGET_LABELS,
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                },
                best_path,
            )

    (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    final_metrics = evaluate(model, val_loader, device, config)
    save_overlays(
        model,
        val_loader,
        device,
        run_dir / "overlays_val",
        int(config["overlay_count"]),
        min_ditch_area=int(config.get("min_ditch_component_area", 0)),
    )
    torch.save(
        {"model": model.state_dict(), "config": config, "labels": LABELS, "target_labels": TARGET_LABELS},
        run_dir / "last_model.pt",
    )
    summary = {
        "device": str(device),
        "labels": list(LABELS),
        "target_labels": list(TARGET_LABELS),
        "init_checkpoint": init_checkpoint,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "best_checkpoint": str(best_path),
        "final_val": final_metrics,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[OK] V4 left-boundary training summary")
    _print_json("[TRAIN]", summary)
    return summary


def main() -> None:
    """Run v4 training with the tunable parameters above."""
    run_training()


# ==== HELPERS ====
def _loss_from_config(logits: torch.Tensor, masks: torch.Tensor, config: dict) -> torch.Tensor:
    return passable_ditch_left_barrier_loss(
        logits,
        masks,
        overlap_weight=float(config["overlap_weight"]),
        ditch_safety_weight=float(config["ditch_safety_weight"]),
        left_barrier_weight=float(config["left_barrier_weight"]),
        boundary_confusion_weight=float(config["boundary_confusion_weight"]),
        wall_negative_weight=float(config["wall_negative_weight"]),
        artifact_weight=float(config["artifact_weight"]),
    )


def _masked_bce(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.sum().item() <= 0:
        return logits.sum() * 0.0
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def _score_metrics(metrics: dict[str, float]) -> float:
    return (
        metrics["safe_iou"]
        + 1.2 * metrics["ditch_iou"]
        + 1.0 * metrics["left_barrier_iou"]
        - 4.0 * metrics["ditch_as_passable_rate"]
        - 3.0 * metrics["left_barrier_as_ditch_rate"]
        - 2.0 * metrics["left_barrier_as_passable_rate"]
        - 2.0 * metrics["artifact_ditch_false_positive_rate"]
        - 2.0 * metrics["wall_ditch_false_positive_rate"]
        - 2.0 * metrics["wall_left_barrier_false_positive_rate"]
        - 0.5 * metrics["artifact_passable_false_negative_rate"]
    )


def _load_partial_checkpoint(model: nn.Module, checkpoint_path: str, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    source = checkpoint.get("model", checkpoint)
    target = model.state_dict()
    for key, value in source.items():
        if key == "out.weight" and key in target and target[key].shape[0] >= value.shape[0]:
            target[key][: value.shape[0]].copy_(value)
        elif key == "out.bias" and key in target and target[key].shape[0] >= value.shape[0]:
            target[key][: value.shape[0]].copy_(value)
        elif key in target and target[key].shape == value.shape:
            target[key].copy_(value)
    model.load_state_dict(target)


def _postprocess_v4_probabilities(probs: np.ndarray, *, min_ditch_area: int = 0) -> np.ndarray:
    processed = probs.copy()
    if min_ditch_area <= 1:
        return processed
    ditch_mask = processed[1] > 0.5
    filtered_ditch = filter_small_binary_components(ditch_mask, min_area=min_ditch_area)
    processed[1][ditch_mask & ~filtered_ditch] = 0.0
    return processed


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
