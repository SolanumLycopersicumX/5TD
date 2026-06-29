#!/usr/bin/env python3
"""Train left/right boundary and tunnel-wall segmentation from video-derived masks."""
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
    add_sensor_noise,
    add_water_reflection,
    dice_loss,
    random_affine,
    random_blur,
    random_light,
    resize_pair,
    seed_everything,
)


# ==== TUNABLE PARAMETERS ====
DATASET_DIR = "data/derived/passable_boundary_obstacle_2026-06-29/boundary"
RUN_DIR = "runs/passable_ego/boundary_wall_right_videos_2026-06-29"
INIT_CHECKPOINT = "runs/passable_ego/boundary_wall_aux_v2_no_testvideo/best_model.pt"
LABELS = ("left_barrier", "right_barrier", "tunnel_wall")
EPOCHS = 70
BATCH_SIZE = 8
LR = 5e-4
WEIGHT_DECAY = 1e-4
BASE_CHANNELS = 16
OVERLAP_WEIGHT = 1.5
CONFUSION_WEIGHT = 2.5
SEED = 43
NUM_WORKERS = 2
OVERLAY_COUNT = 12


# ==== CORE ====
def build_train_config() -> dict:
    """Build the default right-boundary training config."""
    return {
        "dataset_dir": DATASET_DIR,
        "run_dir": RUN_DIR,
        "init_checkpoint": INIT_CHECKPOINT,
        "labels": list(LABELS),
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


def read_boundary_right_wall_manifest(path: Path | str) -> list[tuple[str, Path, tuple[Path, Path, Path]]]:
    """Read stem/image/left/right/wall TSV rows relative to the manifest parent."""
    path = Path(path)
    root = path.parent
    rows: list[tuple[str, Path, tuple[Path, Path, Path]]] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for line_no, row in enumerate(reader, start=1):
            if len(row) != 5:
                raise ValueError(
                    f"{path}:{line_no} must have exactly 5 columns: stem, image, left, right, wall"
                )
            stem, image_rel, left_rel, right_rel, wall_rel = row
            rows.append((stem, root / image_rel, (root / left_rel, root / right_rel, root / wall_rel)))
    return rows


class BoundaryRightWallDataset(Dataset):
    """PyTorch dataset for left/right boundary and tunnel-wall masks."""

    def __init__(
        self,
        manifest_path: Path | str,
        *,
        image_size: tuple[int, int] = IMAGE_SIZE,
        augment: bool = False,
        seed: int = 0,
    ):
        self.rows = read_boundary_right_wall_manifest(manifest_path)
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
            image, mask_stack = augment_boundary_image_and_mask(image, mask_stack, rng)
        image, mask_stack = resize_pair(image, mask_stack, self.image_size)

        image_f = image.astype(np.float32) / 255.0
        image_f = (image_f - MEAN) / STD
        mask_f = (mask_stack > 127).astype(np.float32).transpose(2, 0, 1)
        image_t = torch.from_numpy(image_f.transpose(2, 0, 1)).float()
        mask_t = torch.from_numpy(mask_f).float()
        return {"stem": stem, "image": image_t, "mask": mask_t}


def augment_boundary_image_and_mask(
    image: np.ndarray, mask_stack: np.ndarray, rng: random.Random
) -> tuple[np.ndarray, np.ndarray]:
    """Apply augmentation while preserving left/right boundary channel semantics."""
    if rng.random() < 0.5:
        image = np.ascontiguousarray(image[:, ::-1])
        mask_stack = np.ascontiguousarray(mask_stack[:, ::-1]).copy()
        mask_stack[..., [0, 1]] = mask_stack[..., [1, 0]]

    if rng.random() < 0.7:
        image, mask_stack = random_affine(image, mask_stack, rng)

    if rng.random() < 0.85:
        image = random_light(image, rng)

    if rng.random() < 0.35:
        image = random_blur(image, rng)

    if rng.random() < 0.35:
        image = add_water_reflection(image, rng)

    if rng.random() < 0.25:
        image = add_sensor_noise(image, rng)

    return image, mask_stack


def boundary_right_wall_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    overlap_weight: float = OVERLAP_WEIGHT,
    confusion_weight: float = CONFUSION_WEIGHT,
) -> torch.Tensor:
    """Penalize segmentation error, class overlap, and cross-class confusion."""
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    d_loss = dice_loss(logits, targets)
    probs = torch.sigmoid(logits)

    overlap = logits.sum() * 0.0
    confusion = logits.sum() * 0.0
    pair_count = 0
    for src_idx in range(len(LABELS)):
        for pred_idx in range(src_idx + 1, len(LABELS)):
            overlap = overlap + (probs[:, src_idx : src_idx + 1] * probs[:, pred_idx : pred_idx + 1]).mean()
            pair_count += 1
        src_target = targets[:, src_idx : src_idx + 1]
        for pred_idx in range(len(LABELS)):
            if pred_idx == src_idx:
                continue
            confusion = confusion + masked_bce(
                logits[:, pred_idx : pred_idx + 1],
                torch.zeros_like(src_target),
                src_target,
            )
    overlap = overlap / max(1, pair_count)
    return bce + d_loss + overlap_weight * overlap + confusion_weight * confusion


def masked_bce(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute BCE over masked pixels, returning a differentiable zero for empty masks."""
    if mask.sum().item() <= 0:
        return logits.sum() * 0.0
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def copy_compatible_state(
    model: nn.Module,
    source_state: dict[str, torch.Tensor],
    *,
    source_labels: tuple[str, ...] | list[str] = (),
    target_labels: tuple[str, ...] | list[str] = LABELS,
) -> None:
    """Load matching weights and copy compatible output channels by label when possible."""
    source_labels = tuple(source_labels)
    target_labels = tuple(target_labels)
    target_label_to_idx = {label: idx for idx, label in enumerate(target_labels)}

    target_state = model.state_dict()
    updated = {key: value.detach().clone() for key, value in target_state.items()}
    for key, source_value in source_state.items():
        if key not in updated:
            continue
        target_value = updated[key]
        if key in {"out.weight", "out.bias"}:
            if target_value.ndim != source_value.ndim or target_value.shape[1:] != source_value.shape[1:]:
                continue
            copied = target_value.detach().clone()
            source_detached = source_value.detach()
            if source_labels:
                for source_idx, label in enumerate(source_labels):
                    target_idx = target_label_to_idx.get(label)
                    if target_idx is None:
                        continue
                    if source_idx < source_detached.shape[0] and target_idx < copied.shape[0]:
                        copied[target_idx] = source_detached[source_idx]
                updated[key] = copied
            elif target_value.shape[0] >= source_value.shape[0]:
                copied[: source_value.shape[0]] = source_detached.clone()
                updated[key] = copied
        elif source_value.shape == target_value.shape:
            updated[key] = source_value.detach().clone()
    model.load_state_dict(updated)


def _loss_from_config(logits: torch.Tensor, masks: torch.Tensor, config: dict) -> torch.Tensor:
    return boundary_right_wall_loss(
        logits,
        masks,
        overlap_weight=float(config["overlap_weight"]),
        confusion_weight=float(config["confusion_weight"]),
    )


def new_boundary_metric_totals() -> dict[str, torch.Tensor]:
    """Create corpus-level metric accumulators for boundary/right-wall validation."""
    label_count = len(LABELS)
    return {
        "intersection": torch.zeros(label_count, dtype=torch.float64),
        "union": torch.zeros(label_count, dtype=torch.float64),
        "pred_sum": torch.zeros(label_count, dtype=torch.float64),
        "target_sum": torch.zeros(label_count, dtype=torch.float64),
        "confusion": torch.zeros((label_count, label_count), dtype=torch.float64),
        "overlap": torch.zeros((label_count, label_count), dtype=torch.float64),
        "overlap_denom": torch.zeros((label_count, label_count), dtype=torch.float64),
    }


@torch.no_grad()
def update_boundary_metric_totals(
    totals: dict[str, torch.Tensor], logits: torch.Tensor, targets: torch.Tensor
) -> None:
    """Accumulate boundary intersections and denominators over a batch."""
    preds = (torch.sigmoid(logits) > 0.5).float()
    targets = (targets > 0.5).float()

    totals["intersection"] += (preds * targets).sum(dim=(0, 2, 3)).detach().cpu().double()
    totals["union"] += ((preds + targets) > 0).float().sum(dim=(0, 2, 3)).detach().cpu().double()
    totals["pred_sum"] += preds.sum(dim=(0, 2, 3)).detach().cpu().double()
    totals["target_sum"] += targets.sum(dim=(0, 2, 3)).detach().cpu().double()

    for source_idx in range(len(LABELS)):
        source_tgt = targets[:, source_idx : source_idx + 1]
        for pred_idx in range(len(LABELS)):
            if pred_idx == source_idx:
                continue
            pred = preds[:, pred_idx : pred_idx + 1]
            totals["confusion"][source_idx, pred_idx] += (pred * source_tgt).sum().detach().cpu().double()

    pred_sums = preds.sum(dim=(0, 2, 3)).detach().cpu().double()
    for left_idx in range(len(LABELS)):
        for right_idx in range(left_idx + 1, len(LABELS)):
            left_pred = preds[:, left_idx : left_idx + 1]
            right_pred = preds[:, right_idx : right_idx + 1]
            totals["overlap"][left_idx, right_idx] += (left_pred * right_pred).sum().detach().cpu().double()
            totals["overlap_denom"][left_idx, right_idx] += pred_sums[left_idx] + pred_sums[right_idx]


def finalize_boundary_metrics(totals: dict[str, torch.Tensor]) -> dict[str, float]:
    """Convert corpus-level boundary totals into IoU, Dice, confusion, and overlap metrics."""
    metrics = {}
    for idx, label in enumerate(LABELS):
        intersection = totals["intersection"][idx].item()
        union = totals["union"][idx].item()
        pred_sum = totals["pred_sum"][idx].item()
        tgt_sum = totals["target_sum"][idx].item()
        metrics[f"{label}_iou"] = float(intersection / union) if union > 0 else 0.0
        dice_denom = pred_sum + tgt_sum
        metrics[f"{label}_dice"] = float((2 * intersection) / dice_denom) if dice_denom > 0 else 0.0

    for source_idx, source_label in enumerate(LABELS):
        source_pixels = totals["target_sum"][source_idx].item()
        for pred_idx, pred_label in enumerate(LABELS):
            if pred_idx == source_idx:
                continue
            confusion_pixels = totals["confusion"][source_idx, pred_idx].item()
            metrics[f"{source_label}_as_{pred_label}_rate"] = (
                float(confusion_pixels / source_pixels) if source_pixels > 0 else 0.0
            )

    for left_idx in range(len(LABELS)):
        for right_idx in range(left_idx + 1, len(LABELS)):
            denom = totals["overlap_denom"][left_idx, right_idx].item()
            overlap_pixels = totals["overlap"][left_idx, right_idx].item()
            metrics[f"{LABELS[left_idx]}_{LABELS[right_idx]}_overlap_rate"] = (
                float(overlap_pixels / denom) if denom > 0 else 0.0
            )
    return metrics


@torch.no_grad()
def boundary_right_wall_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    """Compute per-label IoU/Dice and deterministic pairwise confusion rates for one batch."""
    totals = new_boundary_metric_totals()
    update_boundary_metric_totals(totals, logits, targets)
    return finalize_boundary_metrics(totals)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: dict,
) -> dict[str, float]:
    """Run one right-boundary training epoch."""
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
    """Evaluate one right-boundary model."""
    model.eval()
    total_loss = 0.0
    metric_totals = new_boundary_metric_totals()
    n = 0
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        logits = model(images)
        loss = _loss_from_config(logits, masks, config)
        batch_n = images.size(0)
        total_loss += loss.item() * batch_n
        update_boundary_metric_totals(metric_totals, logits, masks)
        n += batch_n
    out = {"loss": total_loss / max(1, n)}
    out.update(finalize_boundary_metrics(metric_totals))
    return out


def make_boundary_right_wall_overlay(
    image: np.ndarray, probs: np.ndarray, target: np.ndarray | None = None
) -> np.ndarray:
    """Build original, prediction, and target panels for right-boundary outputs."""
    left = probs[0] > 0.5
    right = probs[1] > 0.5
    wall = probs[2] > 0.5
    pred_overlay = image.copy()
    pred_overlay[left] = (pred_overlay[left] * 0.35 + np.array([0, 170, 255]) * 0.65).astype(np.uint8)
    pred_overlay[right] = (pred_overlay[right] * 0.35 + np.array([40, 90, 255]) * 0.65).astype(np.uint8)
    pred_overlay[wall] = (pred_overlay[wall] * 0.45 + np.array([120, 120, 120]) * 0.55).astype(np.uint8)
    panels = [image, pred_overlay]

    if target is not None:
        target_overlay = image.copy()
        left_tgt = target[0] > 0.5
        right_tgt = target[1] > 0.5
        wall_tgt = target[2] > 0.5
        target_overlay[left_tgt] = (target_overlay[left_tgt] * 0.35 + np.array([0, 255, 255]) * 0.65).astype(
            np.uint8
        )
        target_overlay[right_tgt] = (target_overlay[right_tgt] * 0.35 + np.array([0, 90, 255]) * 0.65).astype(
            np.uint8
        )
        target_overlay[wall_tgt] = (target_overlay[wall_tgt] * 0.45 + np.array([80, 80, 80]) * 0.55).astype(
            np.uint8
        )
        panels.append(target_overlay)
    return np.concatenate(panels, axis=1)


@torch.no_grad()
def save_overlays(model: nn.Module, loader: DataLoader, device: torch.device, out_dir: Path, limit: int) -> None:
    """Save right-boundary validation overlays."""
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    saved = 0
    for batch in loader:
        images = batch["image"].to(device)
        targets = batch["mask"].to(device)
        probs = torch.sigmoid(model(images)).detach().cpu().numpy()
        for i, stem in enumerate(batch["stem"]):
            image = _denormalize_image(batch["image"][i])
            canvas = make_boundary_right_wall_overlay(image, probs[i], targets[i].detach().cpu().numpy())
            cv2.imwrite(str(out_dir / f"{stem}_overlay.jpg"), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
            saved += 1
            if saved >= limit:
                return


def run_training(config: dict | None = None) -> dict:
    """Train the three-output boundary/right-wall model."""
    config = build_train_config() if config is None else config
    seed_everything(int(config["seed"]))
    dataset_dir = Path(str(config["dataset_dir"]))
    run_dir = Path(str(config["run_dir"]))
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    train_ds = BoundaryRightWallDataset(dataset_dir / "train.tsv", augment=True, seed=int(config["seed"]))
    val_ds = BoundaryRightWallDataset(dataset_dir / "val.tsv", augment=False, seed=int(config["seed"]))
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
    init_checkpoint_value = config.get("init_checkpoint")
    if init_checkpoint_value:
        init_checkpoint = Path(str(init_checkpoint_value))
        if init_checkpoint.exists():
            checkpoint = torch.load(init_checkpoint, map_location=device, weights_only=False)
            source_state = checkpoint.get("model", checkpoint)
            source_labels = tuple(checkpoint.get("labels", ())) if isinstance(checkpoint, dict) else ()
            copy_compatible_state(model, source_state, source_labels=source_labels)
            print(f"[TRAIN] Loaded compatible checkpoint weights from {init_checkpoint}")
        else:
            print(f"[WARN] Init checkpoint not found, training from scratch: {init_checkpoint}")

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
            + val_metrics["right_barrier_iou"]
            + val_metrics["tunnel_wall_iou"]
            - val_metrics["left_barrier_as_right_barrier_rate"]
            - val_metrics["right_barrier_as_left_barrier_rate"]
            - val_metrics["tunnel_wall_as_left_barrier_rate"]
            - val_metrics["tunnel_wall_as_right_barrier_rate"]
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
            f"left_iou={val_metrics['left_barrier_iou']:.4f} "
            f"right_iou={val_metrics['right_barrier_iou']:.4f} "
            f"wall_iou={val_metrics['tunnel_wall_iou']:.4f}"
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
    print("[OK] Boundary-right-wall training summary")
    _print_json("[TRAIN]", summary)
    return summary


def main() -> None:
    """Run right-boundary training with the tunable parameters above."""
    run_training()


# ==== HELPERS ====
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
