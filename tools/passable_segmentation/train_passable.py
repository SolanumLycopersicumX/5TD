#!/usr/bin/env python3
"""Train first-stage binary ego-passable segmentation."""
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

# ==== TUNABLE PARAMETERS ====
DATASET_DIR = "data/derived/passable_ego_2026-06-24"
RUN_DIR = "runs/passable_ego/first_augmented"
EPOCHS = 80
BATCH_SIZE = 8
LR = 1e-3
WEIGHT_DECAY = 1e-4
BASE_CHANNELS = 16
SEED = 7
NUM_WORKERS = 2
OVERLAY_COUNT = 12
IMAGE_SIZE = (384, 640)  # height, width
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


# ==== CORE ====
def build_train_config() -> dict:
    """Build the default training config from module-level tunables."""
    return {
        "dataset_dir": DATASET_DIR,
        "run_dir": RUN_DIR,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "base_channels": BASE_CHANNELS,
        "seed": SEED,
        "num_workers": NUM_WORKERS,
        "overlay_count": OVERLAY_COUNT,
    }


def read_manifest(path: Path) -> list[tuple[str, Path, Path]]:
    """Read a three-column image/mask TSV manifest."""
    rows: list[tuple[str, Path, Path]] = []
    root = path.parent
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for stem, image_rel, mask_rel in reader:
            rows.append((stem, root / image_rel, root / mask_rel))
    return rows


class PassableDataset(Dataset):
    """PyTorch dataset for binary passable-road segmentation."""

    def __init__(
        self,
        manifest_path: Path | str,
        *,
        image_size: tuple[int, int] = IMAGE_SIZE,
        augment: bool = False,
        seed: int = 0,
    ):
        self.rows = read_manifest(Path(manifest_path))
        self.image_size = image_size
        self.augment = augment
        self.seed = seed

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        stem, image_path, mask_path = self.rows[idx]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(mask_path)

        rng = random.Random(self.seed + idx * 1009 + random.randint(0, 1_000_000))
        if self.augment:
            image, mask = augment_image_and_mask(image, mask, rng)
        image, mask = resize_pair(image, mask, self.image_size)

        image_f = image.astype(np.float32) / 255.0
        image_f = (image_f - MEAN) / STD
        mask_f = (mask > 127).astype(np.float32)[None, ...]
        image_t = torch.from_numpy(image_f.transpose(2, 0, 1)).float()
        mask_t = torch.from_numpy(mask_f).float()
        return {"stem": stem, "image": image_t, "mask": mask_t}


def resize_pair(
    image: np.ndarray, mask: np.ndarray, image_size: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Resize image and mask with safe interpolation choices."""
    height, width = image_size
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return image, mask


def augment_image_and_mask(
    image: np.ndarray, mask: np.ndarray, rng: random.Random
) -> tuple[np.ndarray, np.ndarray]:
    """Apply tunnel-specific image augmentation while preserving masks."""
    if rng.random() < 0.5:
        image = np.ascontiguousarray(image[:, ::-1])
        mask = np.ascontiguousarray(mask[:, ::-1])

    if rng.random() < 0.7:
        image, mask = random_affine(image, mask, rng)

    if rng.random() < 0.85:
        image = random_light(image, rng)

    if rng.random() < 0.35:
        image = random_blur(image, rng)

    if rng.random() < 0.35:
        image = add_water_reflection(image, rng)

    if rng.random() < 0.25:
        image = add_sensor_noise(image, rng)

    return image, mask


def random_affine(
    image: np.ndarray, mask: np.ndarray, rng: random.Random
) -> tuple[np.ndarray, np.ndarray]:
    """Apply small geometric perturbations."""
    height, width = image.shape[:2]
    angle = rng.uniform(-4.0, 4.0)
    scale = rng.uniform(0.94, 1.06)
    tx = rng.uniform(-0.025, 0.025) * width
    ty = rng.uniform(-0.02, 0.02) * height
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, scale)
    matrix[:, 2] += [tx, ty]
    image_aug = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    mask_aug = cv2.warpAffine(
        mask,
        matrix,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return image_aug, mask_aug


def random_light(image: np.ndarray, rng: random.Random) -> np.ndarray:
    """Perturb brightness, contrast, gamma, and headlight intensity."""
    img = image.astype(np.float32) / 255.0
    contrast = rng.uniform(0.65, 1.35)
    brightness = rng.uniform(-0.22, 0.22)
    gamma = rng.uniform(0.65, 1.55)
    img = np.clip((img - 0.5) * contrast + 0.5 + brightness, 0, 1)
    img = np.power(img, gamma)
    if rng.random() < 0.35:
        img *= rng.uniform(0.35, 0.75)
    if rng.random() < 0.35:
        add_headlight_spot(img, rng)
    return np.clip(img * 255.0, 0, 255).astype(np.uint8)


def add_headlight_spot(img: np.ndarray, rng: random.Random) -> None:
    """Add a synthetic headlight spot in-place."""
    height, width = img.shape[:2]
    cx = rng.randint(width // 4, width * 3 // 4)
    cy = rng.randint(height // 3, height * 4 // 5)
    radius = rng.randint(max(12, width // 10), max(20, width // 4))
    yy, xx = np.ogrid[:height, :width]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    alpha = np.clip(1.0 - dist / radius, 0, 1) ** 2
    strength = rng.uniform(0.25, 0.65)
    img[:] = np.clip(img + alpha[..., None] * strength, 0, 1)


def random_blur(image: np.ndarray, rng: random.Random) -> np.ndarray:
    """Apply Gaussian or motion blur."""
    if rng.random() < 0.5:
        kernel = rng.choice([3, 5])
        return cv2.GaussianBlur(image, (kernel, kernel), 0)
    kernel_size = rng.choice([5, 7])
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[kernel_size // 2, :] = 1.0 / kernel_size
    return cv2.filter2D(image, -1, kernel)


def add_water_reflection(image: np.ndarray, rng: random.Random) -> np.ndarray:
    """Add synthetic low-area water reflections."""
    img = image.astype(np.float32)
    height, width = img.shape[:2]
    overlay = np.zeros_like(img)
    for _ in range(rng.randint(1, 4)):
        cx = rng.randint(width // 8, width * 7 // 8)
        cy = rng.randint(height // 2, height - 1)
        axes = (rng.randint(width // 20, width // 7), rng.randint(3, max(5, height // 40)))
        color = rng.choice([(210, 220, 225), (170, 190, 200), (230, 230, 210)])
        cv2.ellipse(overlay, (cx, cy), axes, rng.uniform(-12, 12), 0, 360, color, -1)
    overlay = cv2.GaussianBlur(overlay, (15, 15), 0)
    alpha = rng.uniform(0.12, 0.32)
    return np.clip(img * (1 - alpha) + overlay * alpha, 0, 255).astype(np.uint8)


def add_sensor_noise(image: np.ndarray, rng: random.Random) -> np.ndarray:
    """Add synthetic sensor noise."""
    sigma = rng.uniform(3.0, 14.0)
    noise = np.random.default_rng(rng.randint(0, 1_000_000)).normal(0, sigma, image.shape)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


class ConvBlock(nn.Module):
    """Two-layer convolution block used by the small U-Net."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SmallPassableUNet(nn.Module):
    """Small U-Net for passable-road segmentation."""

    def __init__(self, base_channels: int = 16, out_channels: int = 1):
        super().__init__()
        c = base_channels
        self.out_channels = out_channels
        self.enc1 = ConvBlock(3, c)
        self.enc2 = ConvBlock(c, c * 2)
        self.enc3 = ConvBlock(c * 2, c * 4)
        self.bottleneck = ConvBlock(c * 4, c * 8)
        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, 2, stride=2)
        self.dec3 = ConvBlock(c * 8, c * 4)
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
        self.dec2 = ConvBlock(c * 4, c * 2)
        self.up1 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
        self.dec1 = ConvBlock(c * 2, c)
        self.out = nn.Conv2d(c, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        b = self.bottleneck(F.max_pool2d(e3, 2))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Compute soft Dice loss for segmentation logits."""
    probs = torch.sigmoid(logits)
    dims = (1, 2, 3)
    intersection = (probs * targets).sum(dim=dims)
    denom = probs.sum(dim=dims) + targets.sum(dim=dims)
    dice = (2 * intersection + eps) / (denom + eps)
    return 1.0 - dice.mean()


def segmentation_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Combine BCE and Dice losses."""
    return F.binary_cross_entropy_with_logits(logits, targets) + dice_loss(logits, targets)


@torch.no_grad()
def binary_segmentation_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    """Compute IoU and Dice for binary segmentation."""
    preds = (torch.sigmoid(logits) > 0.5).float()
    targets = (targets > 0.5).float()
    intersection = (preds * targets).sum().item()
    union = ((preds + targets) > 0).float().sum().item()
    pred_sum = preds.sum().item()
    target_sum = targets.sum().item()
    iou = (intersection + 1e-6) / (union + 1e-6)
    dice = (2 * intersection + 1e-6) / (pred_sum + target_sum + 1e-6)
    return {"iou": float(iou), "dice": float(dice)}


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    """Run one training epoch."""
    model.train()
    total_loss = 0.0
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        logits = model(images)
        loss = segmentation_loss(logits, masks)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return {"loss": total_loss / max(1, len(loader.dataset))}


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    """Evaluate binary segmentation on one loader."""
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    n = 0
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        logits = model(images)
        loss = segmentation_loss(logits, masks)
        metrics = binary_segmentation_metrics(logits, masks)
        batch_n = images.size(0)
        total_loss += loss.item() * batch_n
        total_iou += metrics["iou"] * batch_n
        total_dice += metrics["dice"] * batch_n
        n += batch_n
    return {
        "loss": total_loss / max(1, n),
        "iou": total_iou / max(1, n),
        "dice": total_dice / max(1, n),
    }


def denormalize_image(image: torch.Tensor) -> np.ndarray:
    """Convert a normalized CHW tensor back to an RGB image."""
    arr = image.detach().cpu().numpy().transpose(1, 2, 0)
    arr = np.clip((arr * STD + MEAN) * 255.0, 0, 255).astype(np.uint8)
    return arr


@torch.no_grad()
def save_overlays(model: nn.Module, loader: DataLoader, device: torch.device, out_dir: Path, limit: int) -> None:
    """Save validation overlays for quick visual inspection."""
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    saved = 0
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        logits = model(images)
        probs = torch.sigmoid(logits)
        for i, stem in enumerate(batch["stem"]):
            image = denormalize_image(batch["image"][i])
            pred = (probs[i, 0].detach().cpu().numpy() > 0.5).astype(np.uint8)
            target = (masks[i, 0].detach().cpu().numpy() > 0.5).astype(np.uint8)
            overlay = image.copy()
            overlay[pred > 0] = (overlay[pred > 0] * 0.45 + np.array([0, 220, 80]) * 0.55).astype(np.uint8)
            overlay[target > 0] = (overlay[target > 0] * 0.75 + np.array([255, 210, 0]) * 0.25).astype(np.uint8)
            canvas = np.concatenate([image, overlay], axis=1)
            cv2.imwrite(str(out_dir / f"{stem}_overlay.jpg"), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
            saved += 1
            if saved >= limit:
                return


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_training(config: dict | None = None) -> dict:
    """Train the binary passable-road model."""
    config = build_train_config() if config is None else config
    seed_everything(int(config["seed"]))
    dataset_dir = Path(str(config["dataset_dir"]))
    run_dir = Path(str(config["run_dir"]))
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    train_ds = PassableDataset(dataset_dir / "train.tsv", augment=True, seed=int(config["seed"]))
    val_ds = PassableDataset(dataset_dir / "val.tsv", augment=False, seed=int(config["seed"]))
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
    model = SmallPassableUNet(base_channels=int(config["base_channels"])).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(config["epochs"])))

    best_iou = -math.inf
    best_path = run_dir / "best_model.pt"
    history = []
    for epoch in range(1, int(config["epochs"]) + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "val_iou": val_metrics["iou"],
            "val_dice": val_metrics["dice"],
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(row)
        print(
            f"[TRAIN] epoch {epoch:03d}/{int(config['epochs'])} "
            f"train_loss={row['train_loss']:.4f} "
            f"val_loss={row['val_loss']:.4f} "
            f"val_iou={row['val_iou']:.4f} "
            f"val_dice={row['val_dice']:.4f}"
        )
        if val_metrics["iou"] > best_iou:
            best_iou = val_metrics["iou"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                },
                best_path,
            )

    (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    final_metrics = evaluate(model, val_loader, device)
    save_overlays(model, val_loader, device, run_dir / "overlays_val", int(config["overlay_count"]))
    torch.save({"model": model.state_dict(), "config": config}, run_dir / "last_model.pt")
    summary = {
        "device": str(device),
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "best_checkpoint": str(best_path),
        "final_val": final_metrics,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[OK] Binary training summary")
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
