"""Vision gate helpers for low-speed autonomous forward driving."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class DriveGateConfig:
    """ROI and threshold configuration for conservative vision-to-drive gating."""

    roi_x_min: float = 0.35
    roi_x_max: float = 0.65
    roi_y_min: float = 0.60
    roi_y_max: float = 0.95
    min_safe_ratio: float = 0.65
    max_hazard_ratio: float = 0.02
    hazard_labels: Sequence[str] = ("ditch", "tunnel_wall", "left_barrier")


@dataclass(frozen=True)
class DriveDecision:
    """Result of evaluating whether the vehicle may move forward."""

    allow_forward: bool
    reason: str
    safe_ratio: float
    hazard_ratio: float
    roi_bounds: tuple[int, int, int, int]


def _clamp_fraction(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def roi_bounds(shape: tuple[int, int], config: DriveGateConfig) -> tuple[int, int, int, int]:
    """Return pixel ROI bounds as x0, y0, x1, y1 for a mask shape."""
    height, width = int(shape[0]), int(shape[1])
    if height <= 0 or width <= 0:
        raise ValueError("mask shape must be non-empty")

    x_min = min(_clamp_fraction(config.roi_x_min), _clamp_fraction(config.roi_x_max))
    x_max = max(_clamp_fraction(config.roi_x_min), _clamp_fraction(config.roi_x_max))
    y_min = min(_clamp_fraction(config.roi_y_min), _clamp_fraction(config.roi_y_max))
    y_max = max(_clamp_fraction(config.roi_y_min), _clamp_fraction(config.roi_y_max))

    x0 = min(width - 1, max(0, int(round(x_min * width))))
    x1 = min(width, max(x0 + 1, int(round(x_max * width))))
    y0 = min(height - 1, max(0, int(round(y_min * height))))
    y1 = min(height, max(y0 + 1, int(round(y_max * height))))
    return x0, y0, x1, y1


def _require_mask(fused: Mapping[str, np.ndarray], label: str) -> np.ndarray:
    try:
        mask = fused[label]
    except KeyError as exc:
        raise KeyError(f"missing fused mask: {label}") from exc
    if mask.ndim != 2:
        raise ValueError(f"mask {label} must be 2-D, got shape {mask.shape}")
    return mask.astype(bool, copy=False)


def evaluate_drive_gate(fused: Mapping[str, np.ndarray], config: DriveGateConfig) -> DriveDecision:
    """Decide whether fused segmentation permits low-speed straight motion."""
    safe_mask = _require_mask(fused, "safe_passable")
    bounds = roi_bounds(safe_mask.shape, config)
    x0, y0, x1, y1 = bounds
    safe_roi = safe_mask[y0:y1, x0:x1]
    total = int(safe_roi.size)
    if total <= 0:
        return DriveDecision(False, "empty_roi", 0.0, 1.0, bounds)

    hazard_roi = np.zeros_like(safe_roi, dtype=bool)
    for label in config.hazard_labels:
        if label not in fused:
            continue
        mask = _require_mask(fused, label)
        if mask.shape != safe_mask.shape:
            raise ValueError(f"mask {label} shape {mask.shape} does not match safe_passable {safe_mask.shape}")
        hazard_roi |= mask[y0:y1, x0:x1]

    safe_ratio = float(safe_roi.mean())
    hazard_ratio = float(hazard_roi.mean())
    if hazard_ratio > max(0.0, float(config.max_hazard_ratio)):
        return DriveDecision(False, "hazard", safe_ratio, hazard_ratio, bounds)
    if safe_ratio < max(0.0, min(1.0, float(config.min_safe_ratio))):
        return DriveDecision(False, "low_passable", safe_ratio, hazard_ratio, bounds)
    return DriveDecision(True, "clear", safe_ratio, hazard_ratio, bounds)
