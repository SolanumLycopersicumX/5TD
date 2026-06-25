"""Image-space trajectory generation from fused passable-road segmentation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from src.tunnel_nav.vision_autodrive import DriveDecision, DriveGateConfig, evaluate_drive_gate


@dataclass(frozen=True)
class TrajectoryConfig:
    """Parameters for centerline extraction and low-speed steering."""

    scan_y_min: float = 0.48
    scan_y_max: float = 0.92
    scan_count: int = 7
    row_band: int = 5
    min_segment_width_fraction: float = 0.08
    min_points: int = 3
    lookahead_index: int = 3
    center_deadband: float = 0.04
    angular_gain: float = 0.18
    max_angular_radps: float = 0.08


@dataclass(frozen=True)
class TrajectoryPoint:
    """One image-space centerline point."""

    x: int
    y: int


@dataclass(frozen=True)
class TrajectoryCommand:
    """Velocity command derived from the current segmentation trajectory."""

    allow_motion: bool
    reason: str
    linear_mps: float
    angular_radps: float
    target_error: float
    target_point: TrajectoryPoint | None
    points: tuple[TrajectoryPoint, ...]
    gate_decision: DriveDecision


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _largest_true_segment(row: np.ndarray) -> tuple[int, int] | None:
    true_indices = np.flatnonzero(row.astype(bool, copy=False))
    if true_indices.size == 0:
        return None

    breaks = np.where(np.diff(true_indices) > 1)[0]
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks, [true_indices.size - 1]))

    best_start = int(true_indices[starts[0]])
    best_end = int(true_indices[ends[0]])
    best_width = best_end - best_start + 1
    for start_idx, end_idx in zip(starts[1:], ends[1:]):
        segment_start = int(true_indices[start_idx])
        segment_end = int(true_indices[end_idx])
        width = segment_end - segment_start + 1
        if width > best_width:
            best_start = segment_start
            best_end = segment_end
            best_width = width
    return best_start, best_end


def extract_centerline(safe_passable: np.ndarray, config: TrajectoryConfig) -> tuple[TrajectoryPoint, ...]:
    """Extract bottom-to-top centerline points from a binary safe-passable mask."""
    if safe_passable.ndim != 2:
        raise ValueError(f"safe_passable mask must be 2-D, got shape {safe_passable.shape}")
    height, width = safe_passable.shape
    if height <= 0 or width <= 0:
        return ()

    scan_count = max(1, int(config.scan_count))
    y_min = _clamp(config.scan_y_min, 0.0, 1.0)
    y_max = _clamp(config.scan_y_max, 0.0, 1.0)
    if y_min > y_max:
        y_min, y_max = y_max, y_min

    y_bottom = int(round(y_max * (height - 1)))
    y_top = int(round(y_min * (height - 1)))
    y_values = np.linspace(y_bottom, y_top, scan_count).astype(int)
    half_band = max(0, int(config.row_band) // 2)
    min_width = max(1, int(round(_clamp(config.min_segment_width_fraction, 0.0, 1.0) * width)))
    mask = safe_passable.astype(bool, copy=False)

    points: list[TrajectoryPoint] = []
    for y in y_values:
        y0 = max(0, int(y) - half_band)
        y1 = min(height, int(y) + half_band + 1)
        row = mask[y0:y1, :].any(axis=0)
        segment = _largest_true_segment(row)
        if segment is None:
            continue
        x0, x1 = segment
        if x1 - x0 + 1 < min_width:
            continue
        points.append(TrajectoryPoint(x=int(round((x0 + x1) / 2.0)), y=int(y)))
    return tuple(points)


def _select_target(points: tuple[TrajectoryPoint, ...], config: TrajectoryConfig) -> TrajectoryPoint | None:
    if len(points) < max(1, int(config.min_points)):
        return None
    index = min(len(points) - 1, max(0, int(config.lookahead_index)))
    return points[index]


def _angular_from_target(target: TrajectoryPoint, width: int, config: TrajectoryConfig) -> tuple[float, float]:
    image_center = (width - 1) / 2.0
    half_width = max(1.0, width / 2.0)
    error = (float(target.x) - image_center) / half_width
    deadband = _clamp(config.center_deadband, 0.0, 1.0)
    if abs(error) <= deadband:
        return 0.0, error

    max_angular = max(0.0, float(config.max_angular_radps))
    angular = -float(config.angular_gain) * error
    angular = _clamp(angular, -max_angular, max_angular)
    return round(angular, 6), round(error, 6)


def compute_trajectory_command(
    fused: Mapping[str, np.ndarray],
    gate_config: DriveGateConfig,
    trajectory_config: TrajectoryConfig,
    *,
    base_linear_mps: float,
) -> TrajectoryCommand:
    """Compute low-speed linear/angular command from fused segmentation masks."""
    gate = evaluate_drive_gate(fused, gate_config)
    safe_mask = fused["safe_passable"].astype(bool, copy=False)
    points = extract_centerline(safe_mask, trajectory_config)

    if not gate.allow_forward:
        return TrajectoryCommand(False, gate.reason, 0.0, 0.0, 0.0, None, points, gate)

    target = _select_target(points, trajectory_config)
    if target is None:
        return TrajectoryCommand(False, "path_lost", 0.0, 0.0, 0.0, None, points, gate)

    angular, error = _angular_from_target(target, safe_mask.shape[1], trajectory_config)
    linear = max(0.0, float(base_linear_mps))
    return TrajectoryCommand(True, "clear", round(linear, 6), angular, error, target, points, gate)
