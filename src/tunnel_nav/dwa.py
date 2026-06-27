"""Conservative DWA-style local planning over a BEV risk grid."""
from __future__ import annotations

import numpy as np

from .motion import BEVGrid, DWAConfig, Trajectory


def metric_to_grid_cell(grid: BEVGrid, x_m: float, y_m: float) -> tuple[int, int]:
    """Convert local metric coordinates to BEV grid row/column."""
    if x_m < grid.x_min_m or x_m >= grid.x_max_m or y_m < grid.y_min_m or y_m >= grid.y_max_m:
        raise ValueError("Point is outside BEV grid.")
    col = int((x_m - grid.x_min_m) / grid.resolution_m)
    row_from_near = int((y_m - grid.y_min_m) / grid.resolution_m)
    row = grid.occupancy.shape[0] - 1 - row_from_near
    return row, col


def sample_controls(config: DWAConfig) -> list[tuple[float, float]]:
    """Sample low-speed linear and angular controls."""
    velocities = np.linspace(config.min_velocity_mps, config.max_velocity_mps, config.velocity_samples)
    angulars = np.linspace(-config.max_angular_radps, config.max_angular_radps, config.angular_samples)
    return [(float(v), float(w)) for v in velocities for w in angulars]


def simulate_trajectory(linear_mps: float, angular_radps: float, config: DWAConfig) -> np.ndarray:
    """Forward simulate one differential-drive trajectory from the vehicle origin."""
    steps = max(1, int(round(config.predict_time_s / config.dt_s)))
    x_m = 0.0
    y_m = 0.0
    yaw = 0.0
    points = [(x_m, y_m, yaw)]
    for _ in range(steps):
        x_m += linear_mps * np.sin(yaw) * config.dt_s
        y_m += linear_mps * np.cos(yaw) * config.dt_s
        yaw += angular_radps * config.dt_s
        points.append((x_m, y_m, yaw))
    return np.asarray(points, dtype=np.float32)


def _clearance_to_occupied(grid: BEVGrid, cells: list[tuple[int, int]]) -> float:
    occupied = np.argwhere(grid.occupancy)
    if occupied.size == 0:
        return grid.width_m
    min_cells = grid.occupancy.shape[0] + grid.occupancy.shape[1]
    for row, col in cells:
        distances = np.abs(occupied[:, 0] - row) + np.abs(occupied[:, 1] - col)
        min_cells = min(min_cells, int(distances.min()))
    return float(min_cells * grid.resolution_m)


def evaluate_trajectory(
    points: np.ndarray,
    *,
    linear_mps: float,
    angular_radps: float,
    grid: BEVGrid,
) -> Trajectory:
    """Evaluate one trajectory against occupancy and risk constraints."""
    cells: list[tuple[int, int]] = []
    risks: list[float] = []
    for x_m, y_m, _yaw in points:
        try:
            row, col = metric_to_grid_cell(grid, float(x_m), float(y_m))
        except ValueError:
            return Trajectory(
                points=points,
                linear_mps=linear_mps,
                angular_radps=angular_radps,
                score=-1e9,
                max_risk=1.0,
                min_clearance_m=0.0,
                feasible=False,
                reason="outside grid",
            )
        cells.append((row, col))
        risks.append(float(grid.risk[row, col]))
        if grid.occupancy[row, col]:
            return Trajectory(
                points=points,
                linear_mps=linear_mps,
                angular_radps=angular_radps,
                score=-1e9,
                max_risk=max(risks),
                min_clearance_m=0.0,
                feasible=False,
                reason="occupied cell",
            )

    max_risk = max(risks) if risks else 1.0
    clearance_m = _clearance_to_occupied(grid, cells)
    progress_m = float(points[-1, 1])
    smoothness_penalty = abs(float(angular_radps)) * 0.05
    clearance_bonus = min(clearance_m, 2.0) * 0.2
    risk_penalty = max_risk * 2.0
    score = progress_m + clearance_bonus - risk_penalty - smoothness_penalty
    return Trajectory(
        points=points,
        linear_mps=linear_mps,
        angular_radps=angular_radps,
        score=float(score),
        max_risk=float(max_risk),
        min_clearance_m=float(clearance_m),
        feasible=True,
        reason="ok",
    )


def select_dwa_trajectory(grid: BEVGrid, config: DWAConfig) -> tuple[Trajectory | None, list[Trajectory]]:
    """Select the best feasible DWA candidate trajectory."""
    candidates = []
    for linear_mps, angular_radps in sample_controls(config):
        points = simulate_trajectory(linear_mps, angular_radps, config)
        candidates.append(
            evaluate_trajectory(
                points,
                linear_mps=linear_mps,
                angular_radps=angular_radps,
                grid=grid,
            )
        )

    feasible = [candidate for candidate in candidates if candidate.feasible]
    if not feasible:
        return None, candidates
    feasible.sort(key=lambda candidate: candidate.score, reverse=True)
    return feasible[0], candidates
