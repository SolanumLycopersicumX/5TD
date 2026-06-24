"""Pseudo-BEV occupancy and risk grid generation from fused masks."""
from __future__ import annotations

import numpy as np

from .motion import BEVGrid, MaskBundle


def dilate_binary(mask: np.ndarray, radius_cells: int) -> np.ndarray:
    """Dilate a binary mask with a square kernel using deterministic numpy logic."""
    mask = mask.astype(bool)
    if radius_cells <= 0 or not mask.any():
        return mask.copy()

    padded = np.pad(mask, radius_cells, mode="constant", constant_values=False)
    dilated = np.zeros_like(mask, dtype=bool)
    size = radius_cells * 2 + 1
    for dy in range(size):
        for dx in range(size):
            dilated |= padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
    return dilated


def _sample_mask_to_grid(
    mask: np.ndarray,
    *,
    rows: int,
    cols: int,
    lateral_margin_fraction: float,
) -> np.ndarray:
    """Sample image-space mask into a pseudo-BEV grid.

    This is not calibrated IPM. It gives the planner a stable offline grid
    while preserving the image bottom as the near-vehicle side.
    """
    height, width = mask.shape
    row_fraction = np.linspace(0.0, 1.0, rows)
    src_y = np.clip(np.rint(row_fraction * (height - 1)).astype(int), 0, height - 1)

    margin = width * lateral_margin_fraction
    src_x_float = np.linspace(margin, width - 1 - margin, cols)
    src_x = np.clip(np.rint(src_x_float).astype(int), 0, width - 1)
    return mask[src_y[:, None], src_x[None, :]].astype(bool)


def build_pseudo_bev_grid(
    bundle: MaskBundle,
    *,
    x_min_m: float = -2.5,
    x_max_m: float = 2.5,
    y_min_m: float = 0.0,
    y_max_m: float = 8.0,
    resolution_m: float = 0.1,
    boundary_dilation_m: float = 0.3,
    lateral_margin_fraction: float = 0.05,
) -> BEVGrid:
    """Build a conservative pseudo-BEV occupancy and risk grid.

    Occupancy is true for any forbidden or unknown cell. Risk is continuous,
    with hard boundaries and non-passable areas at 1.0.
    """
    rows = int(round((y_max_m - y_min_m) / resolution_m))
    cols = int(round((x_max_m - x_min_m) / resolution_m))
    if rows <= 0 or cols <= 0:
        raise ValueError("BEV extent and resolution must produce a non-empty grid.")

    safe = _sample_mask_to_grid(
        bundle.safe_passable,
        rows=rows,
        cols=cols,
        lateral_margin_fraction=lateral_margin_fraction,
    )
    ditch = _sample_mask_to_grid(
        bundle.ditch,
        rows=rows,
        cols=cols,
        lateral_margin_fraction=lateral_margin_fraction,
    )
    wall = _sample_mask_to_grid(
        bundle.tunnel_wall,
        rows=rows,
        cols=cols,
        lateral_margin_fraction=lateral_margin_fraction,
    )

    hard_boundary = ditch | wall
    free = safe & ~hard_boundary
    occupancy = ~free

    risk = np.full((rows, cols), 0.10, dtype=np.float32)
    risk[occupancy] = 1.0

    dilation_cells = int(round(boundary_dilation_m / resolution_m))
    keepout_band = dilate_binary(hard_boundary, dilation_cells) & ~hard_boundary & ~occupancy
    risk[keepout_band] = np.maximum(risk[keepout_band], 0.75)

    return BEVGrid(
        occupancy=occupancy,
        risk=risk,
        x_min_m=x_min_m,
        x_max_m=x_max_m,
        y_min_m=y_min_m,
        y_max_m=y_max_m,
        resolution_m=resolution_m,
    )
