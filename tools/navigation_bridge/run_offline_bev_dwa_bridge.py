#!/usr/bin/env python3
"""Run the offline pseudo-BEV DWA navigation bridge."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.tunnel_nav.bev import build_pseudo_bev_grid
from src.tunnel_nav.dwa import metric_to_grid_cell, select_dwa_trajectory
from src.tunnel_nav.motion import DWAConfig, MaskBundle, NavigationConfig, Trajectory
from src.tunnel_nav.rs232 import Rs232DryRunAdapter
from src.tunnel_nav.safety import command_from_dwa_result


def _read_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 0


def _load_bundle(mask_dir: Path, stem: str) -> MaskBundle:
    return MaskBundle(
        safe_passable=_read_mask(mask_dir / "safe_passable" / f"{stem}.png"),
        ditch=_read_mask(mask_dir / "ditch" / f"{stem}.png"),
        left_barrier=_read_mask(mask_dir / "left_barrier" / f"{stem}.png"),
        tunnel_wall=_read_mask(mask_dir / "tunnel_wall" / f"{stem}.png"),
        source_frame=stem,
    )


def _collect_images(image_dir: Path) -> list[Path]:
    suffixes = {".jpg", ".jpeg", ".png"}
    return sorted(path for path in image_dir.iterdir() if path.suffix.lower() in suffixes)


def _risk_grid_image(risk: np.ndarray) -> Image.Image:
    risk_u8 = np.clip(risk * 255.0, 0, 255).astype(np.uint8)
    rgb = np.zeros((*risk_u8.shape, 3), dtype=np.uint8)
    rgb[..., 0] = risk_u8
    rgb[..., 1] = 180 - np.minimum(risk_u8, 180)
    return Image.fromarray(rgb, mode="RGB")


def _trajectory_cells(grid, trajectory: Trajectory) -> list[tuple[int, int]]:
    cells = []
    for x_m, y_m, _yaw in trajectory.points:
        try:
            row, col = metric_to_grid_cell(grid, float(x_m), float(y_m))
        except ValueError:
            continue
        cells.append((col, row))
    return cells


def _draw_trajectories(grid, candidates: list[Trajectory], selected: Trajectory | None) -> Image.Image:
    canvas = _risk_grid_image(grid.risk)
    draw = ImageDraw.Draw(canvas)
    for candidate in candidates:
        cells = _trajectory_cells(grid, candidate)
        if len(cells) < 2:
            continue
        color = (255, 230, 0) if candidate.feasible else (180, 80, 80)
        draw.line(cells, fill=color, width=1)
    if selected is not None:
        cells = _trajectory_cells(grid, selected)
        if len(cells) >= 2:
            draw.line(cells, fill=(0, 255, 80), width=2)
    return canvas


def _draw_overlay(
    image_path: Path,
    grid,
    candidates: list[Trajectory],
    selected: Trajectory | None,
    command,
    output_path: Path,
) -> None:
    risk_panel = _draw_trajectories(grid, candidates, selected).resize((400, 640), Image.Resampling.NEAREST)
    image_panel = Image.open(image_path).convert("RGB").resize((960, 640))
    canvas = Image.new("RGB", (1360, 640), "black")
    canvas.paste(image_panel, (0, 0))
    canvas.paste(risk_panel, (960, 0))
    draw = ImageDraw.Draw(canvas)
    status = "BRAKE" if command.brake else "DRIVE"
    draw.rectangle((8, 8, 680, 88), fill=(0, 0, 0))
    draw.text(
        (18, 18),
        (
            f"{status} {command.safety_state} "
            f"v={command.linear_mps:.3f}m/s w={command.angular_radps:.3f}rad/s"
        ),
        fill=(255, 255, 255),
    )
    draw.text((18, 48), command.reason, fill=(255, 255, 255))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=95)


def run_offline_bev_dwa_bridge(
    *,
    image_dir: Path | str,
    mask_dir: Path | str,
    output_dir: Path | str,
    max_speed_mps: float = 0.10,
    max_angular_radps: float = 0.50,
    grid_resolution_m: float = 0.10,
    node_addr: int = 0x06,
) -> int:
    """Run the offline bridge over saved fused masks."""
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)
    output_dir = Path(output_dir)
    command_dir = output_dir / "commands"
    rs232_dir = output_dir / "rs232_dry_run"
    overlay_dir = output_dir / "overlays"
    command_dir.mkdir(parents=True, exist_ok=True)
    rs232_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    nav_config = NavigationConfig(max_speed_mps=max_speed_mps, max_angular_radps=max_angular_radps, dry_run=True)
    dwa_config = DWAConfig(max_velocity_mps=max_speed_mps, max_angular_radps=max_angular_radps)
    adapter = Rs232DryRunAdapter(node_addr=node_addr)

    count = 0
    for image_path in _collect_images(image_dir):
        stem = image_path.stem
        bundle = _load_bundle(mask_dir, stem)
        grid = build_pseudo_bev_grid(bundle, resolution_m=grid_resolution_m)
        selected, candidates = select_dwa_trajectory(grid, dwa_config)
        command = command_from_dwa_result(selected, candidates, nav_config, source_frame=stem)
        rs232_record = adapter.send(command, nav_config)

        (command_dir / f"{stem}.json").write_text(
            json.dumps(command.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (rs232_dir / f"{stem}.json").write_text(
            json.dumps(rs232_record, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        _draw_overlay(image_path, grid, candidates, selected, command, overlay_dir / f"{stem}_bev_dwa.jpg")
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--mask-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-speed-mps", type=float, default=0.10)
    parser.add_argument("--max-angular-radps", type=float, default=0.50)
    parser.add_argument("--grid-resolution-m", type=float, default=0.10)
    parser.add_argument("--node-addr", type=lambda value: int(value, 0), default=0x06)
    args = parser.parse_args()
    count = run_offline_bev_dwa_bridge(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        output_dir=args.output_dir,
        max_speed_mps=args.max_speed_mps,
        max_angular_radps=args.max_angular_radps,
        grid_resolution_m=args.grid_resolution_m,
        node_addr=args.node_addr,
    )
    print(f"[OK] Wrote {count} offline BEV/DWA command(s) to {args.output_dir}")


if __name__ == "__main__":
    main()
