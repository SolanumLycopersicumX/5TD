#!/usr/bin/env python3
"""Shared Labelme helpers for passable-road segmentation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw


# ==== TYPES ====
LabelmeRecord = tuple[str, Path, Path]


# ==== CORE ====
def label_prefix(stem: str) -> str:
    """Return the dataset split prefix for one image stem."""
    if stem.startswith("demo_video_"):
        return "demo_video"
    if stem.startswith("test_video_"):
        return "test_video"
    return stem.split("_", 1)[0]


def discover_labelme_records(
    image_dir: Path | str,
    *,
    exclude_prefixes: Sequence[str] = ("demo_video",),
) -> list[LabelmeRecord]:
    """Find image/json pairs in a Labelme image directory."""
    image_dir = Path(image_dir)
    records: list[LabelmeRecord] = []
    for image_path in sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.jpeg"))):
        stem = image_path.stem
        prefix = label_prefix(stem)
        if prefix in exclude_prefixes:
            continue
        annotation_path = image_path.with_suffix(".json")
        if not annotation_path.exists():
            continue
        records.append((stem, image_path, annotation_path))
    return records


def load_labelme_json(path: Path | str) -> dict:
    """Load one Labelme JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rasterize_labelme_mask(annotation: dict, *, label: str = "ego_passable") -> np.ndarray:
    """Rasterize one Labelme label into a uint8 mask."""
    width = int(annotation["imageWidth"])
    height = int(annotation["imageHeight"])
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    for shape in annotation.get("shapes", []):
        if shape.get("label") != label:
            continue
        shape_type = shape.get("shape_type", "polygon")
        points = shape.get("points", [])
        if shape_type == "polygon" and len(points) >= 3:
            draw.polygon(_clip_points(points, width, height), fill=255)
        elif shape_type == "rectangle" and len(points) == 2:
            (x1, y1), (x2, y2) = _clip_points(points, width, height)
            left, right = sorted((x1, x2))
            top, bottom = sorted((y1, y2))
            draw.rectangle([left, top, right, bottom], fill=255)

    return np.array(mask, dtype=np.uint8)


def split_records_by_prefix(
    records: Sequence[LabelmeRecord],
    *,
    val_prefixes: Sequence[str] = (),
) -> tuple[list[LabelmeRecord], list[LabelmeRecord]]:
    """Split records by validation prefixes."""
    val_set = set(val_prefixes)
    train: list[LabelmeRecord] = []
    val: list[LabelmeRecord] = []
    for record in records:
        stem, _, _ = record
        if label_prefix(stem) in val_set:
            val.append(record)
        else:
            train.append(record)
    return train, val


def write_manifest(path: Path | str, records: Iterable[tuple[str, Path, Path]], *, root: Path) -> None:
    """Write a three-column TSV manifest."""
    path = Path(path)
    lines = []
    for stem, image_path, mask_path in records:
        image_rel = image_path.resolve().relative_to(root.resolve())
        mask_rel = mask_path.resolve().relative_to(root.resolve())
        lines.append(f"{stem}\t{image_rel.as_posix()}\t{mask_rel.as_posix()}\n")
    path.write_text("".join(lines), encoding="utf-8")


# ==== HELPERS ====
def _clip_points(points: Sequence[Sequence[float]], width: int, height: int) -> list[tuple[float, float]]:
    clipped = []
    for x, y in points:
        clipped.append((min(max(float(x), 0.0), width - 1), min(max(float(y), 0.0), height - 1)))
    return clipped
