#!/usr/bin/env python3
"""Prepare passable, boundary, and obstacle segmentation datasets from Labelme JSON files."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Sequence

from PIL import Image

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.passable_segmentation.common import (
    label_prefix,
    load_labelme_json,
    rasterize_labelme_mask,
)


# ==== TUNABLE PARAMETERS ====
LABELS = (
    "ego_passable",
    "ditch",
    "left_barrier",
    "right_barrier",
    "tunnel_wall",
    "worker",
    "construction_vehicle",
    "suspended_object",
    "debris",
    "surface_artifact_passable",
)
PASSABLE_LABELS = ("ego_passable", "ditch", "surface_artifact_passable")
BOUNDARY_LABELS = ("left_barrier", "right_barrier", "tunnel_wall")
OBSTACLE_LABELS = ("worker", "construction_vehicle", "suspended_object", "debris")
DEFAULT_IMAGE_DIRS = (
    Path("data/annotation_batches/rgb_keyframes_2026-06-22/images"),
    Path("data/annotation_batches/rgb_keyframes_2026-06-24_more_keyframes/images"),
    Path("data/annotation_batches/rgb_keyframes_2026-06-29_videos/images"),
)
DEFAULT_OUTPUT_DIR = Path("data/derived/passable_boundary_obstacle_2026-06-29")
DEFAULT_VAL_PREFIXES = ("IMG_3197", "b0c37d")
DEFAULT_EXCLUDE_PREFIXES = ("demo_video", "test_video")
VIEWS = {
    "passable": PASSABLE_LABELS,
    "boundary": BOUNDARY_LABELS,
    "obstacle": OBSTACLE_LABELS,
}

Record = tuple[str, Path, Path, str]
InternalRecord = tuple[str, str, Path, Path, str]
ManifestRow = tuple[str, Path, tuple[Path, ...]]


# ==== CORE ====
def validate_annotation_labels(annotation: dict, *, allowed_labels: Sequence[str] = LABELS) -> None:
    """Raise when a Labelme annotation contains unknown labels or unsupported shapes."""
    allowed = set(allowed_labels)
    unknown = sorted(
        {shape.get("label") for shape in annotation.get("shapes", []) if shape.get("label") not in allowed},
        key=str,
    )
    if unknown:
        raise ValueError(f"unknown Labelme labels: {unknown}")

    supported_shape_types = {"polygon", "rectangle"}
    unsupported = sorted(
        {
            f"{shape.get('label')}:{shape.get('shape_type') or 'polygon'}"
            for shape in annotation.get("shapes", [])
            if shape.get("label") in allowed
            and (shape.get("shape_type") or "polygon") not in supported_shape_types
        }
    )
    if unsupported:
        raise ValueError(f"unsupported Labelme shape types: {unsupported}")


def discover_records(image_dirs: Sequence[Path | str]) -> list[Record]:
    """Find jpg/jpeg Labelme image/json pairs across image directories."""
    return [
        (output_stem, image_path, annotation_path, batch_name)
        for output_stem, _source_stem, image_path, annotation_path, batch_name in _discover_records_with_source_stems(image_dirs)
    ]


def _discover_records_with_source_stems(image_dirs: Sequence[Path | str]) -> list[InternalRecord]:
    records: list[InternalRecord] = []
    used_stems: set[str] = set()
    seen_source_stems: set[str] = set()

    for image_dir_like in image_dirs:
        image_dir = Path(image_dir_like)
        if not image_dir.exists():
            continue
        batch_name = image_dir.parent.name
        image_paths = sorted(
            path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"}
        )
        for image_path in image_paths:
            annotation_path = image_path.with_suffix(".json")
            if not annotation_path.exists():
                continue

            source_stem = image_path.stem
            output_stem = source_stem
            if source_stem in seen_source_stems or output_stem in used_stems:
                output_stem = f"{batch_name}_{source_stem}"
                suffix = 2
                while output_stem in used_stems:
                    output_stem = f"{batch_name}_{source_stem}_{suffix}"
                    suffix += 1

            seen_source_stems.add(source_stem)
            used_stems.add(output_stem)
            records.append((output_stem, source_stem, image_path, annotation_path, batch_name))

    return records


def prepare_multitask_dataset(
    *,
    image_dirs: Sequence[Path | str] = DEFAULT_IMAGE_DIRS,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    val_prefixes: Sequence[str] = DEFAULT_VAL_PREFIXES,
    exclude_prefixes: Sequence[str] = DEFAULT_EXCLUDE_PREFIXES,
    labels: Sequence[str] = LABELS,
) -> dict:
    """Create copied images, raster masks, manifests, view manifests, and summary JSON."""
    output_dir = Path(output_dir)
    label_names = tuple(labels)
    discovered_records = _discover_records_with_source_stems(image_dirs)
    if not discovered_records:
        raise RuntimeError("No annotated image records were discovered.")

    exclude_set = set(exclude_prefixes)
    records = [
        record
        for record in discovered_records
        if not _matches_stem_policy(record[1], exclude_set)
    ]
    if not records:
        raise RuntimeError("No annotated image records were discovered after applying exclusions.")

    out_images = output_dir / "images"
    out_masks = output_dir / "masks"
    out_images.mkdir(parents=True, exist_ok=True)
    out_masks.mkdir(parents=True, exist_ok=True)

    for label_name in label_names:
        (out_masks / label_name).mkdir(parents=True, exist_ok=True)

    val_set = set(val_prefixes)
    rows: list[ManifestRow] = []
    train_rows: list[ManifestRow] = []
    val_rows: list[ManifestRow] = []
    mask_pixels: dict[str, dict[str, int]] = {label_name: {} for label_name in label_names}
    surface_artifact_outside_ego: list[dict[str, int | str]] = []

    for output_stem, source_stem, image_path, annotation_path, _batch_name in records:
        annotation = load_labelme_json(annotation_path)
        validate_annotation_labels(annotation, allowed_labels=label_names)

        image_out = out_images / f"{output_stem}{image_path.suffix.lower()}"
        shutil.copy2(image_path, image_out)

        mask_paths = []
        row_masks = {}
        for label_name in label_names:
            mask = rasterize_labelme_mask(annotation, label=label_name)
            row_masks[label_name] = mask
            mask_out = out_masks / label_name / f"{output_stem}.png"
            Image.fromarray(mask).save(mask_out)
            mask_pixels[label_name][output_stem] = int((mask > 0).sum())
            mask_paths.append(mask_out)

        artifact_mask = row_masks.get("surface_artifact_passable")
        ego_mask = row_masks.get("ego_passable")
        if artifact_mask is not None and ego_mask is not None:
            outside_pixels = int(((artifact_mask > 0) & ~(ego_mask > 0)).sum())
            if outside_pixels > 0:
                surface_artifact_outside_ego.append({"stem": output_stem, "pixels": outside_pixels})

        row = (output_stem, image_out, tuple(mask_paths))
        rows.append(row)
        if _is_validation_stem(source_stem, val_set):
            val_rows.append(row)
        else:
            train_rows.append(row)

    write_full_manifest(output_dir / "manifest.tsv", rows, root=output_dir)
    write_full_manifest(output_dir / "train.tsv", train_rows, root=output_dir)
    write_full_manifest(output_dir / "val.tsv", val_rows, root=output_dir)
    write_view_manifests(output_dir, train_rows=train_rows, val_rows=val_rows, labels=label_names)

    summary = {
        "source_image_dirs": [str(Path(image_dir)) for image_dir in image_dirs],
        "output_dir": str(output_dir),
        "labels": list(label_names),
        "val_prefixes": list(val_prefixes),
        "exclude_prefixes": list(exclude_prefixes),
        "view_labels": {view_name: list(view_labels) for view_name, view_labels in VIEWS.items()},
        "discovered": len(discovered_records),
        "excluded": len(discovered_records) - len(records),
        "total": len(rows),
        "train": len(train_rows),
        "val": len(val_rows),
        "surface_artifact_outside_ego": surface_artifact_outside_ego,
        "empty_masks": {
            label_name: sorted(stem for stem, pixels in label_pixels.items() if pixels == 0)
            for label_name, label_pixels in mask_pixels.items()
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def write_full_manifest(path: Path | str, rows: Sequence[ManifestRow], *, root: Path) -> None:
    """Write a full multitask manifest with all mask columns."""
    _write_manifest(path, rows, root=root)


def write_view_manifests(
    output_dir: Path | str,
    *,
    train_rows: Sequence[ManifestRow],
    val_rows: Sequence[ManifestRow],
    labels: Sequence[str] = LABELS,
) -> None:
    """Write train/val manifests for passable, boundary, and obstacle views."""
    output_dir = Path(output_dir)
    label_names = tuple(labels)
    for view_name, view_labels in VIEWS.items():
        view_dir = output_dir / view_name
        view_dir.mkdir(parents=True, exist_ok=True)
        write_view_manifest(view_dir / "train.tsv", train_rows, root=view_dir, labels=label_names, view_labels=view_labels)
        write_view_manifest(view_dir / "val.tsv", val_rows, root=view_dir, labels=label_names, view_labels=view_labels)


def write_view_manifest(
    path: Path | str,
    rows: Sequence[ManifestRow],
    *,
    root: Path,
    labels: Sequence[str] = LABELS,
    view_labels: Sequence[str],
) -> None:
    """Write one view manifest with paths relative to the manifest directory."""
    indexes = [tuple(labels).index(label_name) for label_name in view_labels]
    view_rows = [
        (stem, image_path, tuple(mask_paths[index] for index in indexes))
        for stem, image_path, mask_paths in rows
    ]
    _write_manifest(path, view_rows, root=root)


def main(argv: Sequence[str] | None = None) -> None:
    """Run dataset preparation from CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", action="append", type=Path, dest="image_dirs")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--val-prefix", action="append", dest="val_prefixes")
    parser.add_argument("--exclude-prefix", action="append", dest="exclude_prefixes")
    args = parser.parse_args(argv)

    summary = prepare_multitask_dataset(
        image_dirs=tuple(args.image_dirs) if args.image_dirs else DEFAULT_IMAGE_DIRS,
        output_dir=args.output_dir,
        val_prefixes=tuple(args.val_prefixes) if args.val_prefixes else DEFAULT_VAL_PREFIXES,
        exclude_prefixes=tuple(args.exclude_prefixes) if args.exclude_prefixes else DEFAULT_EXCLUDE_PREFIXES,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


# ==== HELPERS ====
def _write_manifest(path: Path | str, rows: Sequence[ManifestRow], *, root: Path) -> None:
    path = Path(path)
    lines = []
    for stem, image_path, mask_paths in rows:
        image_rel = _relative_path(image_path, root)
        mask_rels = [_relative_path(mask_path, root) for mask_path in mask_paths]
        parts = [stem, image_rel, *mask_rels]
        lines.append("\t".join(parts) + "\n")
    path.write_text("".join(lines), encoding="utf-8")


def _relative_path(path: Path, root: Path) -> str:
    return os.path.relpath(path.resolve(), start=root.resolve()).replace(os.sep, "/")


def _is_validation_stem(source_stem: str, val_prefixes: set[str]) -> bool:
    return _matches_stem_policy(source_stem, val_prefixes)


def _matches_stem_policy(source_stem: str, prefixes: set[str]) -> bool:
    if label_prefix(source_stem) in prefixes:
        return True
    for prefix in prefixes:
        if source_stem == prefix or source_stem.startswith(f"{prefix}_"):
            return True
    return False


if __name__ == "__main__":
    main()
