#!/usr/bin/env python3
"""Prepare passable-road segmentation masks from Labelme JSON files."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Sequence

from PIL import Image

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.passable_segmentation.common import (
    discover_labelme_records,
    load_labelme_json,
    rasterize_labelme_mask,
    split_records_by_prefix,
)


# ==== TUNABLE PARAMETERS ====
IMAGE_DIR = Path("data/annotation_batches/rgb_keyframes_2026-06-22/images")
OUTPUT_DIR = Path("data/derived/passable_ditch_artifact_2026-06-24")
VAL_PREFIXES = ("test_video",)
EXCLUDE_PREFIXES = ("demo_video",)
LABELS = ("ego_passable", "ditch", "surface_artifact_passable")
# OUTPUT_DIR = Path("data/derived/passable_ditch_2026-06-24")
# LABELS = ("ego_passable", "ditch")
# OUTPUT_DIR = Path("data/derived/passable_ego_2026-06-24")
# LABELS = ("ego_passable",)


# ==== CORE ====
def prepare_dataset(
    *,
    image_dir: Path | str,
    output_dir: Path | str,
    val_prefixes: Sequence[str] = VAL_PREFIXES,
    exclude_prefixes: Sequence[str] = EXCLUDE_PREFIXES,
    label: str = "ego_passable",
    labels: Sequence[str] | None = None,
) -> dict:
    """Create copied images, raster masks, manifests, and summary JSON."""
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    out_images = output_dir / "images"
    out_masks = output_dir / "masks"
    out_images.mkdir(parents=True, exist_ok=True)
    out_masks.mkdir(parents=True, exist_ok=True)
    label_names = tuple(labels or (label,))
    for label_name in label_names:
        (out_masks / label_name).mkdir(parents=True, exist_ok=True)

    records = discover_labelme_records(image_dir, exclude_prefixes=exclude_prefixes)
    train_records, val_records = split_records_by_prefix(records, val_prefixes=val_prefixes)

    manifest_rows: list[tuple[str, Path, tuple[Path, ...]]] = []
    mask_pixels: dict[str, dict[str, int]] = {label_name: {} for label_name in label_names}
    for stem, image_path, annotation_path in records:
        annotation = load_labelme_json(annotation_path)
        image_out = out_images / f"{stem}{image_path.suffix.lower()}"
        shutil.copy2(image_path, image_out)

        mask_paths = []
        for label_name in label_names:
            mask = rasterize_labelme_mask(annotation, label=label_name)
            mask_out = out_masks / label_name / f"{stem}.png"
            Image.fromarray(mask).save(mask_out)
            mask_pixels[label_name][stem] = int((mask > 0).sum())
            mask_paths.append(mask_out)

        if len(label_names) == 1:
            legacy_mask_out = out_masks / f"{stem}.png"
            shutil.copy2(mask_paths[0], legacy_mask_out)
            manifest_rows.append((stem, image_out, (legacy_mask_out,)))
        else:
            manifest_rows.append((stem, image_out, tuple(mask_paths)))

    row_by_stem = {stem: (stem, image, masks) for stem, image, masks in manifest_rows}
    _write_multilabel_manifest(output_dir / "manifest.tsv", manifest_rows, root=output_dir)
    _write_multilabel_manifest(
        output_dir / "train.tsv",
        [row_by_stem[stem] for stem, _, _ in train_records],
        root=output_dir,
    )
    _write_multilabel_manifest(
        output_dir / "val.tsv",
        [row_by_stem[stem] for stem, _, _ in val_records],
        root=output_dir,
    )

    summary = {
        "source_image_dir": str(image_dir),
        "output_dir": str(output_dir),
        "label": label_names[0],
        "labels": list(label_names),
        "excluded_prefixes": list(exclude_prefixes),
        "val_prefixes": list(val_prefixes),
        "total": len(records),
        "train": len(train_records),
        "val": len(val_records),
        "empty_masks": {
            label_name: sorted(stem for stem, pixels in label_pixels.items() if pixels == 0)
            for label_name, label_pixels in mask_pixels.items()
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def main() -> None:
    """Run dataset preparation with the tunable parameters above."""
    summary = prepare_dataset(
        image_dir=IMAGE_DIR,
        output_dir=OUTPUT_DIR,
        val_prefixes=VAL_PREFIXES,
        exclude_prefixes=EXCLUDE_PREFIXES,
        labels=LABELS,
    )
    print("[OK] Prepared segmentation dataset")
    _print_json("[DATA]", summary, ensure_ascii=False)


# ==== HELPERS ====
def _write_multilabel_manifest(
    path: Path | str,
    rows: Sequence[tuple[str, Path, tuple[Path, ...]]],
    *,
    root: Path,
) -> None:
    path = Path(path)
    lines = []
    for stem, image_path, mask_paths in rows:
        image_rel = image_path.resolve().relative_to(root.resolve())
        mask_rels = [mask_path.resolve().relative_to(root.resolve()) for mask_path in mask_paths]
        parts = [stem, image_rel.as_posix(), *[mask_rel.as_posix() for mask_rel in mask_rels]]
        lines.append("\t".join(parts) + "\n")
    path.write_text("".join(lines), encoding="utf-8")


def _print_json(prefix: str, data: dict, *, ensure_ascii: bool = True) -> None:
    for line in json.dumps(data, indent=2, ensure_ascii=ensure_ascii).splitlines():
        print(f"{prefix} {line}")


# ==== TEST ====
if __name__ == "__main__":
    main()
