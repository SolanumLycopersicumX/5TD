import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from tools.passable_segmentation.prepare_multitask_dataset import (
    prepare_multitask_dataset,
    validate_annotation_labels,
)


class PrepareMultitaskDatasetTest(unittest.TestCase):
    def test_validate_annotation_labels_rejects_unknown_label(self):
        annotation = {
            "imageWidth": 8,
            "imageHeight": 8,
            "shapes": [{"label": "unknown_label", "points": [[0, 0], [1, 1]]}],
        }

        with self.assertRaisesRegex(ValueError, "unknown Labelme labels"):
            validate_annotation_labels(annotation)

    def test_prepare_multitask_dataset_writes_full_and_view_manifests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            batch = root / "batch" / "images"
            output_dir = root / "derived"
            batch.mkdir(parents=True)

            Image.new("RGB", (8, 8), "black").save(batch / "IMG_1_0001.jpg")
            annotation = {
                "imagePath": "IMG_1_0001.jpg",
                "imageWidth": 8,
                "imageHeight": 8,
                "shapes": [
                    {
                        "label": "ego_passable",
                        "shape_type": "polygon",
                        "points": [[1, 1], [6, 1], [6, 6], [1, 6]],
                    },
                    {
                        "label": "worker",
                        "shape_type": "rectangle",
                        "points": [[2, 2], [4, 5]],
                    },
                ],
            }
            (batch / "IMG_1_0001.json").write_text(json.dumps(annotation), encoding="utf-8")

            summary = prepare_multitask_dataset(
                image_dirs=[batch],
                output_dir=output_dir,
                val_prefixes=("IMG",),
            )

            self.assertEqual(summary["total"], 1)
            self.assertTrue((output_dir / "masks" / "ego_passable" / "IMG_1_0001.png").exists())
            self.assertTrue((output_dir / "masks" / "worker" / "IMG_1_0001.png").exists())
            self.assertTrue((output_dir / "val.tsv").exists())
            for view in ("passable", "boundary", "obstacle"):
                self.assertTrue((output_dir / view / "val.tsv").exists())

            passable_cols = (output_dir / "passable" / "val.tsv").read_text(encoding="utf-8").strip().split("\t")
            self.assertEqual(passable_cols[0], "IMG_1_0001")
            self.assertTrue((output_dir / "passable" / passable_cols[1]).exists())
            self.assertTrue((output_dir / "passable" / passable_cols[2]).exists())

    def test_prepare_multitask_dataset_uses_label_prefix_for_validation_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            batch = root / "batch" / "images"
            output_dir = root / "derived"
            batch.mkdir(parents=True)

            Image.new("RGB", (8, 8), "black").save(batch / "IMG_3197_0001.jpg")
            annotation = {
                "imagePath": "IMG_3197_0001.jpg",
                "imageWidth": 8,
                "imageHeight": 8,
                "shapes": [{"label": "ego_passable", "shape_type": "polygon", "points": [[1, 1], [6, 1], [6, 6]]}],
            }
            (batch / "IMG_3197_0001.json").write_text(json.dumps(annotation), encoding="utf-8")

            summary = prepare_multitask_dataset(
                image_dirs=[batch],
                output_dir=output_dir,
                val_prefixes=("IMG_3197",),
            )

            self.assertEqual(summary["train"], 1)
            self.assertEqual(summary["val"], 0)
            self.assertEqual((output_dir / "val.tsv").read_text(encoding="utf-8"), "")


if __name__ == "__main__":
    unittest.main()
