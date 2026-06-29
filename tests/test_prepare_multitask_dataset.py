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
            self.assertEqual(
                summary["view_labels"],
                {
                    "passable": ["ego_passable", "ditch", "surface_artifact_passable"],
                    "boundary": ["left_barrier", "right_barrier", "tunnel_wall"],
                    "obstacle": ["worker", "construction_vehicle", "suspended_object", "debris"],
                },
            )
            summary_json = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary_json["view_labels"], summary["view_labels"])
            self.assertTrue((output_dir / "masks" / "ego_passable" / "IMG_1_0001.png").exists())
            self.assertTrue((output_dir / "masks" / "worker" / "IMG_1_0001.png").exists())
            self.assertTrue((output_dir / "val.tsv").exists())
            for view in ("passable", "boundary", "obstacle"):
                self.assertTrue((output_dir / view / "val.tsv").exists())

            full_cols = (output_dir / "val.tsv").read_text(encoding="utf-8").strip().split("	")
            self.assertEqual(len(full_cols), 12)
            self.assertEqual(full_cols[0], "IMG_1_0001")
            self.assertTrue((output_dir / full_cols[1]).exists())
            for rel_path in full_cols[2:]:
                self.assertTrue((output_dir / rel_path).exists())

            expected_view_columns = {"passable": 5, "boundary": 5, "obstacle": 6}
            for view_name, expected_columns in expected_view_columns.items():
                view_dir = output_dir / view_name
                view_cols = (view_dir / "val.tsv").read_text(encoding="utf-8").strip().split("	")
                self.assertEqual(len(view_cols), expected_columns)
                self.assertEqual(view_cols[0], "IMG_1_0001")
                self.assertTrue((view_dir / view_cols[1]).exists())
                for rel_path in view_cols[2:]:
                    self.assertTrue((view_dir / rel_path).exists())

    def test_prepare_multitask_dataset_uses_source_stem_for_validation_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            batch_one = root / "batch_one" / "images"
            batch_two = root / "batch_two" / "images"
            output_dir = root / "derived"
            batch_one.mkdir(parents=True)
            batch_two.mkdir(parents=True)

            Image.new("RGB", (8, 8), "black").save(batch_one / "IMG_3197_0001.jpg")
            Image.new("RGB", (8, 8), "black").save(batch_two / "IMG_3197_0001.jpg")
            annotation = {
                "imagePath": "IMG_3197_0001.jpg",
                "imageWidth": 8,
                "imageHeight": 8,
                "shapes": [{"label": "ego_passable", "shape_type": "polygon", "points": [[1, 1], [6, 1], [6, 6]]}],
            }
            (batch_one / "IMG_3197_0001.json").write_text(json.dumps(annotation), encoding="utf-8")
            (batch_two / "IMG_3197_0001.json").write_text(json.dumps(annotation), encoding="utf-8")

            summary = prepare_multitask_dataset(
                image_dirs=[batch_one, batch_two],
                output_dir=output_dir,
                val_prefixes=("IMG_3197",),
            )

            self.assertEqual(summary["train"], 0)
            self.assertEqual(summary["val"], 2)

            val_rows = (output_dir / "val.tsv").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(val_rows), 2)
            stems = [row.split("	")[0] for row in val_rows]
            self.assertEqual(stems, ["IMG_3197_0001", "batch_two_IMG_3197_0001"])

    def test_prepare_multitask_dataset_rasterizes_reversed_rectangle_points(self):
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
                        "label": "worker",
                        "shape_type": "rectangle",
                        "points": [[5, 6], [2, 2]],
                    },
                ],
            }
            (batch / "IMG_1_0001.json").write_text(json.dumps(annotation), encoding="utf-8")

            prepare_multitask_dataset(
                image_dirs=[batch],
                output_dir=output_dir,
                val_prefixes=("IMG",),
            )

            mask = Image.open(output_dir / "masks" / "worker" / "IMG_1_0001.png")
            self.assertGreater(sum(1 for value in mask.getdata() if value), 0)

    def test_prepare_multitask_dataset_excludes_test_video_holdout_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            batch = root / "batch" / "images"
            output_dir = root / "derived"
            batch.mkdir(parents=True)

            annotation = {
                "imagePath": "placeholder.jpg",
                "imageWidth": 8,
                "imageHeight": 8,
                "shapes": [
                    {
                        "label": "ego_passable",
                        "shape_type": "polygon",
                        "points": [[1, 1], [6, 1], [6, 6]],
                    },
                ],
            }
            for stem in ("IMG_1_0001", "test_video_0001"):
                Image.new("RGB", (8, 8), "black").save(batch / f"{stem}.jpg")
                (batch / f"{stem}.json").write_text(json.dumps(annotation), encoding="utf-8")

            summary = prepare_multitask_dataset(
                image_dirs=[batch],
                output_dir=output_dir,
                val_prefixes=("IMG",),
            )

            self.assertEqual(summary["total"], 1)
            self.assertEqual(summary["excluded"], 1)
            self.assertEqual(summary["exclude_prefixes"], ["demo_video", "test_video"])
            manifest_text = (output_dir / "manifest.tsv").read_text(encoding="utf-8")
            self.assertIn("IMG_1_0001", manifest_text)
            self.assertNotIn("test_video_0001", manifest_text)

    def test_prepare_multitask_dataset_raises_when_no_records_are_discovered(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "derived"
            empty_dir = root / "empty" / "images"
            empty_dir.mkdir(parents=True)

            with self.assertRaisesRegex(RuntimeError, "No annotated image records were discovered"):
                prepare_multitask_dataset(
                    image_dirs=[empty_dir],
                    output_dir=output_dir,
                    val_prefixes=("IMG",),
                )


if __name__ == "__main__":
    unittest.main()
