import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from tools.passable_segmentation.common import (
    discover_labelme_records,
    rasterize_labelme_mask,
    split_records_by_prefix,
)
from tools.passable_segmentation.prepare_dataset import prepare_dataset
from tools.passable_segmentation.train_passable import (
    SmallPassableUNet,
    binary_segmentation_metrics,
    build_train_config,
)
from tools.passable_segmentation.train_passable_ditch import (
    filter_small_binary_components,
    build_train_config as build_dual_train_config,
    make_dual_overlay,
    postprocess_dual_probabilities,
    safe_passable_metrics,
)
from tools.passable_segmentation.train_passable_ditch_artifact import (
    artifact_safe_metrics,
    passable_ditch_artifact_loss,
)
from tools.passable_segmentation.visualize_passable import collect_image_paths, make_overlay_canvas


class PassableSegmentationToolsTest(unittest.TestCase):
    def test_rasterize_labelme_mask_uses_only_ego_passable_polygon(self):
        annotation = {
            "imageWidth": 10,
            "imageHeight": 8,
            "shapes": [
                {
                    "label": "ego_passable",
                    "shape_type": "polygon",
                    "points": [[1, 1], [8, 1], [8, 6], [1, 6]],
                },
                {
                    "label": "tunnel_wall",
                    "shape_type": "polygon",
                    "points": [[0, 0], [9, 0], [9, 1], [0, 1]],
                },
            ],
        }

        mask = rasterize_labelme_mask(annotation, label="ego_passable")

        self.assertEqual(mask.shape, (8, 10))
        self.assertEqual(mask.dtype, np.uint8)
        self.assertEqual(mask[3, 4], 255)
        self.assertEqual(mask[0, 4], 0)

    def test_discover_records_excludes_demo_video_and_requires_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_dir = Path(tmp)
            for stem in ["scene_0001", "demo_video_0001", "missing_json_0001"]:
                Image.new("RGB", (4, 4), "black").save(image_dir / f"{stem}.jpg")
            for stem in ["scene_0001", "demo_video_0001"]:
                (image_dir / f"{stem}.json").write_text(
                    json.dumps({"imageWidth": 4, "imageHeight": 4, "shapes": []})
                )

            records = discover_labelme_records(image_dir, exclude_prefixes=("demo_video",))

        self.assertEqual([record[0] for record in records], ["scene_0001"])

    def test_split_records_by_prefix_can_hold_out_test_video(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_dir = Path(tmp)
            stems = ["aaa_0001", "aaa_0002", "test_video_0001"]
            records = []
            for stem in stems:
                img = image_dir / f"{stem}.jpg"
                ann = image_dir / f"{stem}.json"
                img.touch()
                ann.touch()
                records.append((stem, img, ann))

            train, val = split_records_by_prefix(records, val_prefixes=("test_video",))

        self.assertEqual([record[0] for record in train], ["aaa_0001", "aaa_0002"])
        self.assertEqual([record[0] for record in val], ["test_video_0001"])

    def test_training_configs_use_top_level_tunables(self):
        binary_config = build_train_config()
        dual_config = build_dual_train_config()

        self.assertEqual(binary_config["dataset_dir"], "data/derived/passable_ego_2026-06-24")
        self.assertEqual(dual_config["dataset_dir"], "data/derived/passable_ditch_2026-06-24")
        self.assertEqual(dual_config["overlap_weight"], 1.5)

    def test_prepare_dataset_writes_masks_and_split_manifests(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            image_dir = tmp_path / "labelme"
            output_dir = tmp_path / "derived"
            image_dir.mkdir()

            for stem in ["scene_0001", "test_video_0001", "demo_video_0001"]:
                Image.new("RGB", (10, 8), "black").save(image_dir / f"{stem}.jpg")
                annotation = {
                    "imagePath": f"{stem}.jpg",
                    "imageWidth": 10,
                    "imageHeight": 8,
                    "shapes": [
                        {
                            "label": "ego_passable",
                            "shape_type": "polygon",
                            "points": [[1, 1], [8, 1], [8, 6], [1, 6]],
                        }
                    ],
                }
                (image_dir / f"{stem}.json").write_text(json.dumps(annotation))

            summary = prepare_dataset(
                image_dir=image_dir,
                output_dir=output_dir,
                val_prefixes=("test_video",),
            )

            self.assertEqual(summary["total"], 2)
            self.assertEqual(summary["train"], 1)
            self.assertEqual(summary["val"], 1)
            self.assertTrue((output_dir / "images/scene_0001.jpg").exists())
            self.assertTrue((output_dir / "masks/scene_0001.png").exists())
            self.assertIn("scene_0001", (output_dir / "train.tsv").read_text())
            self.assertIn("test_video_0001", (output_dir / "val.tsv").read_text())

    def test_prepare_dataset_can_write_ego_passable_and_ditch_masks(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            image_dir = tmp_path / "labelme"
            output_dir = tmp_path / "derived"
            image_dir.mkdir()
            Image.new("RGB", (10, 8), "black").save(image_dir / "scene_0001.jpg")
            annotation = {
                "imagePath": "scene_0001.jpg",
                "imageWidth": 10,
                "imageHeight": 8,
                "shapes": [
                    {
                        "label": "ego_passable",
                        "shape_type": "polygon",
                        "points": [[1, 1], [8, 1], [8, 6], [1, 6]],
                    },
                    {
                        "label": "ditch",
                        "shape_type": "polygon",
                        "points": [[7, 1], [9, 1], [9, 6], [7, 6]],
                    },
                ],
            }
            (image_dir / "scene_0001.json").write_text(json.dumps(annotation))

            summary = prepare_dataset(
                image_dir=image_dir,
                output_dir=output_dir,
                val_prefixes=(),
                labels=("ego_passable", "ditch"),
            )

            self.assertEqual(summary["labels"], ["ego_passable", "ditch"])
            self.assertTrue((output_dir / "masks/ego_passable/scene_0001.png").exists())
            self.assertTrue((output_dir / "masks/ditch/scene_0001.png").exists())
            manifest_cols = (output_dir / "manifest.tsv").read_text().strip().split("\t")
            self.assertEqual(len(manifest_cols), 4)

    def test_small_model_outputs_input_mask_size_and_metrics_are_bounded(self):
        import torch

        model = SmallPassableUNet(base_channels=8)
        logits = model(torch.zeros(2, 3, 64, 96))
        self.assertEqual(tuple(logits.shape), (2, 1, 64, 96))

        metrics = binary_segmentation_metrics(logits, torch.zeros(2, 1, 64, 96))
        self.assertGreaterEqual(metrics["iou"], 0.0)
        self.assertLessEqual(metrics["iou"], 1.0)
        self.assertGreaterEqual(metrics["dice"], 0.0)
        self.assertLessEqual(metrics["dice"], 1.0)

        two_head_model = SmallPassableUNet(base_channels=8, out_channels=2)
        two_head_logits = two_head_model(torch.zeros(2, 3, 64, 96))
        self.assertEqual(tuple(two_head_logits.shape), (2, 2, 64, 96))

    def test_safe_passable_metrics_penalizes_ditch_as_passable(self):
        import torch

        logits = torch.zeros(1, 2, 4, 4)
        logits[:, 0] = 10.0
        targets = torch.zeros(1, 2, 4, 4)
        targets[:, 1, :, 2:] = 1.0

        metrics = safe_passable_metrics(logits, targets)

        self.assertGreater(metrics["ditch_as_passable_rate"], 0.9)

    def test_artifact_metrics_penalize_ditch_on_drivable_surface_artifacts(self):
        import torch

        logits = torch.zeros(1, 2, 4, 4)
        logits[:, 0] = -10.0
        logits[:, 1] = 10.0
        targets = torch.zeros(1, 3, 4, 4)
        targets[:, 0, :, :] = 1.0
        targets[:, 2, :, 1:3] = 1.0

        metrics = artifact_safe_metrics(logits, targets)

        self.assertGreater(metrics["artifact_ditch_false_positive_rate"], 0.9)
        self.assertGreater(metrics["artifact_passable_false_negative_rate"], 0.9)

    def test_artifact_loss_is_higher_when_artifact_is_predicted_as_ditch(self):
        import torch

        targets = torch.zeros(1, 3, 4, 4)
        targets[:, 0, :, :] = 1.0
        targets[:, 2, :, 1:3] = 1.0

        good_logits = torch.zeros(1, 2, 4, 4)
        good_logits[:, 0] = 10.0
        good_logits[:, 1] = -10.0
        bad_logits = torch.zeros(1, 2, 4, 4)
        bad_logits[:, 0] = -10.0
        bad_logits[:, 1] = 10.0

        good_loss = passable_ditch_artifact_loss(good_logits, targets)
        bad_loss = passable_ditch_artifact_loss(bad_logits, targets)

        self.assertGreater(float(bad_loss), float(good_loss) + 1.0)

    def test_artifact_loss_keeps_true_ditch_from_being_marked_passable(self):
        import torch

        targets = torch.zeros(1, 3, 4, 4)
        targets[:, 1, :, 2:] = 1.0

        bad_logits = torch.zeros(1, 2, 4, 4)
        bad_logits[:, 0] = 10.0
        bad_logits[:, 1] = -10.0

        weak_loss = passable_ditch_artifact_loss(
            bad_logits,
            targets,
            ditch_safety_weight=0.0,
            artifact_weight=0.0,
        )
        safe_loss = passable_ditch_artifact_loss(
            bad_logits,
            targets,
            ditch_safety_weight=2.0,
            artifact_weight=0.0,
        )

        self.assertGreater(float(safe_loss), float(weak_loss) + 1.0)

    def test_dual_overlay_can_show_prediction_and_targets(self):
        image = np.zeros((8, 10, 3), dtype=np.uint8)
        probs = np.zeros((2, 8, 10), dtype=np.float32)
        probs[0, :, :5] = 1.0
        probs[1, :, 4:] = 1.0
        target = np.zeros((2, 8, 10), dtype=np.float32)
        target[0, :, :6] = 1.0
        target[1, :, 6:] = 1.0

        canvas = make_dual_overlay(image, probs, target)

        self.assertEqual(canvas.shape, (8, 30, 3))

    def test_filter_small_binary_components_removes_only_tiny_ditch_islands(self):
        mask = np.zeros((8, 10), dtype=bool)
        mask[1, 1] = True
        mask[4:6, 6:9] = True

        filtered = filter_small_binary_components(mask, min_area=4)

        self.assertFalse(filtered[1, 1])
        self.assertTrue(filtered[4:6, 6:9].all())

    def test_postprocess_dual_probabilities_filters_ditch_without_changing_passable(self):
        probs = np.zeros((2, 8, 10), dtype=np.float32)
        probs[0, :, :] = 0.75
        probs[1, 1, 1] = 0.9
        probs[1, 4:6, 6:9] = 0.9

        processed = postprocess_dual_probabilities(probs, min_ditch_area=4)

        np.testing.assert_array_equal(processed[0], probs[0])
        self.assertEqual(float(processed[1, 1, 1]), 0.0)
        self.assertAlmostEqual(float(processed[1, 4, 6]), 0.9)
        self.assertAlmostEqual(float(probs[1, 1, 1]), 0.9)

    def test_visualization_collects_images_and_builds_canvas(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ["b.jpg", "a.png", "ignore.json"]:
                (root / name).touch()

            self.assertEqual([p.name for p in collect_image_paths(root)], ["a.png", "b.jpg"])

        image = np.zeros((8, 10, 3), dtype=np.uint8)
        prob = np.ones((8, 10), dtype=np.float32)
        target = np.zeros((8, 10), dtype=np.uint8)
        canvas = make_overlay_canvas(image, prob, target)
        self.assertEqual(canvas.shape, (8, 30, 3))


if __name__ == "__main__":
    unittest.main()
