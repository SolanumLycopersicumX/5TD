import tempfile
import unittest
from pathlib import Path

import torch

from tools.passable_segmentation.train_obstacle_semantic import (
    LABELS,
    obstacle_metrics,
    read_obstacle_manifest,
)


class ObstacleSemanticTrainingTest(unittest.TestCase):
    def test_read_manifest_resolves_four_masks_relative_to_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "train.tsv"
            manifest.write_text(
                "frame_001\timages/frame_001.jpg\tmasks/worker.png\t"
                "masks/construction_vehicle.png\tmasks/suspended_object.png\tmasks/debris.png\n",
                encoding="utf-8",
            )

            rows = read_obstacle_manifest(manifest)

        self.assertEqual(
            rows,
            [
                (
                    "frame_001",
                    root / "images/frame_001.jpg",
                    (
                        root / "masks/worker.png",
                        root / "masks/construction_vehicle.png",
                        root / "masks/suspended_object.png",
                        root / "masks/debris.png",
                    ),
                )
            ],
        )

    def test_labels_are_obstacle_classes(self):
        self.assertEqual(LABELS, ("worker", "construction_vehicle", "suspended_object", "debris"))

    def test_obstacle_metrics_report_hazard_iou_for_matching_union(self):
        logits = torch.full((1, len(LABELS), 4, 4), -10.0)
        targets = torch.zeros((1, len(LABELS), 4, 4))
        logits[:, 0, 1:3, 1:3] = 10.0
        logits[:, 2, 2:4, 0:2] = 10.0
        targets[:, 1, 1:3, 1:3] = 1.0
        targets[:, 3, 2:4, 0:2] = 1.0

        metrics = obstacle_metrics(logits, targets)

        self.assertAlmostEqual(metrics["obstacle_hazard_iou"], 1.0)

    def test_obstacle_metrics_report_zero_for_absent_classes(self):
        logits = torch.zeros((1, len(LABELS), 4, 4))
        targets = torch.zeros((1, len(LABELS), 4, 4))

        metrics = obstacle_metrics(logits, targets)

        self.assertEqual(metrics["obstacle_hazard_iou"], 0.0)
        for label in LABELS:
            self.assertEqual(metrics[f"{label}_iou"], 0.0)
            self.assertEqual(metrics[f"{label}_dice"], 0.0)


if __name__ == "__main__":
    unittest.main()
