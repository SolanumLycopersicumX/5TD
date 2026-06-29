import tempfile
import unittest
from pathlib import Path

import torch

from tools.passable_segmentation.train_boundary_right_wall import (
    LABELS,
    boundary_right_wall_metrics,
    copy_compatible_state,
    finalize_boundary_metrics,
    new_boundary_metric_totals,
    read_boundary_right_wall_manifest,
    update_boundary_metric_totals,
)
from tools.passable_segmentation.train_passable import SmallPassableUNet


class BoundaryRightWallTrainingTest(unittest.TestCase):
    def test_read_manifest_resolves_three_masks_relative_to_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "train.tsv"
            manifest.write_text(
                "frame_001\timages/frame_001.jpg\tmasks/left.png\tmasks/right.png\tmasks/wall.png\n",
                encoding="utf-8",
            )

            rows = read_boundary_right_wall_manifest(manifest)

        self.assertEqual(
            rows,
            [
                (
                    "frame_001",
                    root / "images/frame_001.jpg",
                    (root / "masks/left.png", root / "masks/right.png", root / "masks/wall.png"),
                )
            ],
        )

    def test_copy_compatible_state_maps_output_head_by_label(self):
        torch.manual_seed(123)
        source = SmallPassableUNet(base_channels=4, out_channels=2)
        target = SmallPassableUNet(base_channels=4, out_channels=3)
        original_right_weight = target.out.weight[1].detach().clone()
        original_right_bias = target.out.bias[1].detach().clone()

        with torch.no_grad():
            source.out.weight[0].fill_(0.25)
            source.out.weight[1].fill_(0.75)
            source.out.bias[0].fill_(0.5)
            source.out.bias[1].fill_(1.5)

        copy_compatible_state(
            target,
            source.state_dict(),
            source_labels=("left_barrier", "tunnel_wall"),
        )

        self.assertTrue(torch.equal(target.out.weight[0], source.out.weight[0]))
        self.assertTrue(torch.equal(target.out.bias[0], source.out.bias[0]))
        self.assertTrue(torch.equal(target.out.weight[1], original_right_weight))
        self.assertTrue(torch.equal(target.out.bias[1], original_right_bias))
        self.assertTrue(torch.equal(target.out.weight[2], source.out.weight[1]))
        self.assertTrue(torch.equal(target.out.bias[2], source.out.bias[1]))

    def test_boundary_right_wall_metrics_report_zero_for_absent_classes(self):
        logits = torch.zeros((1, len(LABELS), 4, 4))
        targets = torch.zeros((1, len(LABELS), 4, 4))

        metrics = boundary_right_wall_metrics(logits, targets)

        for key, value in metrics.items():
            self.assertEqual(value, 0.0, key)

    def test_boundary_metric_totals_finalize_corpus_ratios_across_sparse_batches(self):
        totals = new_boundary_metric_totals()

        empty_logits = torch.full((1, len(LABELS), 2, 2), -10.0)
        empty_targets = torch.zeros((1, len(LABELS), 2, 2))
        update_boundary_metric_totals(totals, empty_logits, empty_targets)

        positive_logits = torch.full((1, len(LABELS), 2, 2), -10.0)
        positive_targets = torch.zeros((1, len(LABELS), 2, 2))
        positive_logits[:, :, 0, 0] = 10.0
        positive_targets[:, :, 0, 0] = 1.0
        update_boundary_metric_totals(totals, positive_logits, positive_targets)

        metrics = finalize_boundary_metrics(totals)

        for label in LABELS:
            self.assertEqual(metrics[f"{label}_iou"], 1.0)
            self.assertEqual(metrics[f"{label}_dice"], 1.0)

    def test_labels_include_right_barrier_between_left_and_wall(self):
        self.assertEqual(LABELS, ("left_barrier", "right_barrier", "tunnel_wall"))


if __name__ == "__main__":
    unittest.main()
