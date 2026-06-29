import tempfile
import unittest
from pathlib import Path

import torch

from tools.passable_segmentation.train_boundary_right_wall import (
    LABELS,
    copy_compatible_state,
    read_boundary_right_wall_manifest,
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

    def test_copy_compatible_state_loads_two_output_head_into_three_output_model(self):
        torch.manual_seed(123)
        source = SmallPassableUNet(base_channels=4, out_channels=2)
        target = SmallPassableUNet(base_channels=4, out_channels=3)
        original_third_weight = target.out.weight[2].detach().clone()
        original_third_bias = target.out.bias[2].detach().clone()

        with torch.no_grad():
            source.out.bias.fill_(0.25)
            source.out.weight.fill_(0.5)

        copy_compatible_state(target, source.state_dict())

        self.assertTrue(torch.equal(target.out.weight[:2], source.out.weight))
        self.assertTrue(torch.equal(target.out.bias[:2], source.out.bias))
        self.assertTrue(torch.equal(target.out.weight[2], original_third_weight))
        self.assertTrue(torch.equal(target.out.bias[2], original_third_bias))

    def test_labels_include_right_barrier_between_left_and_wall(self):
        self.assertEqual(LABELS, ("left_barrier", "right_barrier", "tunnel_wall"))


if __name__ == "__main__":
    unittest.main()
