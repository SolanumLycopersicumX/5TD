import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.passable_segmentation.evaluate_multitask_videos import (
    FUSED_LABELS,
    discover_videos,
    fuse_multitask_predictions,
    mask_ratios,
    sample_frame_indices,
)


class MultitaskFusionTest(unittest.TestCase):
    def test_fuse_multitask_predictions_builds_hazard_union_and_safe_passable(self):
        passable = np.array(
            [
                [[0.9, 0.9], [0.2, 0.2]],
                [[0.1, 0.8], [0.1, 0.1]],
            ],
            dtype=np.float32,
        )
        boundary = np.array(
            [
                [[0.1, 0.1], [0.8, 0.1]],
                [[0.1, 0.1], [0.1, 0.8]],
                [[0.1, 0.1], [0.1, 0.1]],
            ],
            dtype=np.float32,
        )
        obstacle = np.array(
            [
                [[0.1, 0.1], [0.1, 0.1]],
                [[0.1, 0.7], [0.1, 0.1]],
                [[0.1, 0.1], [0.1, 0.1]],
                [[0.1, 0.1], [0.9, 0.1]],
            ],
            dtype=np.float32,
        )

        fused = fuse_multitask_predictions(passable, boundary, obstacle)

        expected_hazard = np.array([[False, True], [True, True]])
        expected_safe = np.array([[True, False], [False, False]])
        self.assertEqual(tuple(fused), FUSED_LABELS)
        np.testing.assert_array_equal(fused["hazard"], expected_hazard)
        np.testing.assert_array_equal(fused["safe_passable"], expected_safe)

    def test_mask_ratios_reports_every_fused_label(self):
        fused = {label: np.zeros((2, 2), dtype=bool) for label in FUSED_LABELS}
        fused["hazard"][:, 0] = True
        fused["safe_passable"][0, 0] = True
        fused["construction_vehicle"][1, 1] = True

        ratios = mask_ratios(fused)

        self.assertEqual(set(ratios), {f"{label}_ratio" for label in FUSED_LABELS})
        self.assertEqual(ratios["hazard_ratio"], 0.5)
        self.assertEqual(ratios["safe_passable_ratio"], 0.25)
        self.assertEqual(ratios["construction_vehicle_ratio"], 0.25)

    def test_discover_videos_finds_generic_nested_mov_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "session").mkdir()
            generic = root / "session" / "job_site.MOV"
            camera = root / "camera_front.mp4"
            ignored = root / "notes.txt"
            generic.touch()
            camera.touch()
            ignored.touch()

            videos = discover_videos(root)

        self.assertEqual(videos, sorted([camera, generic]))

    def test_sample_frame_indices_uses_basic_one_fps_step(self):
        self.assertEqual(sample_frame_indices(total_frames=95, source_fps=30.0, sample_fps=1.0), [0, 30, 60, 90])
        self.assertEqual(sample_frame_indices(total_frames=0, source_fps=30.0, sample_fps=1.0), [])


if __name__ == "__main__":
    unittest.main()
