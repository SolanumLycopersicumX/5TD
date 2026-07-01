from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from tools.passable_segmentation.evaluate_recorded_videos import (
    discover_videos,
    mask_ratios,
    sample_frame_indices,
)


class EvaluateRecordedVideosTest(unittest.TestCase):
    def test_sample_frame_indices_uses_requested_sample_rate(self) -> None:
        self.assertEqual(
            sample_frame_indices(total_frames=300, source_fps=30.0, sample_fps=1.0),
            [0, 30, 60, 90, 120, 150, 180, 210, 240, 270],
        )

    def test_mask_ratios_reports_fraction_of_pixels(self) -> None:
        fused = {
            "safe_passable": np.array([[True, False], [True, False]]),
            "ditch": np.array([[False, False], [False, True]]),
            "left_barrier": np.array([[False, True], [False, False]]),
            "tunnel_wall": np.array([[False, False], [True, True]]),
        }

        self.assertEqual(
            mask_ratios(fused),
            {
                "safe_passable_ratio": 0.5,
                "ditch_ratio": 0.25,
                "left_barrier_ratio": 0.25,
                "tunnel_wall_ratio": 0.5,
            },
        )

    def test_discover_videos_finds_camera_mkv_files_in_session_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            first = tmp_path / "2026-06-28_14-19-48"
            second = tmp_path / "2026-06-28_14-25-10"
            first.mkdir()
            second.mkdir()
            (second / "camera_1.mkv").write_bytes(b"video")
            (first / "camera_0.mkv").write_bytes(b"video")
            (first / "notes.txt").write_text("ignore", encoding="utf-8")

            self.assertEqual(discover_videos(tmp_path), [first / "camera_0.mkv", second / "camera_1.mkv"])


if __name__ == "__main__":
    unittest.main()
