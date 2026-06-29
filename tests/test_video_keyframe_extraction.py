from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.passable_segmentation.extract_video_keyframes import (
    discover_videos,
    sample_frame_indices,
    video_prefix,
)


class VideoKeyframeExtractionTest(unittest.TestCase):
    def test_discover_videos_finds_mov_mp4_and_mkv_in_stable_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "IMG_3197.MOV").write_bytes(b"video")
            (root / "b.mp4").write_bytes(b"video")
            (root / "a.mkv").write_bytes(b"video")
            (root / "notes.txt").write_text("ignore", encoding="utf-8")

            self.assertEqual(
                [p.name for p in discover_videos(root)],
                ["IMG_3197.MOV", "a.mkv", "b.mp4"],
            )

    def test_video_prefix_preserves_camera_filename_stem(self) -> None:
        self.assertEqual(video_prefix(Path("Videos/IMG_3161.MOV")), "IMG_3161")

    def test_sample_frame_indices_limits_uniform_samples(self) -> None:
        self.assertEqual(
            sample_frame_indices(total_frames=300, source_fps=30.0, sample_seconds=2.0, max_frames=4),
            [0, 60, 120, 180],
        )

    def test_sample_frame_indices_handles_unknown_fps(self) -> None:
        self.assertEqual(
            sample_frame_indices(total_frames=90, source_fps=0.0, sample_seconds=1.0, max_frames=10),
            [0, 30, 60],
        )


if __name__ == "__main__":
    unittest.main()
