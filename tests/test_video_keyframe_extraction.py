from pathlib import Path
import sys
import tempfile
import unittest

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.passable_segmentation.extract_video_keyframes import (
    discover_videos,
    extract_keyframes,
    extract_video,
    should_save_sequential_frame,
    sample_frame_indices,
    video_output_prefix,
    video_prefix,
    write_batch_files,
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

    def test_video_output_prefix_distinguishes_same_stem_batch_inputs(self) -> None:
        video_root = Path("Videos")

        mp4_prefix = video_output_prefix(video_root / "front.mp4", video_root=video_root)
        mov_prefix = video_output_prefix(video_root / "front.MOV", video_root=video_root)

        self.assertNotEqual(mp4_prefix, mov_prefix)
        self.assertNotEqual(f"{mp4_prefix}_f000000.jpg", f"{mov_prefix}_f000000.jpg")

    def test_extract_keyframes_uses_unique_output_prefix_for_same_stem_videos(self) -> None:
        class FakeFrame:
            shape = (4, 5, 3)

        class FakeCapture:
            def __init__(self, _path: str) -> None:
                self.read_count = 0

            def isOpened(self) -> bool:
                return True

            def get(self, prop: int) -> float:
                if prop == FakeCV2.CAP_PROP_FPS:
                    return 30.0
                if prop == FakeCV2.CAP_PROP_FRAME_COUNT:
                    return 1.0
                return 0.0

            def set(self, _prop: int, _value: int) -> None:
                pass

            def read(self):
                if self.read_count >= 1:
                    return False, None
                self.read_count += 1
                return True, FakeFrame()

            def release(self) -> None:
                pass

        class FakeCV2:
            CAP_PROP_FPS = 1
            CAP_PROP_FRAME_COUNT = 2
            CAP_PROP_POS_FRAMES = 3
            VideoCapture = FakeCapture

            @staticmethod
            def imwrite(path: str, _frame: FakeFrame) -> bool:
                Image.new("RGB", (5, 4), "white").save(path, "JPEG")
                return True

        previous_cv2 = sys.modules.get("cv2")
        sys.modules["cv2"] = FakeCV2
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                video_root = root / "Videos"
                output_root = root / "out"
                video_root.mkdir()
                (video_root / "front.mp4").write_bytes(b"video")
                (video_root / "front.MOV").write_bytes(b"video")

                metadata = extract_keyframes(
                    video_root=video_root,
                    output_root=output_root,
                    labels_path=root / "missing_labels.txt",
                    sample_seconds=1.0,
                    max_frames_per_video=1,
                )
        finally:
            if previous_cv2 is None:
                sys.modules.pop("cv2", None)
            else:
                sys.modules["cv2"] = previous_cv2

        frame_names = [Path(record["file"]).name for record in metadata["frames"]]
        prefixes = [video["prefix"] for video in metadata["videos"]]
        self.assertEqual(len(frame_names), 2)
        self.assertEqual(len(set(frame_names)), 2)
        self.assertEqual(len(set(prefixes)), 2)

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

    def test_should_save_sequential_frame_matches_sample_interval(self) -> None:
        self.assertTrue(
            should_save_sequential_frame(frame_idx=0, source_fps=0.0, sample_seconds=1.0)
        )
        self.assertFalse(
            should_save_sequential_frame(frame_idx=29, source_fps=0.0, sample_seconds=1.0)
        )
        self.assertTrue(
            should_save_sequential_frame(frame_idx=30, source_fps=0.0, sample_seconds=1.0)
        )

    def test_extract_video_reads_sequentially_when_frame_count_unknown(self) -> None:
        class FakeFrame:
            shape = (4, 5, 3)

        class FakeCapture:
            def __init__(self, _path: str) -> None:
                self.next_frame = 0

            def isOpened(self) -> bool:
                return True

            def get(self, _prop: int) -> float:
                return 0.0

            def read(self):
                if self.next_frame >= 35:
                    return False, None
                self.next_frame += 1
                return True, FakeFrame()

            def release(self) -> None:
                pass

        class FakeCV2:
            CAP_PROP_FPS = 1
            CAP_PROP_FRAME_COUNT = 2
            VideoCapture = FakeCapture

            @staticmethod
            def imwrite(path: str, _frame: FakeFrame) -> bool:
                Path(path).write_bytes(b"jpg")
                return True

        previous_cv2 = sys.modules.get("cv2")
        sys.modules["cv2"] = FakeCV2
        try:
            with tempfile.TemporaryDirectory() as tmp:
                image_dir = Path(tmp) / "images"

                records = extract_video(
                    Path("Videos/clip.h264"),
                    image_dir,
                    sample_seconds=1.0,
                    max_frames=2,
                )
        finally:
            if previous_cv2 is None:
                sys.modules.pop("cv2", None)
            else:
                sys.modules["cv2"] = previous_cv2

        self.assertEqual([record["frame_idx"] for record in records], [0, 30])
        self.assertEqual(
            [Path(record["file"]).name for record in records],
            ["clip_f000000.jpg", "clip_f000030.jpg"],
        )

    def test_write_batch_files_uses_nodata_and_shape_specific_guidance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp)

            write_batch_files(output_root)

            readme = (output_root / "README.md").read_text(encoding="utf-8")
            launch_script = (output_root / "launch_labelme.sh").read_text(encoding="utf-8")
            rules = (output_root / "annotation_rules.md").read_text(encoding="utf-8")
            self.assertIn("--nodata", readme)
            self.assertIn("--nodata", launch_script)
            self.assertIn("Use polygons for:", rules)
            self.assertIn("Use rectangles by default for:", rules)
            self.assertIn("worker", rules)
            self.assertIn("construction_vehicle", rules)
            self.assertIn("suspended_object", rules)
            self.assertIn("debris", rules)
            self.assertIn("polygon is allowed for irregular debris", rules)
            self.assertIn("surface_artifact_passable` is not a hazard label", rules)


if __name__ == "__main__":
    unittest.main()
