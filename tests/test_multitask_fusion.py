import tempfile
import unittest
from pathlib import Path

import numpy as np

import tools.passable_segmentation.evaluate_multitask_videos as eval_mod

from tools.passable_segmentation.evaluate_multitask_videos import (
    FUSED_LABELS,
    discover_videos,
    evaluate_video,
    evaluate_videos,
    fuse_multitask_predictions,
    mask_ratios,
    write_overlay_image,
    sample_frame_indices,
    should_sample_frame,
    video_output_slug,
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

    def test_evaluate_videos_raises_when_no_videos_are_discovered(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video_root = root / "Videos"
            video_root.mkdir()

            with self.assertRaisesRegex(RuntimeError, "No supported video files"):
                evaluate_videos(
                    video_root=video_root,
                    output_dir=root / "out",
                    sample_fps=1.0,
                    passable_checkpoint=root / "missing_passable.pt",
                    boundary_checkpoint=root / "missing_boundary.pt",
                    obstacle_checkpoint=root / "missing_obstacle.pt",
                )

    def test_evaluate_videos_allows_empty_output_when_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video_root = root / "Videos"
            output_dir = root / "out"
            video_root.mkdir()

            videos, frames = evaluate_videos(
                video_root=video_root,
                output_dir=output_dir,
                sample_fps=1.0,
                passable_checkpoint=root / "missing_passable.pt",
                boundary_checkpoint=root / "missing_boundary.pt",
                obstacle_checkpoint=root / "missing_obstacle.pt",
                allow_empty=True,
            )

            self.assertEqual((videos, frames), (0, 0))
            self.assertTrue((output_dir / "frame_metrics.csv").exists())
            self.assertTrue((output_dir / "video_summary.csv").exists())

    def test_evaluate_video_raises_when_opened_video_has_no_readable_frames(self):
        import torch

        class FakeCapture:
            def __init__(self, _path: str) -> None:
                pass

            def isOpened(self) -> bool:
                return True

            def get(self, prop: int) -> float:
                if prop == FakeCV2.CAP_PROP_FPS:
                    return 30.0
                if prop == FakeCV2.CAP_PROP_FRAME_COUNT:
                    return 0.0
                return 0.0

            def read(self):
                return False, None

            def release(self) -> None:
                pass

        class FakeCV2:
            CAP_PROP_FPS = 1
            CAP_PROP_FRAME_COUNT = 2
            VideoCapture = FakeCapture

        previous_cv2 = eval_mod.cv2
        eval_mod.cv2 = FakeCV2
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                with self.assertRaisesRegex(RuntimeError, "Could not read any frames"):
                    evaluate_video(
                        root / "broken.mp4",
                        video_root=root,
                        output_dir=root / "out",
                        sample_fps=1.0,
                        models=(object(), object(), object()),
                        device=torch.device("cpu"),
                        max_contact_per_video=1,
                    )
        finally:
            eval_mod.cv2 = previous_cv2

    def test_write_overlay_image_raises_when_cv2_write_fails(self):
        class FakeCV2:
            COLOR_RGB2BGR = 1
            IMWRITE_JPEG_QUALITY = 2

            @staticmethod
            def cvtColor(image, _code):
                return image

            @staticmethod
            def imwrite(_path, _image, _params):
                return False

        previous_cv2 = eval_mod.cv2
        eval_mod.cv2 = FakeCV2
        try:
            with tempfile.TemporaryDirectory() as tmp:
                with self.assertRaisesRegex(RuntimeError, "Could not write overlay image"):
                    write_overlay_image(Path(tmp) / "overlay.jpg", np.zeros((2, 2, 3), dtype=np.uint8))
        finally:
            eval_mod.cv2 = previous_cv2

    def test_sample_frame_indices_uses_basic_one_fps_step(self):
        self.assertEqual(sample_frame_indices(total_frames=95, source_fps=30.0, sample_fps=1.0), [0, 30, 60, 90])
        self.assertEqual(sample_frame_indices(total_frames=0, source_fps=30.0, sample_fps=1.0), [])

    def test_should_sample_frame_supports_unknown_length_streaming(self):
        sampled = [idx for idx in range(95) if should_sample_frame(idx, source_fps=30.0, sample_fps=1.0)]

        self.assertEqual(sampled, [0, 30, 60, 90])

    def test_video_output_slug_uses_relative_path_to_avoid_stem_collisions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "session_a" / "front.mp4"
            second = root / "session_b" / "front.MOV"

            first_slug = video_output_slug(first, root)
            second_slug = video_output_slug(second, root)

            self.assertTrue(first_slug.startswith("session_a_front_mp4_"), first_slug)
            self.assertTrue(second_slug.startswith("session_b_front_MOV_"), second_slug)
            self.assertNotEqual(first_slug, second_slug)

    def test_video_output_slug_distinguishes_same_stem_different_extensions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mp4 = root / "session" / "front.mp4"
            mov = root / "session" / "front.MOV"

            self.assertNotEqual(video_output_slug(mp4, root), video_output_slug(mov, root))


if __name__ == "__main__":
    unittest.main()
