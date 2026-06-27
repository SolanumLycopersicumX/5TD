import unittest

from tools.passable_segmentation import live_webcam_fused_preview as preview


class LiveWebcamFusedPreviewTest(unittest.TestCase):
    def test_parse_camera_keeps_device_path_and_converts_index(self):
        self.assertEqual(preview.parse_camera("/dev/video0"), "/dev/video0")
        self.assertEqual(preview.parse_camera("0"), 0)
        self.assertEqual(preview.parse_camera("2"), 2)

    def test_parser_defaults_to_laptop_webcam_and_existing_checkpoints(self):
        args = preview.build_arg_parser().parse_args([])

        self.assertEqual(args.camera, "/dev/video0")
        self.assertEqual(args.width, 640)
        self.assertEqual(args.height, 480)
        self.assertEqual(args.fps, 15)
        self.assertEqual(args.display_width, 1280)
        self.assertEqual(str(args.passable_checkpoint), "runs/passable_ego/passable_ditch_artifact_v3_finetune/best_model.pt")
        self.assertEqual(str(args.boundary_checkpoint), "runs/passable_ego/boundary_wall_aux_v2_no_testvideo/best_model.pt")

    def test_parser_defaults_to_tk_display_backend(self):
        args = preview.build_arg_parser().parse_args([])

        self.assertEqual(getattr(args, "display_backend", None), "tk")

    def test_scale_to_display_width_preserves_aspect_ratio(self):
        self.assertEqual(preview.scale_to_display_width(1280, 384, 1280), (1280, 384))
        self.assertEqual(preview.scale_to_display_width(2560, 768, 1280), (1280, 384))
        self.assertEqual(preview.scale_to_display_width(640, 480, 1280), (640, 480))


if __name__ == "__main__":
    unittest.main()
