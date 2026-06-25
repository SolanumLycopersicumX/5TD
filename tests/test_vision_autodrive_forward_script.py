import unittest

from tools.robot import vision_autodrive_forward as script


class VisionAutodriveForwardScriptTest(unittest.TestCase):
    def test_parser_defaults_are_low_speed_and_laptop_webcam(self):
        args = script.build_arg_parser().parse_args([])

        self.assertEqual(args.camera, "/dev/video0")
        self.assertEqual(args.port, "/dev/ttyUSB0")
        self.assertAlmostEqual(args.linear, 0.015)
        self.assertFalse(args.enable_driver)
        self.assertFalse(args.release_estop)
        self.assertEqual(args.display_backend, "tk")

    def test_node_address_accepts_hex(self):
        args = script.build_arg_parser().parse_args(["--addr", "0x06"])

        self.assertEqual(args.addr, 0x06)


if __name__ == "__main__":
    unittest.main()
