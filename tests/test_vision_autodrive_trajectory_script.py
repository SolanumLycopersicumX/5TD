import unittest

from tools.robot import vision_autodrive_trajectory as script


class VisionAutodriveTrajectoryScriptTest(unittest.TestCase):
    def test_parser_defaults_are_conservative(self):
        args = script.build_arg_parser().parse_args([])

        self.assertEqual(args.camera, "/dev/video0")
        self.assertEqual(args.port, "/dev/ttyUSB0")
        self.assertAlmostEqual(args.linear, 0.012)
        self.assertAlmostEqual(args.max_angular, 0.08)
        self.assertFalse(args.enable_driver)
        self.assertFalse(args.release_estop)
        self.assertEqual(args.display_backend, "tk")

    def test_node_address_accepts_hex(self):
        args = script.build_arg_parser().parse_args(["--addr", "0x06"])

        self.assertEqual(args.addr, 0x06)

    def test_cpu_flag_defaults_to_false(self):
        args = script.build_arg_parser().parse_args([])

        self.assertFalse(args.cpu)


if __name__ == "__main__":
    unittest.main()
