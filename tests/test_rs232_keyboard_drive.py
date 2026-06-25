import unittest

from tools.robot import rs232_keyboard_drive as keyboard


class Rs232KeyboardDriveTest(unittest.TestCase):
    def test_keyboard_buffer_prefers_stop_and_quit_over_motion(self):
        self.assertEqual(keyboard.select_control_key("wwwwkww"), "k")
        self.assertEqual(keyboard.select_control_key("wwww ww"), " ")
        self.assertEqual(keyboard.select_control_key("wwwwqww"), "q")
        self.assertEqual(keyboard.select_control_key("wwd"), "d")
        self.assertIsNone(keyboard.select_control_key(""))

    def test_command_for_key_is_deadman_stop(self):
        self.assertEqual(keyboard.command_for_key(None, 0.03, 0.10), (0.0, 0.0))
        self.assertEqual(keyboard.command_for_key("x", 0.03, 0.10), (0.0, 0.0))
        self.assertEqual(keyboard.command_for_key("w", 0.03, 0.10), (0.03, 0.0))
        self.assertEqual(keyboard.command_for_key("s", 0.03, 0.10), (-0.03, 0.0))
        self.assertEqual(keyboard.command_for_key("a", 0.03, 0.10), (0.0, 0.10))
        self.assertEqual(keyboard.command_for_key("d", 0.03, 0.10), (0.0, -0.10))
        self.assertEqual(keyboard.command_for_key("k", 0.03, 0.10), (0.0, 0.0))

    def test_parser_defaults_are_low_speed_and_live_serial_explicit(self):
        args = keyboard.build_arg_parser().parse_args([])

        self.assertEqual(args.port, "/dev/ttyUSB0")
        self.assertEqual(args.addr, 0x06)
        self.assertEqual(args.linear, 0.03)
        self.assertEqual(args.angular, 0.10)
        self.assertEqual(args.rate_hz, 10.0)
        self.assertFalse(args.enable)
        self.assertFalse(args.release_estop)

    def test_parse_node_addr_accepts_hex_or_decimal(self):
        self.assertEqual(keyboard.parse_node_addr("0x06"), 0x06)
        self.assertEqual(keyboard.parse_node_addr("6"), 0x06)


if __name__ == "__main__":
    unittest.main()
