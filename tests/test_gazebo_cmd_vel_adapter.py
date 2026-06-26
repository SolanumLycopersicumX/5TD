import unittest

from src.tunnel_nav.gazebo_control import GazeboCmdVelAdapter
from tools.sim.gazebo_driver_keyboard import _KeyPressState, _command_for_key, _command_for_pressed_keys, _select_control_key, build_arg_parser


class FakeRunner:
    def __init__(self):
        self.calls = []

    def __call__(self, args, check, timeout=None):
        self.calls.append((args, check, timeout))


class GazeboCmdVelAdapterTest(unittest.TestCase):
    def test_set_velocity_publishes_gazebo_twist(self):
        runner = FakeRunner()
        adapter = GazeboCmdVelAdapter(runner=runner, gz_command="gz")

        adapter.set_velocity(0.05, 0.10)

        self.assertEqual(len(runner.calls), 1)
        args, check, _timeout = runner.calls[0]
        self.assertTrue(check)
        self.assertEqual(args[:6], ["gz", "topic", "-t", "/cmd_vel", "-m", "gz.msgs.Twist"])
        payload = args[-1]
        self.assertIn("linear: {x: 0.050000}", payload)
        self.assertIn("angular: {z: 0.100000}", payload)

    def test_set_velocity_uses_timeout_so_missing_gazebo_does_not_hang(self):
        runner = FakeRunner()
        adapter = GazeboCmdVelAdapter(runner=runner, gz_command="gz", publish_timeout_s=1.25)

        adapter.set_velocity(0.05, 0.10)

        _args, _check, timeout = runner.calls[0]
        self.assertEqual(timeout, 1.25)

    def test_driver_style_helpers_map_to_linear_and_angular_velocity(self):
        runner = FakeRunner()
        adapter = GazeboCmdVelAdapter(runner=runner, gz_command="gz")

        adapter.forward(0.06)
        adapter.backward(0.04)
        adapter.turn_left(0.30)
        adapter.turn_right(0.20)
        adapter.stop()

        payloads = [call[0][-1] for call in runner.calls]
        self.assertIn("linear: {x: 0.060000}, angular: {z: 0.000000}", payloads[0])
        self.assertIn("linear: {x: -0.040000}, angular: {z: 0.000000}", payloads[1])
        self.assertIn("linear: {x: 0.000000}, angular: {z: 0.300000}", payloads[2])
        self.assertIn("linear: {x: 0.000000}, angular: {z: -0.200000}", payloads[3])
        self.assertIn("linear: {x: 0.000000}, angular: {z: 0.000000}", payloads[4])

    def test_keyboard_idle_is_deadman_stop(self):
        self.assertEqual(_command_for_key(None, 0.10, 0.30), (0.0, 0.0))
        self.assertEqual(_command_for_key("x", 0.10, 0.30), (0.0, 0.0))
        self.assertEqual(_command_for_key("w", 0.10, 0.30), (0.10, 0.0))
        self.assertEqual(_command_for_key("k", 0.10, 0.30), (0.0, 0.0))

    def test_pressed_key_state_stops_immediately_after_release(self):
        self.assertEqual(_command_for_pressed_keys(set(), 0.10, 0.30), (0.0, 0.0))
        self.assertEqual(_command_for_pressed_keys({"w"}, 0.10, 0.30), (0.10, 0.0))
        self.assertEqual(_command_for_pressed_keys(set(), 0.10, 0.30), (0.0, 0.0))

    def test_pressed_key_state_combines_forward_and_turn(self):
        self.assertEqual(_command_for_pressed_keys({"w", "a"}, 0.10, 0.30), (0.10, 0.30))
        self.assertEqual(_command_for_pressed_keys({"w", "d"}, 0.10, 0.30), (0.10, -0.30))
        self.assertEqual(_command_for_pressed_keys({"a"}, 0.10, 0.30), (0.0, 0.30))


    def test_key_press_state_ignores_repeated_press_until_release(self):
        state = _KeyPressState(linear=0.10, angular=0.30)

        self.assertEqual(state.press("w"), (0.10, 0.0))
        self.assertIsNone(state.press("w"))
        self.assertIsNone(state.press("w"))
        self.assertEqual(state.release("w"), (0.0, 0.0))
        self.assertIsNone(state.release("w"))

    def test_keyboard_defaults_to_tk_backend_for_key_release_events(self):
        args = build_arg_parser().parse_args([])

        self.assertEqual(args.backend, "tk")
        self.assertEqual(args.rate_hz, 20.0)
        self.assertEqual(args.release_delay_ms, 0)

    def test_keyboard_buffer_prefers_stop_and_quit_over_buffered_motion(self):
        self.assertEqual(_select_control_key("wwwwkww"), "k")
        self.assertEqual(_select_control_key("wwww ww"), " ")
        self.assertEqual(_select_control_key("wwwwqww"), "q")
        self.assertEqual(_select_control_key("wwd"), "d")
        self.assertIsNone(_select_control_key(""))


if __name__ == "__main__":
    unittest.main()
