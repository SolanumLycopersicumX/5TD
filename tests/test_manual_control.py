import unittest

from src.tunnel_nav.manual_control import (
    ManualControlLimits,
    shape_axis,
    manual_command_from_axes,
    slew_towards,
)


class ManualControlTest(unittest.TestCase):
    def test_deadzone_zeroes_center_jitter(self):
        limits = ManualControlLimits(deadzone=0.18)

        self.assertEqual(shape_axis(0.0, limits), 0.0)
        self.assertEqual(shape_axis(0.10, limits), 0.0)
        self.assertEqual(shape_axis(-0.17, limits), 0.0)

    def test_curve_makes_small_joystick_input_gentle(self):
        limits = ManualControlLimits(deadzone=0.18, response_exponent=2.0)

        shaped = shape_axis(0.30, limits)

        self.assertGreater(shaped, 0.0)
        self.assertLess(shaped, 0.03)

    def test_manual_command_applies_limits_and_deadzone(self):
        limits = ManualControlLimits(
            max_linear_mps=0.04,
            max_angular_radps=0.12,
            deadzone=0.18,
            response_exponent=2.0,
        )

        self.assertEqual(manual_command_from_axes(0.10, 0.10, limits), (0.0, 0.0))
        linear, angular = manual_command_from_axes(1.0, -1.0, limits)

        self.assertEqual(linear, 0.04)
        self.assertEqual(angular, -0.12)

    def test_slew_towards_never_overshoots_zero(self):
        self.assertEqual(slew_towards(0.02, 0.0, 0.05), 0.0)
        self.assertEqual(slew_towards(-0.02, 0.0, 0.05), 0.0)
        self.assertEqual(slew_towards(0.02, -0.02, 0.015), 0.005)
        self.assertEqual(slew_towards(-0.02, 0.02, 0.015), -0.005)

    def test_slew_towards_limits_acceleration(self):
        self.assertAlmostEqual(slew_towards(0.00, 0.04, 0.01), 0.01)
        self.assertAlmostEqual(slew_towards(0.04, 0.00, 0.01), 0.03)

    def test_rotation_start_boost_applies_only_to_in_place_turns(self):
        limits = ManualControlLimits(
            max_linear_mps=0.04,
            max_angular_radps=0.12,
            deadzone=0.18,
            response_exponent=2.0,
            min_turn_start_radps=0.06,
        )

        linear, angular = manual_command_from_axes(0.0, 0.30, limits)
        self.assertEqual(linear, 0.0)
        self.assertEqual(angular, 0.06)

        linear, angular = manual_command_from_axes(0.0, -0.30, limits)
        self.assertEqual(linear, 0.0)
        self.assertEqual(angular, -0.06)

    def test_rotation_start_boost_does_not_create_turn_at_center_or_while_driving(self):
        limits = ManualControlLimits(
            max_linear_mps=0.04,
            max_angular_radps=0.12,
            deadzone=0.18,
            response_exponent=2.0,
            min_turn_start_radps=0.06,
        )

        self.assertEqual(manual_command_from_axes(0.0, 0.10, limits), (0.0, 0.0))
        linear, angular = manual_command_from_axes(1.0, 0.30, limits)
        self.assertGreater(linear, 0.0)
        self.assertLess(abs(angular), 0.06)


if __name__ == "__main__":
    unittest.main()
