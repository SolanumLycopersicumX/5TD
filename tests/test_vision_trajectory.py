import unittest

import numpy as np

from src.tunnel_nav.vision_autodrive import DriveGateConfig
from src.tunnel_nav.vision_trajectory import (
    TrajectoryConfig,
    compute_trajectory_command,
    extract_centerline,
)


def make_fused(height=100, width=100):
    return {
        "safe_passable": np.zeros((height, width), dtype=bool),
        "ditch": np.zeros((height, width), dtype=bool),
        "left_barrier": np.zeros((height, width), dtype=bool),
        "tunnel_wall": np.zeros((height, width), dtype=bool),
    }


def paint_corridor(mask, centers, half_width=14):
    height, width = mask.shape
    anchor_ys = np.linspace(height - 6, int(height * 0.45), len(centers))
    y_start = int(min(anchor_ys))
    y_end = int(max(anchor_ys))
    for y in range(y_start, y_end + 1):
        center = np.interp(y, anchor_ys[::-1], list(centers)[::-1])
        x0 = max(0, int(round(center)) - half_width)
        x1 = min(width, int(round(center)) + half_width + 1)
        mask[y, x0:x1] = True


class VisionTrajectoryTest(unittest.TestCase):
    def test_extract_centerline_returns_bottom_to_top_points(self):
        fused = make_fused()
        paint_corridor(fused["safe_passable"], [50, 50, 50, 50, 50, 50])

        points = extract_centerline(fused["safe_passable"], TrajectoryConfig(scan_count=6))

        self.assertGreaterEqual(len(points), 4)
        self.assertGreater(points[0].y, points[-1].y)
        self.assertTrue(all(abs(point.x - 50) <= 1 for point in points))

    def test_straight_center_path_outputs_forward_with_near_zero_turn(self):
        fused = make_fused()
        paint_corridor(fused["safe_passable"], [50, 50, 50, 50, 50, 50])

        command = compute_trajectory_command(fused, DriveGateConfig(), TrajectoryConfig(), base_linear_mps=0.012)

        self.assertTrue(command.allow_motion)
        self.assertEqual(command.reason, "clear")
        self.assertAlmostEqual(command.linear_mps, 0.012)
        self.assertAlmostEqual(command.angular_radps, 0.0, delta=0.01)

    def test_left_target_outputs_positive_angular_velocity(self):
        fused = make_fused()
        paint_corridor(fused["safe_passable"], [50, 47, 43, 39, 35, 31])

        command = compute_trajectory_command(fused, DriveGateConfig(), TrajectoryConfig(), base_linear_mps=0.012)

        self.assertTrue(command.allow_motion)
        self.assertGreater(command.angular_radps, 0.0)

    def test_right_target_outputs_negative_angular_velocity(self):
        fused = make_fused()
        paint_corridor(fused["safe_passable"], [50, 53, 57, 61, 65, 69])

        command = compute_trajectory_command(fused, DriveGateConfig(), TrajectoryConfig(), base_linear_mps=0.012)

        self.assertTrue(command.allow_motion)
        self.assertLess(command.angular_radps, 0.0)

    def test_stops_when_trajectory_has_too_few_points(self):
        fused = make_fused()
        fused["safe_passable"][75:85, 45:55] = True

        command = compute_trajectory_command(fused, DriveGateConfig(min_safe_ratio=0.01), TrajectoryConfig(min_points=3), base_linear_mps=0.012)

        self.assertFalse(command.allow_motion)
        self.assertEqual(command.reason, "path_lost")
        self.assertEqual(command.linear_mps, 0.0)
        self.assertEqual(command.angular_radps, 0.0)

    def test_stops_when_drive_gate_sees_hazard(self):
        fused = make_fused()
        paint_corridor(fused["safe_passable"], [50, 50, 50, 50, 50, 50])
        fused["ditch"][70:80, 45:55] = True

        command = compute_trajectory_command(fused, DriveGateConfig(max_hazard_ratio=0.01), TrajectoryConfig(), base_linear_mps=0.012)

        self.assertFalse(command.allow_motion)
        self.assertEqual(command.reason, "hazard")


if __name__ == "__main__":
    unittest.main()
