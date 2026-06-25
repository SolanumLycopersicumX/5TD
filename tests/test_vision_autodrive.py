import unittest

import numpy as np

from src.tunnel_nav.vision_autodrive import DriveGateConfig, evaluate_drive_gate


class VisionAutodriveTest(unittest.TestCase):
    def _blank_fused(self, height=100, width=100):
        return {
            "safe_passable": np.zeros((height, width), dtype=bool),
            "ditch": np.zeros((height, width), dtype=bool),
            "left_barrier": np.zeros((height, width), dtype=bool),
            "tunnel_wall": np.zeros((height, width), dtype=bool),
        }

    def test_allows_forward_when_near_center_roi_is_safe(self):
        fused = self._blank_fused()
        fused["safe_passable"][60:95, 35:65] = True

        decision = evaluate_drive_gate(fused, DriveGateConfig())

        self.assertTrue(decision.allow_forward)
        self.assertEqual(decision.reason, "clear")

    def test_stops_when_center_roi_has_too_little_passable_area(self):
        fused = self._blank_fused()
        fused["safe_passable"][80:95, 45:55] = True

        decision = evaluate_drive_gate(fused, DriveGateConfig(min_safe_ratio=0.65))

        self.assertFalse(decision.allow_forward)
        self.assertEqual(decision.reason, "low_passable")

    def test_stops_when_ditch_or_wall_enters_center_roi(self):
        config = DriveGateConfig(max_hazard_ratio=0.02)
        for label in ("ditch", "tunnel_wall"):
            with self.subTest(label=label):
                fused = self._blank_fused()
                fused["safe_passable"][60:95, 35:65] = True
                fused[label][70:80, 45:55] = True

                decision = evaluate_drive_gate(fused, config)

                self.assertFalse(decision.allow_forward)
                self.assertEqual(decision.reason, "hazard")

    def test_roi_pixel_bounds_are_reported_for_visualization(self):
        fused = self._blank_fused(height=120, width=200)
        fused["safe_passable"][:] = True

        decision = evaluate_drive_gate(fused, DriveGateConfig(roi_x_min=0.25, roi_x_max=0.75, roi_y_min=0.5, roi_y_max=1.0))

        self.assertEqual(decision.roi_bounds, (50, 60, 150, 120))


if __name__ == "__main__":
    unittest.main()
