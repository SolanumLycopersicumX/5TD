import json
import unittest

import numpy as np

from src.tunnel_nav.motion import (
    BEVGrid,
    DWAConfig,
    MaskBundle,
    MotionCommand,
    NavigationConfig,
    RiskGrid,
    Trajectory,
)


class CoreModelTest(unittest.TestCase):
    def test_motion_command_serializes_physical_units(self):
        command = MotionCommand(
            linear_mps=0.08,
            angular_radps=-0.12,
            brake=False,
            safety_state="S0_NORMAL",
            reason="risk grid clear",
            confidence=0.75,
            source_frame="frame_001",
            dry_run=True,
        )

        payload = command.to_dict()

        self.assertEqual(payload["linear_mps"], 0.08)
        self.assertEqual(payload["angular_radps"], -0.12)
        self.assertFalse(payload["brake"])
        self.assertEqual(payload["safety_state"], "S0_NORMAL")
        self.assertEqual(payload["reason"], "risk grid clear")
        self.assertEqual(payload["confidence"], 0.75)
        self.assertEqual(payload["source_frame"], "frame_001")
        self.assertTrue(payload["dry_run"])
        json.dumps(payload)

    def test_navigation_defaults_are_conservative(self):
        config = NavigationConfig()

        self.assertEqual(config.max_speed_mps, 0.10)
        self.assertEqual(config.max_angular_radps, 0.50)
        self.assertEqual(config.angular_sign, 1)
        self.assertEqual(config.vehicle_width_m, 0.80)
        self.assertEqual(config.safety_margin_m, 0.30)
        self.assertTrue(config.dry_run)
        self.assertTrue(config.live_requires_explicit_flag)

    def test_dwa_defaults_are_low_speed(self):
        config = DWAConfig()

        self.assertEqual(config.min_velocity_mps, 0.0)
        self.assertEqual(config.max_velocity_mps, 0.10)
        self.assertEqual(config.velocity_samples, 3)
        self.assertEqual(config.max_angular_radps, 0.50)
        self.assertEqual(config.angular_samples, 9)
        self.assertEqual(config.predict_time_s, 2.0)
        self.assertEqual(config.dt_s, 0.2)

    def test_bev_grid_keeps_metric_metadata(self):
        grid = BEVGrid(
            occupancy=np.zeros((80, 50), dtype=bool),
            risk=np.zeros((80, 50), dtype=np.float32),
            x_min_m=-2.5,
            x_max_m=2.5,
            y_min_m=0.0,
            y_max_m=8.0,
            resolution_m=0.1,
        )

        self.assertEqual(grid.shape, (80, 50))
        self.assertAlmostEqual(grid.width_m, 5.0)
        self.assertAlmostEqual(grid.length_m, 8.0)

    def test_mask_bundle_and_trajectory_hold_arrays(self):
        safe = np.ones((8, 10), dtype=bool)
        empty = np.zeros((8, 10), dtype=bool)
        bundle = MaskBundle(
            safe_passable=safe,
            ditch=empty,
            tunnel_wall=empty,
            left_barrier=empty,
            source_frame="frame_001",
        )
        risk = RiskGrid(
            risk=np.zeros((80, 50), dtype=np.float32),
            x_min_m=-2.5,
            x_max_m=2.5,
            y_min_m=0.0,
            y_max_m=8.0,
            resolution_m=0.1,
        )
        trajectory = Trajectory(
            points=np.array([[0.0, 0.0, 0.0], [0.0, 0.1, 0.0]], dtype=np.float32),
            linear_mps=0.05,
            angular_radps=0.0,
            score=1.0,
            max_risk=0.1,
            min_clearance_m=1.0,
            feasible=True,
            reason="clear",
        )

        self.assertEqual(bundle.shape, (8, 10))
        self.assertEqual(risk.shape, (80, 50))
        self.assertEqual(trajectory.points.shape, (2, 3))
