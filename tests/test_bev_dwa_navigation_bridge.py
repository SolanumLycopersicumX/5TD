import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

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


class FusedMaskExportTest(unittest.TestCase):
    def test_write_fused_masks_writes_expected_label_directories(self):
        from tools.passable_segmentation.visualize_fused_passable_boundary import write_fused_masks

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "masks"
            fused = {
                "safe_passable": np.array([[False, True], [True, True]], dtype=bool),
                "ditch": np.array([[False, False], [False, True]], dtype=bool),
                "left_barrier": np.array([[True, False], [False, False]], dtype=bool),
                "tunnel_wall": np.array([[False, True], [False, False]], dtype=bool),
            }

            written = write_fused_masks(output_dir, "frame_001", fused)

            self.assertEqual(len(written), 4)
            for label in ["safe_passable", "ditch", "left_barrier", "tunnel_wall"]:
                path = output_dir / label / "frame_001.png"
                self.assertIn(path, written)
                self.assertTrue(path.exists())
                image = Image.open(path)
                self.assertEqual(image.mode, "L")
                self.assertEqual(np.asarray(image).dtype, np.uint8)

    def test_write_fused_masks_uses_255_for_true_and_0_for_false(self):
        from tools.passable_segmentation.visualize_fused_passable_boundary import write_fused_masks

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "masks"
            fused = {
                "safe_passable": np.array([[False, True]], dtype=bool),
                "ditch": np.array([[True, False]], dtype=bool),
                "left_barrier": np.array([[False, False]], dtype=bool),
                "tunnel_wall": np.array([[True, True]], dtype=bool),
            }

            write_fused_masks(output_dir, "frame_002", fused)

            safe = np.asarray(Image.open(output_dir / "safe_passable" / "frame_002.png"))
            self.assertEqual(safe.tolist(), [[0, 255]])


class BEVGridTest(unittest.TestCase):
    def _bundle(self, safe, ditch=None, wall=None, left=None):
        shape = safe.shape
        return MaskBundle(
            safe_passable=safe.astype(bool),
            ditch=np.zeros(shape, dtype=bool) if ditch is None else ditch.astype(bool),
            tunnel_wall=np.zeros(shape, dtype=bool) if wall is None else wall.astype(bool),
            left_barrier=np.zeros(shape, dtype=bool) if left is None else left.astype(bool),
            source_frame="synthetic",
        )

    def test_center_passable_corridor_becomes_low_risk_free_space(self):
        from src.tunnel_nav.bev import build_pseudo_bev_grid

        safe = np.zeros((80, 120), dtype=bool)
        safe[:, 35:85] = True

        grid = build_pseudo_bev_grid(self._bundle(safe), resolution_m=0.1)
        center_col = grid.occupancy.shape[1] // 2

        self.assertFalse(grid.occupancy[-1, center_col])
        self.assertLess(grid.risk[-1, center_col], 0.3)
        self.assertTrue(grid.occupancy[-1, 0])
        self.assertEqual(grid.risk[-1, 0], 1.0)

    def test_ditch_and_wall_are_occupied_high_risk(self):
        from src.tunnel_nav.bev import build_pseudo_bev_grid

        safe = np.ones((80, 120), dtype=bool)
        ditch = np.zeros_like(safe)
        wall = np.zeros_like(safe)
        ditch[:, 90:110] = True
        wall[:, 0:12] = True

        grid = build_pseudo_bev_grid(self._bundle(safe, ditch=ditch, wall=wall), resolution_m=0.1)

        self.assertTrue(grid.occupancy[-1, -4])
        self.assertEqual(grid.risk[-1, -4], 1.0)
        self.assertTrue(grid.occupancy[-1, 2])
        self.assertEqual(grid.risk[-1, 2], 1.0)

    def test_boundary_dilation_creates_high_risk_keepout_band(self):
        from src.tunnel_nav.bev import build_pseudo_bev_grid

        safe = np.ones((80, 120), dtype=bool)
        ditch = np.zeros_like(safe)
        ditch[:, 75:80] = True

        grid = build_pseudo_bev_grid(
            self._bundle(safe, ditch=ditch),
            resolution_m=0.1,
            boundary_dilation_m=0.3,
        )
        hard_cols = np.flatnonzero(grid.risk[-1] == 1.0)
        high_risk_cols = np.flatnonzero((grid.risk[-1] >= 0.65) & (grid.risk[-1] < 1.0))

        self.assertGreater(len(hard_cols), 0)
        self.assertGreater(len(high_risk_cols), 0)


class DWAPlannerTest(unittest.TestCase):
    def _grid(self, occupancy=None, risk=None):
        if occupancy is None:
            occupancy = np.zeros((80, 50), dtype=bool)
        if risk is None:
            risk = np.where(occupancy, 1.0, 0.1).astype(np.float32)
        return BEVGrid(
            occupancy=occupancy.astype(bool),
            risk=risk.astype(np.float32),
            x_min_m=-2.5,
            x_max_m=2.5,
            y_min_m=0.0,
            y_max_m=8.0,
            resolution_m=0.1,
        )

    def test_clear_grid_selects_forward_trajectory(self):
        from src.tunnel_nav.dwa import select_dwa_trajectory

        selected, candidates = select_dwa_trajectory(self._grid(), DWAConfig())

        self.assertIsNotNone(selected)
        self.assertTrue(selected.feasible)
        self.assertGreater(selected.linear_mps, 0.0)
        self.assertAlmostEqual(selected.angular_radps, 0.0, delta=0.13)
        self.assertGreater(len(candidates), 0)

    def test_all_occupied_grid_has_no_feasible_trajectory(self):
        from src.tunnel_nav.dwa import select_dwa_trajectory

        occupancy = np.ones((80, 50), dtype=bool)
        selected, candidates = select_dwa_trajectory(self._grid(occupancy=occupancy), DWAConfig())

        self.assertIsNone(selected)
        self.assertGreater(len(candidates), 0)
        self.assertTrue(all(not candidate.feasible for candidate in candidates))

    def test_selected_trajectory_never_enters_occupied_cells(self):
        from src.tunnel_nav.dwa import metric_to_grid_cell, select_dwa_trajectory

        occupancy = np.zeros((80, 50), dtype=bool)
        occupancy[70:79, 24:26] = True
        grid = self._grid(occupancy=occupancy)

        selected, _ = select_dwa_trajectory(grid, DWAConfig())

        if selected is None:
            return
        for x_m, y_m, _yaw in selected.points:
            row, col = metric_to_grid_cell(grid, float(x_m), float(y_m))
            self.assertFalse(grid.occupancy[row, col])
