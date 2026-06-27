# BEV DWA RS232 Navigation Bridge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first offline BEV / Risk Grid / DWA navigation bridge from fused segmentation masks to conservative motion commands, with RS232 kept in dry-run mode.

**Architecture:** New runtime code lives under `src/tunnel_nav` and does not import the legacy baseline. The first version uses a simplified pseudo-BEV projection, generates occupancy and risk grids, runs a conservative DWA-style local planner, passes the result through a safety filter, writes command JSON and overlays, and converts the command to RS232 register values only in dry-run form.

**Tech Stack:** Python, `numpy`, `Pillow`, optional `cv2`, `unittest`, existing `conda run -n lerobot python` workflow.

---

## Supersedes

This is now the current execution plan.

It supersedes the earlier image-space-first plan:

- `docs/superpowers/plans/2026-06-24-rs232-navigation-bridge.md`

The older plan remains useful for RS232 dry-run details, but the planner core should now follow this BEV / Risk Grid / DWA plan.

## Safety Boundaries

- No live serial writes in this implementation.
- No code opens a serial port by default.
- The BEV is a simplified pseudo-BEV until camera calibration is available.
- Pseudo-BEV output is acceptable for offline trajectory debugging, not for final meter-accurate real driving.
- Initial speed limits stay conservative: `0.05-0.10 m/s` for dry-run command generation.
- `angular_sign` remains configurable because RS232 and legacy comments disagree on angular direction.

## Files to Create or Modify

- Create `src/tunnel_nav/__init__.py`: package exports.
- Create `src/tunnel_nav/motion.py`: command, config, mask, BEV, risk, trajectory dataclasses.
- Create `src/tunnel_nav/bev.py`: simplified pseudo-BEV projection and occupancy/risk grid generation.
- Create `src/tunnel_nav/dwa.py`: conservative DWA trajectory sampling over the risk grid.
- Create `src/tunnel_nav/safety.py`: safety state filter and final command selection.
- Create `src/tunnel_nav/rs232.py`: Modbus CRC, velocity-to-register conversion, dry-run adapter.
- Create `tools/navigation_bridge/run_offline_bev_dwa_bridge.py`: offline CLI.
- Modify `tools/passable_segmentation/visualize_fused_passable_boundary.py`: optional fused mask export.
- Modify `configs/robot/vehicle.yaml`: differential-drive and RS232 dry-run defaults.
- Create `tests/test_bev_dwa_navigation_bridge.py`: unit tests for BEV, DWA, safety, CLI, and RS232.
- Update `docs/progress/LOG.md`: record results after implementation.

## Task 1: Core Data Model

**Files:**
- Create: `src/tunnel_nav/__init__.py`
- Create: `src/tunnel_nav/motion.py`
- Test: `tests/test_bev_dwa_navigation_bridge.py`

- [ ] **Step 1: Write failing tests for command and grid structures**

Create `tests/test_bev_dwa_navigation_bridge.py` with tests equivalent to:

```python
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
        json.dumps(payload)

    def test_navigation_defaults_are_conservative(self):
        config = NavigationConfig()
        self.assertEqual(config.max_speed_mps, 0.10)
        self.assertEqual(config.max_angular_radps, 0.50)
        self.assertTrue(config.dry_run)
        self.assertTrue(config.live_requires_explicit_flag)

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
```

- [ ] **Step 2: Verify the tests fail**

Run:

```bash
conda run -n lerobot python -m unittest tests.test_bev_dwa_navigation_bridge.CoreModelTest -v
```

Expected: fail because `src.tunnel_nav.motion` does not exist.

- [ ] **Step 3: Implement the data classes**

Implement in `src/tunnel_nav/motion.py`:

- `NavigationConfig`
- `DWAConfig`
- `MaskBundle`
- `BEVGrid`
- `RiskGrid`
- `Trajectory`
- `MotionCommand`

Required design:

```python
NavigationConfig(
    max_speed_mps=0.10,
    max_angular_radps=0.50,
    angular_sign=1,
    vehicle_width_m=0.80,
    safety_margin_m=0.30,
    dry_run=True,
    live_requires_explicit_flag=True,
)
```

```python
DWAConfig(
    min_velocity_mps=0.0,
    max_velocity_mps=0.10,
    velocity_samples=3,
    max_angular_radps=0.50,
    angular_samples=9,
    predict_time_s=2.0,
    dt_s=0.2,
)
```

`MotionCommand.to_dict()` must output physical units:

- `linear_mps`
- `angular_radps`
- `brake`
- `safety_state`
- `reason`
- `confidence`
- `source_frame`
- `dry_run`

- [ ] **Step 4: Export public symbols**

Create `src/tunnel_nav/__init__.py` exporting the new dataclasses.

- [ ] **Step 5: Verify and commit**

Run:

```bash
conda run -n lerobot python -m unittest tests.test_bev_dwa_navigation_bridge.CoreModelTest -v
```

Expected: all core model tests pass.

Commit:

```bash
git add src/tunnel_nav/__init__.py src/tunnel_nav/motion.py tests/test_bev_dwa_navigation_bridge.py
git commit -m "Add BEV DWA navigation data model"
```

## Task 2: Fused Mask Export

**Files:**
- Modify: `tools/passable_segmentation/visualize_fused_passable_boundary.py`
- Test: `tests/test_bev_dwa_navigation_bridge.py`

- [ ] **Step 1: Add failing mask export tests**

Add tests requiring `write_fused_masks(output_dir, stem, fused)` to write:

- `safe_passable/<stem>.png`
- `ditch/<stem>.png`
- `left_barrier/<stem>.png`
- `tunnel_wall/<stem>.png`

Each mask must be an 8-bit grayscale PNG with 0 for false and 255 for true.

- [ ] **Step 2: Verify failure**

Run:

```bash
conda run -n lerobot python -m unittest tests.test_bev_dwa_navigation_bridge.FusedMaskExportTest -v
```

Expected: fail because `write_fused_masks` is missing.

- [ ] **Step 3: Implement fused mask export**

Add `write_fused_masks()` to `tools/passable_segmentation/visualize_fused_passable_boundary.py`.

Also add optional `mask_output_dir` to `write_fused_visualizations()` and call `write_fused_masks()` after fusion when provided.

- [ ] **Step 4: Verify and commit**

Run:

```bash
conda run -n lerobot python -m unittest tests.test_bev_dwa_navigation_bridge.FusedMaskExportTest tests.test_passable_segmentation_tools.PassableSegmentationToolsTest -v
```

Expected: mask export and existing segmentation tests pass.

Commit:

```bash
git add tools/passable_segmentation/visualize_fused_passable_boundary.py tests/test_bev_dwa_navigation_bridge.py
git commit -m "Export fused masks for BEV navigation"
```

## Task 3: Simplified BEV, Occupancy Grid, and Risk Grid

**Files:**
- Create: `src/tunnel_nav/bev.py`
- Modify: `src/tunnel_nav/__init__.py`
- Test: `tests/test_bev_dwa_navigation_bridge.py`

- [ ] **Step 1: Add failing BEV tests**

Add tests for:

- a centered `safe_passable` image mask maps to a low-risk center corridor in BEV.
- `ditch` and `tunnel_wall` map to occupied cells with risk `1.0`.
- outside `safe_passable` maps to occupied cells.
- risk dilation around hard boundaries creates a higher-risk band.

Use synthetic masks shaped `(80, 120)` and a BEV grid roughly:

```python
x_min_m=-2.5
x_max_m=2.5
y_min_m=0.0
y_max_m=8.0
resolution_m=0.1
```

- [ ] **Step 2: Verify failure**

Run:

```bash
conda run -n lerobot python -m unittest tests.test_bev_dwa_navigation_bridge.BEVGridTest -v
```

Expected: fail because `src.tunnel_nav.bev` does not exist.

- [ ] **Step 3: Implement pseudo-BEV projection**

Implement `build_pseudo_bev_grid(bundle, config, ...)` in `src/tunnel_nav/bev.py`.

First-version projection rules:

- This is explicitly pseudo-BEV, not calibrated IPM.
- Map image rows to forward distance with a monotonic bottom-to-front transform.
- Map image columns to lateral x with perspective widening toward the bottom.
- Conservative default: unknown cells are occupied.
- Occupancy rules:
  - `ditch` = occupied.
  - `tunnel_wall` = occupied.
  - outside `safe_passable` = occupied.
  - `safe_passable` without hard boundary = free.
- Risk rules:
  - hard boundary = `1.0`.
  - outside `safe_passable` = `1.0`.
  - free road = `0.05-0.20`.
  - near hard boundary after dilation = `0.65-0.90`.

- [ ] **Step 4: Add dilation helper**

Implement a small binary dilation helper using `cv2` when available and a numpy fallback when not available.

The helper must be deterministic for tests.

- [ ] **Step 5: Verify and commit**

Run:

```bash
conda run -n lerobot python -m unittest tests.test_bev_dwa_navigation_bridge.BEVGridTest -v
```

Expected: BEV occupancy/risk tests pass.

Commit:

```bash
git add src/tunnel_nav/__init__.py src/tunnel_nav/bev.py tests/test_bev_dwa_navigation_bridge.py
git commit -m "Add pseudo BEV risk grid generation"
```

## Task 4: Conservative DWA Planner

**Files:**
- Create: `src/tunnel_nav/dwa.py`
- Modify: `src/tunnel_nav/__init__.py`
- Test: `tests/test_bev_dwa_navigation_bridge.py`

- [ ] **Step 1: Add failing DWA tests**

Add tests for:

- clear center corridor selects a forward trajectory.
- all occupied grid returns no feasible trajectory.
- a center hard boundary causes DWA to select a safer side trajectory or stop if no side path exists.
- selected trajectories never include occupied cells.

- [ ] **Step 2: Verify failure**

Run:

```bash
conda run -n lerobot python -m unittest tests.test_bev_dwa_navigation_bridge.DWAPlannerTest -v
```

Expected: fail because `src.tunnel_nav.dwa` does not exist.

- [ ] **Step 3: Implement DWA trajectory simulation**

Implement in `src/tunnel_nav/dwa.py`:

- `sample_controls(dwa_config)`
- `simulate_trajectory(v, omega, dwa_config)`
- `score_trajectory(trajectory, bev_grid, previous_command=None)`
- `select_dwa_trajectory(bev_grid, dwa_config)`

Use differential-drive kinematics:

```text
x += v * sin(yaw) * dt
y += v * cos(yaw) * dt
yaw += omega * dt
```

Grid convention:

- `x`: left/right in meters.
- `y`: forward in meters.
- origin at vehicle center.

Hard constraints:

- any point outside the grid rejects the trajectory.
- any occupied cell rejects the trajectory.
- max risk above stop threshold rejects or heavily penalizes the trajectory.

Score components:

- lower risk is better.
- larger clearance is better.
- forward progress is better.
- smaller absolute angular velocity is smoother.

- [ ] **Step 4: Verify and commit**

Run:

```bash
conda run -n lerobot python -m unittest tests.test_bev_dwa_navigation_bridge.DWAPlannerTest -v
```

Expected: DWA tests pass.

Commit:

```bash
git add src/tunnel_nav/__init__.py src/tunnel_nav/dwa.py tests/test_bev_dwa_navigation_bridge.py
git commit -m "Add conservative DWA planner"
```

## Task 5: Safety Filter and Motion Command Selection

**Files:**
- Create: `src/tunnel_nav/safety.py`
- Modify: `src/tunnel_nav/__init__.py`
- Test: `tests/test_bev_dwa_navigation_bridge.py`

- [ ] **Step 1: Add failing safety tests**

Add tests for:

- no feasible DWA trajectory returns `brake=True` and `S3_STOP`.
- high risk on selected trajectory returns stop.
- medium risk returns `S1_CAUTIOUS` or `S2_SLOWDOWN` with reduced speed.
- valid low-risk trajectory returns `S0_NORMAL`.

- [ ] **Step 2: Verify failure**

Run:

```bash
conda run -n lerobot python -m unittest tests.test_bev_dwa_navigation_bridge.SafetyFilterTest -v
```

Expected: fail because `src.tunnel_nav.safety` does not exist.

- [ ] **Step 3: Implement safety state selection**

Implement `command_from_dwa_result(selected_trajectory, candidates, config, source_frame)` in `src/tunnel_nav/safety.py`.

Safety states:

- `S0_NORMAL`
- `S1_CAUTIOUS`
- `S2_SLOWDOWN`
- `S3_STOP`
- `S4_MANUAL_TAKEOVER`

Rules:

- no feasible trajectory: stop.
- trajectory max risk above stop threshold: stop.
- trajectory close to high-risk cells: slow down.
- low-confidence or malformed input: stop.
- output `MotionCommand(linear_mps, angular_radps, brake, safety_state, reason, confidence, source_frame, dry_run)`.

- [ ] **Step 4: Verify and commit**

Run:

```bash
conda run -n lerobot python -m unittest tests.test_bev_dwa_navigation_bridge.SafetyFilterTest -v
```

Expected: safety tests pass.

Commit:

```bash
git add src/tunnel_nav/__init__.py src/tunnel_nav/safety.py tests/test_bev_dwa_navigation_bridge.py
git commit -m "Add navigation safety filter"
```

## Task 6: Offline BEV / DWA Bridge CLI and Overlays

**Files:**
- Create: `tools/navigation_bridge/run_offline_bev_dwa_bridge.py`
- Test: `tests/test_bev_dwa_navigation_bridge.py`

- [ ] **Step 1: Add failing CLI test**

Add a temporary-directory test that creates:

- `images/frame_001.jpg`
- `masks/safe_passable/frame_001.png`
- `masks/ditch/frame_001.png`
- `masks/left_barrier/frame_001.png`
- `masks/tunnel_wall/frame_001.png`

Expected outputs:

- `commands/frame_001.json`
- `rs232_dry_run/frame_001.json`
- `overlays/frame_001_bev_dwa.jpg`

- [ ] **Step 2: Verify failure**

Run:

```bash
conda run -n lerobot python -m unittest tests.test_bev_dwa_navigation_bridge.OfflineBridgeCliTest -v
```

Expected: fail because the CLI module does not exist.

- [ ] **Step 3: Implement offline CLI**

CLI arguments:

- `--image-dir`
- `--mask-dir`
- `--output-dir`
- `--max-speed-mps`
- `--max-angular-radps`
- `--grid-resolution-m`
- `--dry-run`

Pipeline:

```text
read RGB + masks
→ MaskBundle
→ pseudo BEV occupancy/risk grid
→ DWA candidate trajectories
→ safety filter
→ MotionCommand JSON
→ RS232 dry-run JSON
→ overlay with image, BEV risk grid, candidate trajectories, selected trajectory, command stats
```

- [ ] **Step 4: Verify and commit**

Run:

```bash
conda run -n lerobot python -m unittest tests.test_bev_dwa_navigation_bridge.OfflineBridgeCliTest -v
```

Expected: CLI test passes and writes JSON plus overlay.

Commit:

```bash
git add tools/navigation_bridge/run_offline_bev_dwa_bridge.py tests/test_bev_dwa_navigation_bridge.py
git commit -m "Add offline BEV DWA navigation bridge"
```

## Task 7: RS232 Dry-Run Adapter

**Files:**
- Create: `src/tunnel_nav/rs232.py`
- Modify: `src/tunnel_nav/__init__.py`
- Test: `tests/test_bev_dwa_navigation_bridge.py`

- [ ] **Step 1: Add failing RS232 tests**

Tests must cover:

- Modbus CRC16 helper.
- velocity clamping.
- signed int16 conversion.
- `angular_sign` inversion.
- dry-run adapter does not import or open serial.
- brake command converts to zero velocity registers.

- [ ] **Step 2: Verify failure**

Run:

```bash
conda run -n lerobot python -m unittest tests.test_bev_dwa_navigation_bridge.Rs232AdapterTest -v
```

Expected: fail because `src.tunnel_nav.rs232` does not exist.

- [ ] **Step 3: Implement RS232 dry-run conversion**

Implement constants:

- `REG_LINEAR_VEL = 1040`
- `REG_ANGULAR_VEL = 1041`
- `REG_FUNC_CTRL = 1045`
- `REG_DRV_ENABLE = 1049`

Implement:

- `modbus_crc16(data)`
- `clamp_int16(value)`
- `velocity_to_registers(linear_mps, angular_radps, max_speed_mps, max_angular_radps, angular_sign)`
- `Rs232DryRunAdapter.send(command, config)`

Register conversion:

```text
linear register = int(clamp(linear_mps) * 1000)
angular register = int(clamp(angular_radps) * angular_sign * 1000)
brake=True -> both registers = 0
```

- [ ] **Step 4: Verify and commit**

Run:

```bash
conda run -n lerobot python -m unittest tests.test_bev_dwa_navigation_bridge.Rs232AdapterTest -v
```

Expected: RS232 dry-run tests pass.

Commit:

```bash
git add src/tunnel_nav/__init__.py src/tunnel_nav/rs232.py tests/test_bev_dwa_navigation_bridge.py
git commit -m "Add RS232 dry-run adapter"
```

## Task 8: Configuration, Documentation, and Final Verification

**Files:**
- Modify: `configs/robot/vehicle.yaml`
- Modify: `docs/progress/LOG.md`

- [ ] **Step 1: Update config**

Update `configs/robot/vehicle.yaml` with conservative defaults:

```yaml
vehicle:
  name: tunnel_ugv
  drive_type: differential
  max_speed_mps: 0.5
  max_steering_rad: 0.5
  max_angular_radps: 0.5
  angular_sign: 1
  safety_margin_m: 0.5
  rs232:
    port: /dev/ttyUSB0
    baudrate: 115200
    node_addr: 6
    dry_run: true
    live_requires_explicit_flag: true
```

Add a new `navigation_bridge` block if configuration loading needs tool-specific defaults:

```yaml
navigation_bridge:
  bev:
    x_min_m: -2.5
    x_max_m: 2.5
    y_min_m: 0.0
    y_max_m: 8.0
    resolution_m: 0.1
    projection: pseudo_bev
  dwa:
    max_velocity_mps: 0.10
    max_angular_radps: 0.50
    velocity_samples: 3
    angular_samples: 9
    predict_time_s: 2.0
    dt_s: 0.2
```

- [ ] **Step 2: Update progress log**

Record that the active execution plan changed from image-space-first to simplified BEV / Risk Grid / DWA offline prototype.

- [ ] **Step 3: Run focused tests**

Run:

```bash
conda run -n lerobot python -m unittest tests.test_bev_dwa_navigation_bridge -v
```

Expected: all BEV / DWA navigation bridge tests pass.

- [ ] **Step 4: Run existing segmentation regression tests**

Run:

```bash
conda run -n lerobot python -m unittest tests.test_passable_segmentation_tools.PassableSegmentationToolsTest -v
```

Expected: existing passable segmentation tests pass.

- [ ] **Step 5: Run syntax checks**

Run:

```bash
conda run -n lerobot python -m py_compile src/tunnel_nav/*.py tools/navigation_bridge/run_offline_bev_dwa_bridge.py tools/passable_segmentation/visualize_fused_passable_boundary.py tests/test_bev_dwa_navigation_bridge.py
```

Expected: no output and exit code 0.

- [ ] **Step 6: Inspect git diff and avoid unrelated files**

Run:

```bash
git status --short --branch
git diff --stat
```

Do not stage unrelated files:

- `1/`
- `archive/1.zip`
- `tunnel_ugv_plan_A_BEV_DWA_trajectory.md` unless the user asks to preserve it in Git.
- unrelated LAN file server files.

- [ ] **Step 7: Final commit**

Run:

```bash
git add src/tunnel_nav tools/navigation_bridge tools/passable_segmentation/visualize_fused_passable_boundary.py tests/test_bev_dwa_navigation_bridge.py configs/robot/vehicle.yaml docs/progress/LOG.md
git commit -m "Add offline BEV DWA RS232 navigation bridge"
```

Expected: commit succeeds with only BEV / DWA / RS232 navigation bridge files staged.

## Acceptance Criteria

- Fused masks can be exported as PNGs for `safe_passable`, `ditch`, `left_barrier`, and `tunnel_wall`.
- Offline CLI converts saved masks into pseudo-BEV occupancy and risk grids.
- DWA produces candidate trajectories over the risk grid.
- Unsafe grids produce stop commands with clear reasons.
- Output includes command JSON, RS232 dry-run JSON, and overlay visualization.
- RS232 adapter can be tested without hardware.
- No first-phase code opens serial by default.
- Angular sign is configurable.
- New runtime path does not import the legacy baseline.
- New BEV / DWA tests and existing segmentation tests pass.

## Explicit Non-Goals

- No live vehicle control.
- No real calibrated IPM until camera calibration is measured.
- No object detection dependency in the first version.
- No OpenPilot-style learned waypoint head in this phase.
- No direct model-to-motor control.

