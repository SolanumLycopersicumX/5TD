# RS232 Navigation Bridge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **Status:** Superseded as the active execution plan by `docs/superpowers/plans/2026-06-24-bev-dwa-rs232-navigation-bridge.md`. Keep this document as the earlier RS232/image-space bridge reference.

**Goal:** Convert the current fused passable-road segmentation output into conservative motion commands, then prepare a default-dry-run RS232 adapter for the vehicle controller.

**Architecture:** Keep new runtime code under `src/tunnel_nav`. The old `baselines/vision_obstacle_avoidance_legacy` code is reference-only, not a runtime dependency. The first implementation consumes saved fused masks, writes command JSON and overlays, and never opens a serial port unless a later live mode is explicitly enabled.

**Tech Stack:** Python, `numpy`, `Pillow`, optional `cv2`, `unittest`, existing `conda run -n lerobot python` workflow.

---

## Current Context

- Current best perception route is staged fusion:
  - main passable/ditch model: `runs/passable_ego/passable_ditch_artifact_v3_finetune/best_model.pt`
  - auxiliary boundary/wall model: `runs/passable_ego/boundary_wall_aux_v2_no_testvideo/best_model.pt`
  - fusion script: `tools/passable_segmentation/visualize_fused_passable_boundary.py`
- Current fusion rules:
  - `ditch` has highest priority.
  - `tunnel_wall` is always non-passable.
  - `left_barrier` is a boundary cue only.
  - `safe_passable = passable - ditch - tunnel_wall`.
- Driver choice: RS232 / Modbus RTU, based on `1.zip` / `driver_controller.py`.
- Important risk: angular sign is inconsistent between the old baseline and the new driver file:
  - new driver file says positive angular velocity turns left / counter-clockwise.
  - old baseline comments say left is negative and right is positive.
  - therefore `angular_sign` must be configurable and verified before live motion.

## Files to Create or Modify

- Create `src/tunnel_nav/__init__.py`: package exports.
- Create `src/tunnel_nav/motion.py`: `MotionCommand`, `NavigationConfig`, `MaskBundle`, `PathCandidate`.
- Create `src/tunnel_nav/mask_planner.py`: convert fused masks to candidate paths and conservative commands.
- Create `src/tunnel_nav/rs232.py`: Modbus CRC, velocity-to-register conversion, dry-run adapter.
- Create `tools/navigation_bridge/run_offline_navigation_bridge.py`: offline CLI for saved masks.
- Modify `tools/passable_segmentation/visualize_fused_passable_boundary.py`: optionally export fused masks.
- Modify `configs/robot/vehicle.yaml`: add differential-drive and RS232 dry-run defaults.
- Create `tests/test_navigation_bridge.py`: unit tests for the new runtime path.
- Update `docs/progress/LOG.md`: record implementation and verification results.

## Task 1: Motion Data Model

**Files:**
- Create: `src/tunnel_nav/__init__.py`
- Create: `src/tunnel_nav/motion.py`
- Test: `tests/test_navigation_bridge.py`

- [ ] Add tests for `MotionCommand.to_dict()` and conservative `NavigationConfig` defaults.
- [ ] Expected initial failure: `src.tunnel_nav.motion` does not exist.
- [ ] Implement `MotionCommand` with physical units: `linear_mps`, `angular_radps`, `stop`, `reason`, `confidence`, `source_frame`, `dry_run`.
- [ ] Implement `NavigationConfig` with conservative defaults:
  - `max_speed_mps = 0.10`
  - `max_angular_radps = 0.50`
  - `angular_sign = 1`
  - `dry_run = True`
  - `live_requires_explicit_flag = True`
- [ ] Implement `MaskBundle` and `PathCandidate`.
- [ ] Verify:

```bash
conda run -n lerobot python -m unittest tests.test_navigation_bridge.MotionModelTest -v
```

- [ ] Commit only these files:

```bash
git add src/tunnel_nav/__init__.py src/tunnel_nav/motion.py tests/test_navigation_bridge.py
git commit -m "Add navigation bridge motion model"
```

## Task 2: Mask Planner and Safety Filter

**Files:**
- Create: `src/tunnel_nav/mask_planner.py`
- Modify: `src/tunnel_nav/__init__.py`
- Test: `tests/test_navigation_bridge.py`

- [ ] Add tests with synthetic masks:
  - center passable corridor should produce a forward command.
  - empty passable mask should stop with reason `no bottom-connected passable region`.
  - ditch or tunnel wall crossing selected path should stop with an unsafe reason.
- [ ] Expected initial failure: `plan_motion_from_masks` does not exist.
- [ ] Implement bottom-connected passable filtering.
- [ ] Generate simple image-space candidate paths from bottom center toward several target x positions.
- [ ] Score candidates by center preference and clearance from `ditch | tunnel_wall`.
- [ ] Treat `left_barrier` only as a soft boundary cue, not as ditch.
- [ ] Output `MotionCommand` in physical units, not normalized steering.
- [ ] Verify:

```bash
conda run -n lerobot python -m unittest tests.test_navigation_bridge.MaskPlannerTest -v
```

- [ ] Commit:

```bash
git add src/tunnel_nav/__init__.py src/tunnel_nav/mask_planner.py tests/test_navigation_bridge.py
git commit -m "Add mask-based navigation planner"
```

## Task 3: Fused Mask Export

**Files:**
- Modify: `tools/passable_segmentation/visualize_fused_passable_boundary.py`
- Test: `tests/test_navigation_bridge.py`

- [ ] Add test for writing four mask directories:
  - `safe_passable/<stem>.png`
  - `ditch/<stem>.png`
  - `left_barrier/<stem>.png`
  - `tunnel_wall/<stem>.png`
- [ ] Expected initial failure: `write_fused_masks` does not exist.
- [ ] Add `write_fused_masks(output_dir, stem, fused)` helper.
- [ ] Add optional `mask_output_dir` argument to `write_fused_visualizations`.
- [ ] Keep existing overlay behavior unchanged.
- [ ] Verify:

```bash
conda run -n lerobot python -m unittest tests.test_navigation_bridge.FusedMaskExportTest tests.test_passable_segmentation_tools.PassableSegmentationToolsTest -v
```

- [ ] Commit:

```bash
git add tools/passable_segmentation/visualize_fused_passable_boundary.py tests/test_navigation_bridge.py
git commit -m "Export fused navigation masks"
```

## Task 4: Offline Navigation Bridge CLI

**Files:**
- Create: `tools/navigation_bridge/run_offline_navigation_bridge.py`
- Test: `tests/test_navigation_bridge.py`

- [ ] Add test that creates a temporary image and four mask PNGs.
- [ ] Expected output:
  - `commands/<stem>.json`
  - `overlays/<stem>_navigation.jpg`
- [ ] Expected initial failure: CLI module does not exist.
- [ ] Implement CLI arguments:
  - `--image-dir`
  - `--mask-dir`
  - `--output-dir`
  - `--max-speed-mps`
  - `--max-angular-radps`
- [ ] The CLI must always run dry-run planning and must not import or open serial.
- [ ] Verify:

```bash
conda run -n lerobot python -m unittest tests.test_navigation_bridge.OfflineNavigationBridgeCliTest -v
```

- [ ] Commit:

```bash
git add tools/navigation_bridge/run_offline_navigation_bridge.py tests/test_navigation_bridge.py
git commit -m "Add offline navigation bridge CLI"
```

## Task 5: RS232 Dry-Run Adapter

**Files:**
- Create: `src/tunnel_nav/rs232.py`
- Modify: `src/tunnel_nav/__init__.py`
- Test: `tests/test_navigation_bridge.py`

- [ ] Add tests for:
  - Modbus CRC16 helper.
  - velocity clamping.
  - signed int16 conversion.
  - `angular_sign` inversion.
  - dry-run adapter not opening serial.
- [ ] Expected initial failure: `src.tunnel_nav.rs232` does not exist.
- [ ] Implement constants:
  - `REG_LINEAR_VEL = 1040`
  - `REG_ANGULAR_VEL = 1041`
  - `REG_FUNC_CTRL = 1045`
  - `REG_DRV_ENABLE = 1049`
- [ ] Implement `velocity_to_registers()`:
  - linear unit: `m/s * 1000`
  - angular unit: `rad/s * angular_sign * 1000`
  - clamp both to signed int16.
- [ ] Implement `Rs232DryRunAdapter.send()` returning intended register values as a dictionary.
- [ ] Do not import `serial` in the dry-run path.
- [ ] Verify:

```bash
conda run -n lerobot python -m unittest tests.test_navigation_bridge.Rs232AdapterTest -v
```

- [ ] Commit:

```bash
git add src/tunnel_nav/__init__.py src/tunnel_nav/rs232.py tests/test_navigation_bridge.py
git commit -m "Add RS232 dry-run adapter"
```

## Task 6: Configuration and Final Verification

**Files:**
- Modify: `configs/robot/vehicle.yaml`
- Modify: `docs/progress/LOG.md`

- [ ] Update `configs/robot/vehicle.yaml` with:

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

- [ ] Run focused tests:

```bash
conda run -n lerobot python -m unittest tests.test_navigation_bridge -v
```

- [ ] Run existing segmentation tests:

```bash
conda run -n lerobot python -m unittest tests.test_passable_segmentation_tools.PassableSegmentationToolsTest -v
```

- [ ] Run syntax checks:

```bash
conda run -n lerobot python -m py_compile src/tunnel_nav/*.py tools/navigation_bridge/run_offline_navigation_bridge.py tools/passable_segmentation/visualize_fused_passable_boundary.py tests/test_navigation_bridge.py
```

- [ ] Inspect changed files before the final commit:

```bash
git status --short --branch
git diff --stat
```

- [ ] Do not stage unrelated LAN file server files, `1.zip`, `archive/1.zip`, `1/`, or unrelated `.gitignore` changes.
- [ ] Final commit:

```bash
git add src/tunnel_nav tools/navigation_bridge tools/passable_segmentation/visualize_fused_passable_boundary.py tests/test_navigation_bridge.py configs/robot/vehicle.yaml docs/progress/LOG.md
git commit -m "Add offline RS232 navigation bridge"
```

## Final Acceptance Criteria

- Offline bridge reads saved fused masks and writes command JSON.
- Offline bridge writes a visual overlay showing selected/rejected path candidates.
- Unsafe input masks produce stop commands with clear reasons.
- RS232 adapter can be tested without hardware.
- No first-phase code opens serial by default.
- New runtime path does not import the legacy baseline.
- Angular sign is configurable and documented as needing physical validation.
- All focused and existing segmentation tests pass.
