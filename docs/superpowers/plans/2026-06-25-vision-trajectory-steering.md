# Vision Trajectory Steering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a low-speed camera-based trajectory follower that derives a centerline from the fused safe-passable mask and sends linear/angular RS232 commands to the differential-drive vehicle.

**Architecture:** Keep perception inference unchanged. Add a pure trajectory module that converts fused masks into path points and a bounded velocity command, then add a new runtime script that connects webcam inference, trajectory drawing, and RS232 `set_velocity(linear, angular)`.

**Tech Stack:** Python, NumPy, OpenCV, Tk/Pillow display backend, existing PyTorch segmentation models, existing RS232 Modbus driver.

---

### Task 1: Pure Trajectory Logic

**Files:**
- Create: `src/tunnel_nav/vision_trajectory.py`
- Test: `tests/test_vision_trajectory.py`

- [x] Write tests for centerline extraction, left/right steering sign, missing path stop, and hazard stop.
- [x] Implement dataclasses for config, trajectory, and command result.
- [x] Implement scanline center extraction over the safe-passable mask.
- [x] Implement target point selection and bounded proportional angular command.
- [x] Run `python3 -m unittest tests/test_vision_trajectory.py -v`.

### Task 2: Runtime Script

**Files:**
- Create: `tools/robot/vision_autodrive_trajectory.py`
- Test: `tests/test_vision_autodrive_trajectory_script.py`

- [x] Write parser tests for low-speed defaults and explicit driver enable behavior.
- [x] Reuse model loading and display helpers from `tools/passable_segmentation/live_webcam_fused_preview.py`.
- [x] Reuse RS232 controller building and node address parsing from `tools/robot/rs232_keyboard_drive.py`.
- [x] Send `(linear, angular)` only when the trajectory command permits motion; otherwise ramp to zero.
- [x] Finalizer must stop and disable the vehicle on window close, q/Esc, Ctrl+C, camera failure, or exception.

### Task 3: Verification and Log

**Files:**
- Modify: `docs/progress/LOG.md`

- [x] Run unit tests for trajectory and script parser.
- [x] Run `py_compile` for new module and script.
- [x] Verify `lerobot` runtime imports torch, cv2, Tk/Pillow, serial, and the new script.
- [x] Append a concise progress log entry with command and safety defaults.
