#!/usr/bin/env python3
"""Low-speed vision-gated autonomous forward drive over RS232."""
from __future__ import annotations

import argparse
import contextlib
import io
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tunnel_nav.manual_control import ManualControlLimits, ramp_command
from src.tunnel_nav.vision_autodrive import DriveGateConfig, evaluate_drive_gate
from tools.passable_segmentation.live_webcam_fused_preview import (
    DEFAULT_BOUNDARY_CHECKPOINT,
    DEFAULT_PASSABLE_CHECKPOINT,
    _load_runtime_helpers,
    _require_runtime_modules,
    make_display_backend,
    parse_camera,
    scale_to_display_width,
)
from tools.robot.rs232_keyboard_drive import build_controller, parse_node_addr


class DryRunVehicle:
    """No-op vehicle controller for perception-only deployment tests."""

    def __init__(self) -> None:
        self.commands: list[tuple[float, float]] = []

    def set_velocity(self, linear_mps: float, angular_radps: float) -> None:
        self.commands.append((float(linear_mps), float(angular_radps)))

    def stop(self) -> None:
        self.set_velocity(0.0, 0.0)

    def enable(self) -> None:
        pass

    def disable(self) -> None:
        pass

    def emergency_release(self) -> None:
        pass

    def close(self) -> None:
        pass


def build_vehicle(args: argparse.Namespace) -> Any:
    if args.dry_run:
        return DryRunVehicle()
    return build_controller(port=args.port, addr=args.addr, baudrate=args.baudrate)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", default="/dev/video0", help="camera index, V4L2 device, or video URL")
    parser.add_argument("--width", type=int, default=640, help="capture width")
    parser.add_argument("--height", type=int, default=480, help="capture height")
    parser.add_argument("--fps", type=int, default=15, help="requested capture FPS")
    parser.add_argument("--display-width", type=int, default=1280, help="maximum display window image width")
    parser.add_argument("--display-backend", choices=("tk", "opencv"), default="tk", help="preview window backend")
    parser.add_argument("--no-display", action="store_true", help="run without opening a preview window")
    parser.add_argument("--port", default="/dev/ttyUSB0", help="RS232 serial device")
    parser.add_argument("--addr", type=parse_node_addr, default=0x06, help="Modbus node address, e.g. 0x06")
    parser.add_argument("--baudrate", type=int, default=115200, help="serial baudrate")
    parser.add_argument("--linear", type=float, default=0.015, help="allowed forward speed in m/s")
    parser.add_argument("--rate-hz", type=float, default=8.0, help="maximum command send rate")
    parser.add_argument("--max-linear-accel", type=float, default=0.04, help="linear speed ramp limit in m/s^2")
    parser.add_argument("--min-safe-ratio", type=float, default=0.65, help="minimum safe-passable area ratio in the drive ROI")
    parser.add_argument("--max-hazard-ratio", type=float, default=0.02, help="maximum hazard area ratio in the drive ROI")
    parser.add_argument("--roi-x-min", type=float, default=0.35, help="drive ROI left edge as image fraction")
    parser.add_argument("--roi-x-max", type=float, default=0.65, help="drive ROI right edge as image fraction")
    parser.add_argument("--roi-y-min", type=float, default=0.60, help="drive ROI top edge as image fraction")
    parser.add_argument("--roi-y-max", type=float, default=0.95, help="drive ROI bottom edge as image fraction")
    parser.add_argument("--enable-driver", action="store_true", help="enable the driver and allow nonzero forward commands")
    parser.add_argument("--release-estop", action="store_true", help="clear driver emergency-stop bit before the loop")
    parser.add_argument("--dry-run", action="store_true", help="do not open RS232; log no-op vehicle commands only")
    parser.add_argument("--cpu", action="store_true", help="force CPU inference even if CUDA is available")
    parser.add_argument("--passable-checkpoint", type=Path, default=DEFAULT_PASSABLE_CHECKPOINT)
    parser.add_argument("--boundary-checkpoint", type=Path, default=DEFAULT_BOUNDARY_CHECKPOINT)
    return parser


def _quiet_call(obj: Any, name: str, *args: Any) -> Any:
    method = getattr(obj, name)
    with contextlib.redirect_stdout(io.StringIO()):
        return method(*args)


def _safe_stop(vehicle: Any) -> None:
    with contextlib.suppress(Exception):
        _quiet_call(vehicle, "stop")


def _draw_status(cv2: Any, display_bgr: Any, text: str, decision: Any) -> None:
    color = (40, 220, 40) if decision.allow_forward else (40, 40, 230)
    x0, y0, x1, y1 = decision.roi_bounds
    panel_width = display_bgr.shape[1] // 2
    cv2.rectangle(display_bgr, (panel_width + x0, y0), (panel_width + x1, y1), color, 2)
    cv2.putText(display_bgr, text, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(display_bgr, text, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)


def run_autodrive(args: argparse.Namespace) -> int:
    if args.rate_hz <= 0:
        raise ValueError("rate_hz must be positive")
    if args.linear < 0:
        raise ValueError("linear speed must be non-negative")
    if not args.passable_checkpoint.exists():
        raise FileNotFoundError(args.passable_checkpoint)
    if not args.boundary_checkpoint.exists():
        raise FileNotFoundError(args.boundary_checkpoint)

    cv2, _np, torch = _require_runtime_modules()
    helpers = _load_runtime_helpers()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    print(f"[INFO] Loading models on {device}...")
    passable_model = helpers["load_model"](
        args.passable_checkpoint,
        expected_labels=helpers["PASSABLE_LABELS"],
        device=device,
    )
    boundary_model = helpers["load_model"](
        args.boundary_checkpoint,
        expected_labels=helpers["BOUNDARY_LABELS"],
        device=device,
    )

    cap = cv2.VideoCapture(parse_camera(str(args.camera)))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera: {args.camera}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(args.width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(args.height))
    cap.set(cv2.CAP_PROP_FPS, int(args.fps))

    vehicle = build_vehicle(args)
    preview_window = None
    window_name = "Vision Autodrive Forward"
    if not args.no_display:
        preview_window = make_display_backend(args.display_backend, cv2, window_name)

    gate_config = DriveGateConfig(
        roi_x_min=args.roi_x_min,
        roi_x_max=args.roi_x_max,
        roi_y_min=args.roi_y_min,
        roi_y_max=args.roi_y_max,
        min_safe_ratio=args.min_safe_ratio,
        max_hazard_ratio=args.max_hazard_ratio,
    )
    ramp_limits = ManualControlLimits(
        max_linear_mps=max(0.0, float(args.linear)),
        max_angular_radps=0.0,
        max_linear_accel_mps2=max(0.0, float(args.max_linear_accel)),
        max_angular_accel_radps2=0.0,
    )

    drive_armed = bool(args.enable_driver)
    current_command = (0.0, 0.0)
    period = 1.0 / float(args.rate_hz)
    last_send = 0.0
    last_loop = time.monotonic()
    last_print = 0.0

    print(f"[INFO] Port={args.port} addr=0x{args.addr:02X} speed={args.linear:.3f} m/s armed={drive_armed}")
    print("[INFO] Close the window or press q/Esc in the preview window to stop. Ctrl+C also stops.")
    if args.dry_run:
        print("[WARN] Dry-run vehicle controller; no RS232 serial port will be opened.")
    if not drive_armed:
        print("[WARN] Driver is not enabled; this run will visualize and send zero-speed stop commands only.")

    try:
        _safe_stop(vehicle)
        if args.release_estop:
            _quiet_call(vehicle, "emergency_release")
            time.sleep(0.1)
        if args.enable_driver:
            _quiet_call(vehicle, "enable")
            time.sleep(0.1)

        while True:
            ok, frame_bgr = cap.read()
            now = time.monotonic()
            dt = max(0.0, now - last_loop)
            last_loop = now

            if not ok:
                target = (0.0, 0.0)
                current_command = ramp_command(current_command, target, ramp_limits, dt)
                _quiet_call(vehicle, "set_velocity", *current_command)
                print("[WARN] Camera frame read failed; stopping")
                time.sleep(period)
                continue

            image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            passable_probs = helpers["predict_probabilities"](passable_model, image_rgb, device)
            boundary_probs = helpers["predict_probabilities"](boundary_model, image_rgb, device)
            height, width = passable_probs.shape[1:]
            image_resized = cv2.resize(image_rgb, (width, height), interpolation=cv2.INTER_AREA)
            fused = helpers["fuse_passable_boundary_predictions"](passable_probs, boundary_probs)
            decision = evaluate_drive_gate(fused, gate_config)

            target_linear = float(args.linear) if drive_armed and decision.allow_forward else 0.0
            current_command = ramp_command(current_command, (target_linear, 0.0), ramp_limits, dt)
            if now - last_send >= period:
                _quiet_call(vehicle, "set_velocity", *current_command)
                last_send = now

            if now - last_print >= 0.5:
                print(
                    f"[STATE] {decision.reason} safe={decision.safe_ratio:.2f} "
                    f"hazard={decision.hazard_ratio:.2f} cmd={current_command[0]:.3f} m/s"
                )
                last_print = now

            if preview_window is not None:
                canvas_rgb = helpers["make_fused_overlay"](image_resized, fused)
                display = cv2.cvtColor(canvas_rgb, cv2.COLOR_RGB2BGR)
                status = (
                    f"{decision.reason} safe={decision.safe_ratio:.2f} hazard={decision.hazard_ratio:.2f} "
                    f"cmd={current_command[0]:.3f}m/s q/Esc stop"
                )
                _draw_status(cv2, display, status, decision)
                scaled_w, scaled_h = scale_to_display_width(display.shape[1], display.shape[0], args.display_width)
                if (scaled_w, scaled_h) != (display.shape[1], display.shape[0]):
                    display = cv2.resize(display, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
                if not preview_window.show(display):
                    break
    except KeyboardInterrupt:
        print("\n[INFO] Keyboard interrupt; stopping")
    finally:
        _safe_stop(vehicle)
        time.sleep(0.1)
        with contextlib.suppress(Exception):
            _quiet_call(vehicle, "disable")
        with contextlib.suppress(Exception):
            vehicle.close()
        cap.release()
        if preview_window is not None:
            preview_window.close()
        print("[OK] Vehicle stopped and driver disabled")
    return 0


def main() -> None:
    args = build_arg_parser().parse_args()
    raise SystemExit(run_autodrive(args))


if __name__ == "__main__":
    main()
