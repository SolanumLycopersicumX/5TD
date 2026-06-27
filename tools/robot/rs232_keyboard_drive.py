#!/usr/bin/env python3
"""Keyboard teleop for the RS232 / Modbus RTU vehicle driver."""
from __future__ import annotations

import argparse
import importlib.util
import select
import sys
import termios
import time
import tty
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DRIVER_PATH = ROOT / "1" / "driver_controller.py"

HELP = """
RS232 keyboard control

Hold W/S: forward/backward
Hold A/D: rotate left/right
Hold I/J/L: forward arc left/straight/right
Release keys, Space, or K: stop
Q: quit and stop

Keep one hand near the physical emergency stop. Start with wheels clear or the
vehicle restrained if possible.
""".strip()


def parse_node_addr(value: str) -> int:
    addr = int(value, 0)
    if not 0 <= addr <= 0xFF:
        raise argparse.ArgumentTypeError("node address must be between 0x00 and 0xFF")
    return addr


def select_control_key(keys: str) -> str | None:
    if not keys:
        return None

    normalized = keys.lower()
    if "q" in normalized:
        return "q"
    if " " in normalized:
        return " "
    if "k" in normalized:
        return "k"
    return normalized[-1]


def read_key(timeout_s: float) -> str | None:
    readable, _, _ = select.select([sys.stdin], [], [], timeout_s)
    if not readable:
        return None

    keys = [sys.stdin.read(1)]
    while select.select([sys.stdin], [], [], 0)[0]:
        keys.append(sys.stdin.read(1))
    return select_control_key("".join(keys))


def command_for_key(key: str | None, linear: float, angular: float) -> tuple[float, float]:
    if key == "w":
        return linear, 0.0
    if key == "s":
        return -linear, 0.0
    if key == "a":
        return 0.0, angular
    if key == "d":
        return 0.0, -angular
    if key == "i":
        return linear, 0.0
    if key == "j":
        return linear, angular
    if key == "l":
        return linear, -angular
    return 0.0, 0.0


def load_driver_module():
    spec = importlib.util.spec_from_file_location("legacy_rs232_driver_controller", DRIVER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load RS232 driver from {DRIVER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_controller(*, port: str, addr: int, baudrate: int):
    module = load_driver_module()
    controller = module.VehicleController(port=port, addr=addr, baudrate=baudrate)
    if not getattr(controller, "ser", None):
        raise RuntimeError(f"Failed to open serial port {port}")
    return controller


def run_keyboard(
    *,
    port: str,
    addr: int,
    baudrate: int,
    linear: float,
    angular: float,
    rate_hz: float,
    enable: bool,
    release_estop: bool,
) -> int:
    if rate_hz <= 0:
        raise ValueError("rate_hz must be positive")

    vehicle = build_controller(port=port, addr=addr, baudrate=baudrate)
    period = 1.0 / rate_hz
    old_settings = termios.tcgetattr(sys.stdin)

    print(HELP)
    print(f"Port={port} addr=0x{addr:02X} baud={baudrate} linear={linear:.3f}m/s angular={angular:.3f}rad/s")
    print("Press Ctrl+C or Q to quit; finalizer sends stop and disable.")

    try:
        tty.setcbreak(sys.stdin.fileno())
        vehicle.stop()
        if release_estop:
            vehicle.emergency_release()
            time.sleep(0.1)
        if enable:
            vehicle.enable()
            time.sleep(0.1)

        while True:
            key = read_key(period)
            if key == "q":
                break
            command = command_for_key(key, linear, angular)
            vehicle.set_velocity(*command)
            time.sleep(0.001)
    finally:
        try:
            vehicle.stop()
            time.sleep(0.1)
            vehicle.disable()
        finally:
            vehicle.close()
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="/dev/ttyUSB0", help="RS232 serial device")
    parser.add_argument("--addr", type=parse_node_addr, default=0x06, help="Modbus node address, e.g. 0x06")
    parser.add_argument("--baudrate", type=int, default=115200, help="serial baudrate")
    parser.add_argument("--linear", type=float, default=0.03, help="linear speed in m/s")
    parser.add_argument("--angular", type=float, default=0.10, help="angular speed in rad/s")
    parser.add_argument("--rate-hz", type=float, default=10.0, help="command publish rate")
    parser.add_argument("--enable", action="store_true", help="write driver enable register before keyboard loop")
    parser.add_argument("--release-estop", action="store_true", help="clear driver emergency-stop bit before keyboard loop")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    raise SystemExit(
        run_keyboard(
            port=args.port,
            addr=args.addr,
            baudrate=args.baudrate,
            linear=args.linear,
            angular=args.angular,
            rate_hz=args.rate_hz,
            enable=args.enable,
            release_estop=args.release_estop,
        )
    )


if __name__ == "__main__":
    main()
