#!/usr/bin/env python3
"""Keyboard teleop that uses the driver-style Gazebo velocity adapter."""
from __future__ import annotations

import argparse
import select
import sys
import termios
import time
import tty
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.tunnel_nav.gazebo_control import GazeboCmdVelAdapter

HELP = """
Driver-style Gazebo keyboard control

Hold W/S: forward/backward
Hold A/D: rotate left/right
Hold I/J/L: forward arc left/straight/right
Release keys, Space, or K: stop
Q: quit
""".strip()


def _select_control_key(keys: str) -> str | None:
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


def _read_key(timeout_s: float) -> str | None:
    readable, _, _ = select.select([sys.stdin], [], [], timeout_s)
    if not readable:
        return None

    keys = [sys.stdin.read(1)]
    while select.select([sys.stdin], [], [], 0)[0]:
        keys.append(sys.stdin.read(1))
    return _select_control_key("".join(keys))


def _command_for_key(key: str | None, linear: float, angular: float) -> tuple[float, float] | None:
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
    if key in {" ", "k"}:
        return 0.0, 0.0
    return 0.0, 0.0


def run_keyboard(*, linear: float, angular: float, rate_hz: float, topic: str) -> int:
    adapter = GazeboCmdVelAdapter(topic=topic)
    period = 1.0 / rate_hz
    old_settings = termios.tcgetattr(sys.stdin)
    print(HELP)
    try:
        tty.setcbreak(sys.stdin.fileno())
        while True:
            key = _read_key(period)
            if key == "q":
                break
            command = _command_for_key(key, linear, angular)
            adapter.set_velocity(*command)
            time.sleep(0.001)
    finally:
        adapter.stop()
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--linear", type=float, default=0.10, help="linear speed in m/s")
    parser.add_argument("--angular", type=float, default=0.30, help="angular speed in rad/s")
    parser.add_argument("--rate-hz", type=float, default=5.0, help="publish rate")
    parser.add_argument("--topic", default="/cmd_vel", help="Gazebo cmd_vel topic")
    args = parser.parse_args()
    raise SystemExit(run_keyboard(linear=args.linear, angular=args.angular, rate_hz=args.rate_hz, topic=args.topic))


if __name__ == "__main__":
    main()
