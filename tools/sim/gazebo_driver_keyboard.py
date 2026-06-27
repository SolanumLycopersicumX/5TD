#!/usr/bin/env python3
"""Keyboard teleop that uses the driver-style Gazebo velocity adapter."""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import select
import sys
import termios
import time
import tty
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.tunnel_nav.gazebo_control import GazeboCmdVelAdapter

TERMINAL_HELP = """
Driver-style Gazebo terminal keyboard control

This backend reads terminal characters, so it cannot receive true key-release events.
Use the default Tk backend for immediate stop-on-release behavior.

Hold W/S: forward/backward
Hold A/D: rotate left/right
Hold I/J/L: forward arc left/straight/right
Release keys, Space, or K: stop
Q: quit
""".strip()

TK_HELP = """
Gazebo keyboard control

Click this control window first. Hold a movement key to command motion; release it to brake.

W/S: forward/backward
A/D: rotate left/right
W+A / W+D: forward arc left/right
Space or K: stop
Q: quit
""".strip()

_MOVEMENT_KEYS = {"w", "a", "s", "d", "i", "j", "l"}


@dataclass
class _KeyPressState:
    linear: float
    angular: float
    pressed: set[str] = field(default_factory=set)

    def command(self) -> tuple[float, float]:
        return _command_for_pressed_keys(self.pressed, self.linear, self.angular)

    def press(self, key: str | None) -> tuple[float, float] | None:
        if key in {" ", "k"}:
            return self.stop()
        if key not in _MOVEMENT_KEYS:
            return None
        if key in self.pressed:
            return None
        self.pressed.add(key)
        return self.command()

    def release(self, key: str | None) -> tuple[float, float] | None:
        if key not in _MOVEMENT_KEYS or key not in self.pressed:
            return None
        self.pressed.discard(key)
        return self.command()

    def stop(self) -> tuple[float, float]:
        self.pressed.clear()
        return 0.0, 0.0


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


def _command_for_key(key: str | None, linear: float, angular: float) -> tuple[float, float]:
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


def _command_for_pressed_keys(keys: set[str], linear: float, angular: float) -> tuple[float, float]:
    pressed = {key.lower() for key in keys}
    if not pressed or " " in pressed or "k" in pressed:
        return 0.0, 0.0

    if "j" in pressed:
        return linear, angular
    if "l" in pressed:
        return linear, -angular
    if "i" in pressed:
        return linear, 0.0

    linear_cmd = 0.0
    if "w" in pressed and "s" not in pressed:
        linear_cmd = linear
    elif "s" in pressed and "w" not in pressed:
        linear_cmd = -linear

    angular_cmd = 0.0
    if "a" in pressed and "d" not in pressed:
        angular_cmd = angular
    elif "d" in pressed and "a" not in pressed:
        angular_cmd = -angular

    return linear_cmd, angular_cmd


def _normalize_tk_key(event) -> str | None:
    if getattr(event, "char", "") == " ":
        return " "
    char = getattr(event, "char", "")
    if char and len(char) == 1 and char.isprintable():
        return char.lower()
    keysym = getattr(event, "keysym", "")
    if keysym == "space":
        return " "
    return keysym.lower() if keysym else None


def run_terminal_keyboard(*, linear: float, angular: float, rate_hz: float, topic: str) -> int:
    adapter = GazeboCmdVelAdapter(topic=topic)
    period = 1.0 / rate_hz
    old_settings = termios.tcgetattr(sys.stdin)
    print(TERMINAL_HELP)
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


def run_tk_keyboard(
    *,
    linear: float,
    angular: float,
    rate_hz: float,
    topic: str,
    release_delay_ms: int = 0,
) -> int:
    try:
        import tkinter as tk
    except ModuleNotFoundError as exc:
        raise SystemExit("Tkinter is required for key-release control. Use --backend terminal as a fallback.") from exc

    adapter = GazeboCmdVelAdapter(topic=topic)
    root = tk.Tk()
    root.title("Gazebo Keyboard Control")
    root.geometry("520x230")
    root.minsize(460, 200)

    key_state = _KeyPressState(linear=linear, angular=angular)
    release_jobs: dict[str, str] = {}
    last_command: tuple[float, float] | None = None
    running = True

    text = tk.Label(root, text=TK_HELP, justify="left", anchor="w", padx=14, pady=12)
    text.pack(fill="x")
    status = tk.Label(root, text="cmd=(+0.000,+0.000)  focus this window", anchor="w", padx=14, pady=8)
    status.pack(fill="x")

    def current_command() -> tuple[float, float]:
        return key_state.command()

    def publish(command: tuple[float, float], *, force: bool = False) -> None:
        nonlocal last_command
        if not force and command == last_command:
            return
        try:
            adapter.set_velocity(*command)
            last_command = command
            status.configure(text=f"cmd=({command[0]:+.3f},{command[1]:+.3f})  pressed={''.join(sorted(key_state.pressed)) or '-'}")
        except RuntimeError as exc:
            key_state.stop()
            status.configure(text=str(exc))
            last_command = None

    def remove_key(key: str) -> None:
        release_jobs.pop(key, None)
        command = key_state.release(key)
        if command is not None:
            publish(command, force=True)

    def on_key_press(event) -> str:
        key = _normalize_tk_key(event)
        if key == "q":
            close()
            return "break"
        if key in {" ", "k"}:
            for job in release_jobs.values():
                root.after_cancel(job)
            release_jobs.clear()
            publish(key_state.stop(), force=True)
            return "break"
        if key in _MOVEMENT_KEYS:
            job = release_jobs.pop(key, None)
            if job is not None:
                root.after_cancel(job)
            command = key_state.press(key)
            if command is not None:
                publish(command, force=True)
        return "break"

    def on_key_release(event) -> str:
        key = _normalize_tk_key(event)
        if key in _MOVEMENT_KEYS:
            job = release_jobs.pop(key, None)
            if job is not None:
                root.after_cancel(job)
            delay_ms = max(0, int(release_delay_ms))
            if delay_ms == 0:
                remove_key(key)
            else:
                release_jobs[key] = root.after(delay_ms, remove_key, key)
        return "break"

    def heartbeat() -> None:
        if not running:
            return
        publish(current_command())
        root.after(max(1, int(1000 / rate_hz)), heartbeat)

    def close() -> None:
        nonlocal running
        running = False
        key_state.stop()
        try:
            adapter.stop()
        except Exception:
            pass
        root.destroy()

    print(TK_HELP)
    root.bind("<KeyPress>", on_key_press)
    root.bind("<KeyRelease>", on_key_release)
    root.protocol("WM_DELETE_WINDOW", close)
    root.after(0, heartbeat)
    root.after(100, root.focus_force)
    root.mainloop()
    return 0


def run_keyboard(
    *,
    linear: float,
    angular: float,
    rate_hz: float,
    topic: str,
    backend: str = "tk",
    release_delay_ms: int = 0,
) -> int:
    if rate_hz <= 0:
        raise ValueError("rate_hz must be positive")
    if backend == "terminal":
        return run_terminal_keyboard(linear=linear, angular=angular, rate_hz=rate_hz, topic=topic)
    return run_tk_keyboard(
        linear=linear,
        angular=angular,
        rate_hz=rate_hz,
        topic=topic,
        release_delay_ms=release_delay_ms,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--linear", type=float, default=0.10, help="linear speed in m/s")
    parser.add_argument("--angular", type=float, default=0.30, help="angular speed in rad/s")
    parser.add_argument("--rate-hz", type=float, default=20.0, help="publish rate")
    parser.add_argument("--topic", default="/cmd_vel", help="Gazebo cmd_vel topic")
    parser.add_argument("--backend", choices=("tk", "terminal"), default="tk", help="keyboard input backend")
    parser.add_argument("--release-delay-ms", type=int, default=0, help="debounce delay for Tk key-release events")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    raise SystemExit(
        run_keyboard(
            linear=args.linear,
            angular=args.angular,
            rate_hz=args.rate_hz,
            topic=args.topic,
            backend=args.backend,
            release_delay_ms=args.release_delay_ms,
        )
    )


if __name__ == "__main__":
    main()
