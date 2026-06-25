"""Gazebo command-velocity adapter with the same shape as the RS232 driver."""
from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

Runner = Callable[[Sequence[str]], object]


def _default_gz_command() -> str:
    """Return a usable Gazebo CLI path."""
    return shutil.which("gz") or "/opt/ros/jazzy/opt/gz_tools_vendor/bin/gz"


@dataclass
class GazeboCmdVelAdapter:
    """Publish driver-style velocity commands to Gazebo `/cmd_vel`.

    This mirrors the important motion methods of `1/driver_controller.py`, but
    sends `gz.msgs.Twist` messages instead of writing RS232 Modbus registers.
    """

    topic: str = "/cmd_vel"
    gz_command: str = field(default_factory=_default_gz_command)
    runner: Callable[..., object] = subprocess.run

    def set_velocity(self, linear_mps: float, angular_radps: float) -> None:
        """Set linear and angular velocity in physical units."""
        payload = self.twist_payload(linear_mps, angular_radps)
        self.runner(
            [
                self.gz_command,
                "topic",
                "-t",
                self.topic,
                "-m",
                "gz.msgs.Twist",
                "-p",
                payload,
            ],
            check=True,
        )

    @staticmethod
    def twist_payload(linear_mps: float, angular_radps: float) -> str:
        """Build Gazebo protobuf text for a Twist command."""
        return (
            f"linear: {{x: {float(linear_mps):.6f}}}, "
            f"angular: {{z: {float(angular_radps):.6f}}}"
        )

    def stop(self) -> None:
        self.set_velocity(0.0, 0.0)

    def forward(self, speed_mps: float = 0.10) -> None:
        self.set_velocity(speed_mps, 0.0)

    def backward(self, speed_mps: float = 0.10) -> None:
        self.set_velocity(-speed_mps, 0.0)

    def turn_left(self, angular_radps: float = 0.30) -> None:
        self.set_velocity(0.0, angular_radps)

    def turn_right(self, angular_radps: float = 0.30) -> None:
        self.set_velocity(0.0, -angular_radps)

    def arc_left(self, speed_mps: float = 0.10, angular_radps: float = 0.20) -> None:
        self.set_velocity(speed_mps, angular_radps)

    def arc_right(self, speed_mps: float = 0.10, angular_radps: float = 0.20) -> None:
        self.set_velocity(speed_mps, -angular_radps)
