"""RS232 / Modbus RTU dry-run conversion helpers."""
from __future__ import annotations

from dataclasses import dataclass

from .motion import MotionCommand, NavigationConfig

REG_LINEAR_VEL = 1040
REG_ANGULAR_VEL = 1041
REG_FUNC_CTRL = 1045
REG_DRV_ENABLE = 1049


def modbus_crc16(data: bytes) -> int:
    """Return Modbus RTU CRC16 as an integer."""
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
    return crc


def clamp_int16(value: int) -> int:
    """Clamp integer value to signed int16 range."""
    return max(-32768, min(32767, int(value)))


def velocity_to_registers(
    *,
    linear_mps: float,
    angular_radps: float,
    max_speed_mps: float,
    max_angular_radps: float,
    angular_sign: int,
) -> tuple[int, int]:
    """Convert physical velocity commands into VCU register values."""
    linear = max(-max_speed_mps, min(max_speed_mps, float(linear_mps)))
    angular = max(-max_angular_radps, min(max_angular_radps, float(angular_radps)))
    return (
        clamp_int16(round(linear * 1000)),
        clamp_int16(round(angular * int(angular_sign) * 1000)),
    )


@dataclass(frozen=True)
class Rs232DryRunAdapter:
    """Report intended RS232 register writes without opening a serial port."""

    node_addr: int = 0x06

    def send(self, command: MotionCommand, config: NavigationConfig) -> dict:
        linear_mps = 0.0 if command.brake else command.linear_mps
        angular_radps = 0.0 if command.brake else command.angular_radps
        linear, angular = velocity_to_registers(
            linear_mps=linear_mps,
            angular_radps=angular_radps,
            max_speed_mps=config.max_speed_mps,
            max_angular_radps=config.max_angular_radps,
            angular_sign=config.angular_sign,
        )
        return {
            "dry_run": True,
            "node_addr": self.node_addr,
            "register_start": REG_LINEAR_VEL,
            "registers": {
                "linear": linear,
                "angular": angular,
            },
            "source_frame": command.source_frame,
            "safety_state": command.safety_state,
            "reason": command.reason,
        }
