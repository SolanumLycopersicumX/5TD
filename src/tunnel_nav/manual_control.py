"""Manual-control shaping helpers for low-speed vehicle bring-up."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ManualControlLimits:
    """Safety limits for direct manual driving."""

    max_linear_mps: float = 0.04
    max_angular_radps: float = 0.12
    deadzone: float = 0.18
    response_exponent: float = 2.0
    max_linear_accel_mps2: float = 0.08
    max_angular_accel_radps2: float = 0.25
    min_turn_start_radps: float = 0.0


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def shape_axis(value: float, limits: ManualControlLimits) -> float:
    """Apply deadzone and response curve to a joystick axis in [-1, 1]."""
    axis = _clamp(value, -1.0, 1.0)
    deadzone = _clamp(limits.deadzone, 0.0, 0.95)
    magnitude = abs(axis)
    if magnitude <= deadzone:
        return 0.0

    normalized = (magnitude - deadzone) / (1.0 - deadzone)
    exponent = max(1.0, float(limits.response_exponent))
    shaped = normalized**exponent
    return shaped if axis > 0 else -shaped


def manual_command_from_axes(
    forward_axis: float,
    turn_axis: float,
    limits: ManualControlLimits,
) -> tuple[float, float]:
    """Convert joystick axes to physical linear/angular velocity commands."""
    shaped_forward = shape_axis(forward_axis, limits)
    shaped_turn = shape_axis(turn_axis, limits)
    linear = shaped_forward * max(0.0, float(limits.max_linear_mps))
    angular = shaped_turn * max(0.0, float(limits.max_angular_radps))

    min_turn = max(0.0, min(float(limits.min_turn_start_radps), float(limits.max_angular_radps)))
    if linear == 0.0 and angular != 0.0 and abs(angular) < min_turn:
        angular = min_turn if angular > 0 else -min_turn

    return (round(linear, 6), round(angular, 6))


def slew_towards(current: float, target: float, max_delta: float) -> float:
    """Move current toward target without overshooting the target."""
    current = float(current)
    target = float(target)
    step = abs(float(max_delta))
    if step == 0:
        return current

    delta = target - current
    if abs(delta) <= step:
        return round(target, 6)
    return round(current + step * (1.0 if delta > 0 else -1.0), 6)


def ramp_command(
    current: tuple[float, float],
    target: tuple[float, float],
    limits: ManualControlLimits,
    dt_s: float,
) -> tuple[float, float]:
    """Rate-limit linear and angular commands."""
    dt = max(0.0, float(dt_s))
    linear_step = max(0.0, float(limits.max_linear_accel_mps2)) * dt
    angular_step = max(0.0, float(limits.max_angular_accel_radps2)) * dt
    return (
        round(slew_towards(current[0], target[0], linear_step), 6),
        round(slew_towards(current[1], target[1], angular_step), 6),
    )
