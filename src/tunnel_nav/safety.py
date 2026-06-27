"""Safety filtering for DWA trajectory outputs."""
from __future__ import annotations

from .motion import MotionCommand, NavigationConfig, Trajectory

RISK_CAUTION_THRESHOLD = 0.35
RISK_SLOWDOWN_THRESHOLD = 0.65
RISK_STOP_THRESHOLD = 0.95


def _clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, float(value)))


def command_from_dwa_result(
    selected_trajectory: Trajectory | None,
    candidates: list[Trajectory],
    config: NavigationConfig,
    *,
    source_frame: str,
) -> MotionCommand:
    """Convert DWA output into a safety-filtered motion command."""
    del candidates
    if selected_trajectory is None or not selected_trajectory.feasible:
        return MotionCommand.stop_command(
            "no feasible DWA trajectory",
            source_frame=source_frame,
            dry_run=config.dry_run,
        )

    max_risk = float(selected_trajectory.max_risk)
    if max_risk >= RISK_STOP_THRESHOLD:
        return MotionCommand.stop_command(
            "risk stop threshold exceeded",
            source_frame=source_frame,
            dry_run=config.dry_run,
        )

    linear_mps = max(0.0, min(float(selected_trajectory.linear_mps), config.max_speed_mps))
    angular_radps = _clamp(selected_trajectory.angular_radps, config.max_angular_radps)
    brake = False
    if max_risk >= RISK_SLOWDOWN_THRESHOLD:
        safety_state = "S2_SLOWDOWN"
        linear_mps = min(linear_mps, config.max_speed_mps * 0.25)
        reason = "risk slowdown threshold exceeded"
    elif max_risk >= RISK_CAUTION_THRESHOLD:
        safety_state = "S1_CAUTIOUS"
        linear_mps = min(linear_mps, config.max_speed_mps * 0.50)
        reason = "risk caution threshold exceeded"
    else:
        safety_state = "S0_NORMAL"
        reason = "DWA trajectory accepted"

    return MotionCommand(
        linear_mps=linear_mps,
        angular_radps=angular_radps,
        brake=brake,
        safety_state=safety_state,
        reason=reason,
        confidence=max(0.0, min(1.0, 1.0 - max_risk)),
        source_frame=source_frame,
        dry_run=config.dry_run,
    )
