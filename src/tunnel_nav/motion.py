"""Shared data structures for the BEV/DWA navigation bridge."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class NavigationConfig:
    """Conservative vehicle and bridge limits."""

    max_speed_mps: float = 0.10
    max_angular_radps: float = 0.50
    angular_sign: int = 1
    vehicle_width_m: float = 0.80
    safety_margin_m: float = 0.30
    dry_run: bool = True
    live_requires_explicit_flag: bool = True


@dataclass(frozen=True)
class DWAConfig:
    """Low-speed DWA sampling defaults for offline bring-up."""

    min_velocity_mps: float = 0.0
    max_velocity_mps: float = 0.10
    velocity_samples: int = 3
    max_angular_radps: float = 0.50
    angular_samples: int = 9
    predict_time_s: float = 2.0
    dt_s: float = 0.2


@dataclass(frozen=True)
class MaskBundle:
    """Fused binary masks for one source frame."""

    safe_passable: np.ndarray
    ditch: np.ndarray
    tunnel_wall: np.ndarray
    left_barrier: np.ndarray
    source_frame: str = ""

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.safe_passable.shape)


@dataclass(frozen=True)
class RiskGrid:
    """Continuous risk grid with local metric metadata."""

    risk: np.ndarray
    x_min_m: float
    x_max_m: float
    y_min_m: float
    y_max_m: float
    resolution_m: float

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.risk.shape)

    @property
    def width_m(self) -> float:
        return float(self.x_max_m - self.x_min_m)

    @property
    def length_m(self) -> float:
        return float(self.y_max_m - self.y_min_m)


@dataclass(frozen=True)
class BEVGrid(RiskGrid):
    """Binary occupancy plus continuous risk in local vehicle coordinates."""

    occupancy: np.ndarray


@dataclass(frozen=True)
class Trajectory:
    """One DWA trajectory candidate in local metric coordinates."""

    points: np.ndarray
    linear_mps: float
    angular_radps: float
    score: float
    max_risk: float
    min_clearance_m: float
    feasible: bool
    reason: str


@dataclass(frozen=True)
class MotionCommand:
    """Physical-unit command at the planner-driver boundary."""

    linear_mps: float
    angular_radps: float
    brake: bool
    safety_state: str
    reason: str
    confidence: float
    source_frame: str
    dry_run: bool = True

    @classmethod
    def stop_command(cls, reason: str, *, source_frame: str = "", dry_run: bool = True) -> "MotionCommand":
        return cls(
            linear_mps=0.0,
            angular_radps=0.0,
            brake=True,
            safety_state="S3_STOP",
            reason=reason,
            confidence=0.0,
            source_frame=source_frame,
            dry_run=dry_run,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "linear_mps": round(float(self.linear_mps), 4),
            "angular_radps": round(float(self.angular_radps), 4),
            "brake": bool(self.brake),
            "safety_state": self.safety_state,
            "reason": self.reason,
            "confidence": round(float(self.confidence), 4),
            "source_frame": self.source_frame,
            "dry_run": bool(self.dry_run),
        }
