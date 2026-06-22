"""控制命令格式。DWA 输出 + 安全状态机修正 → 最终控制指令。"""
import time
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class ControlCommand:
    """统一控制命令格式。"""
    timestamp: float = 0.0
    target_speed: float = 0.0
    target_steering: float = 0.0
    brake: bool = False
    safety_state: str = "S0_NORMAL"
    reason: str = ""
    debug: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "target_speed": self.target_speed,
            "target_steering": self.target_steering,
            "brake": self.brake,
            "safety_state": self.safety_state,
            "reason": self.reason,
            "debug": self.debug,
        }


def generate_command(dwa_output: Dict, safety_output: Dict) -> ControlCommand:
    """从 DWA 输出和安全状态机输出生成最终控制命令。"""
    cmd = ControlCommand()
    cmd.timestamp = time.time()
    cmd.target_speed = dwa_output.get("target_speed", 0.0)
    cmd.target_steering = dwa_output.get("target_steering", 0.0)
    cmd.safety_state = safety_output.get("safety_state", "S0_NORMAL")
    cmd.brake = safety_output.get("brake", False)
    cmd.reason = safety_output.get("reason", "")

    # 安全修正: STOP/TAKEOVER 强制速度为零
    if cmd.safety_state in ("S3_STOP", "S4_MANUAL_TAKEOVER"):
        cmd.target_speed = 0.0
        cmd.brake = True
    else:
        # 速度上限修正
        limit = safety_output.get("speed_limit_ratio", 1.0)
        cmd.target_speed *= limit

    return cmd
