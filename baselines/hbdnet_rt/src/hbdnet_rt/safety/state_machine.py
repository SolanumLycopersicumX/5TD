"""
安全状态机。
根据感知置信度、风险等级、障碍距离和 DWA 可行性，
决定车辆最高允许行驶状态。DWA 输出必须经此模块修正。
"""
from enum import IntEnum
from typing import Dict, Optional, Tuple


class SafetyState(IntEnum):
    """安全状态 (数值越大越严格, 便于比较)。"""
    S0_NORMAL = 0
    S1_CAUTIOUS = 1
    S2_SLOWDOWN = 2
    S3_STOP = 3
    S4_MANUAL_TAKEOVER = 4


class SafetyStateMachine:
    """
    固定场景安全状态机。

    输入因素:
      - overall_confidence: 感知综合置信度
      - max_risk: risk_grid 最大风险值
      - boundary_distance: 距硬边界最近距离 (m)
      - worker_distance: 距最近工人的距离 (m)
      - has_feasible_path: DWA 是否有可行轨迹

    控制约束:
      S0_NORMAL → 正常速度, steering 不受限
      S1_CAUTIOUS → 限速 50%, 禁止大角度绕行
      S2_SLOWDOWN → 限速 25%, 仅直行微调
      S3_STOP → 速度强制为 0, brake=True
      S4_MANUAL_TAKEOVER → 速度强制为 0, brake=True, 声光报警
    """

    def __init__(self, config):
        safety_cfg = config.safety.get("safety", {})
        self.conf = safety_cfg.get("confidence", {})
        self.risk_cfg = safety_cfg.get("risk", {})
        self.dist = safety_cfg.get("distance", {})
        self.speed_limits = safety_cfg.get("speed_limits", {})
        self.recovery_frames = safety_cfg.get("recovery_frames", 5)
        self.takeover_stop_frames = safety_cfg.get("consecutive_stop_for_takeover", 10)
        self.takeover_anomaly_frames = safety_cfg.get("consecutive_anomaly_for_takeover", 30)

        # 阈值
        self.conf_normal_min = self.conf.get("normal_min", 0.5)
        self.conf_cautious_min = self.conf.get("cautious_min", 0.3)
        self.conf_slowdown_min = self.conf.get("slowdown_min", 0.1)
        self.risk_cautious = self.risk_cfg.get("cautious_threshold", 0.3)
        self.risk_slowdown = self.risk_cfg.get("slowdown_threshold", 0.6)
        self.risk_stop = self.risk_cfg.get("stop_threshold", 0.9)
        self.bound_cautious_m = self.dist.get("hard_boundary_cautious_m", 1.0)
        self.worker_stop_m = self.dist.get("worker_stop_m", 2.0)
        self.worker_emergency_m = self.dist.get("worker_emergency_m", 1.0)

        # 状态
        self._state = SafetyState.S0_NORMAL
        self._stop_counter = 0
        self._anomaly_counter = 0
        self._recovery_counter = 0
        self._prev_target = SafetyState.S0_NORMAL
        self._transition_reason = ""

    # ── 属性 ──

    @property
    def state(self) -> SafetyState:
        return self._state

    @property
    def state_name(self) -> str:
        return self._state.name

    # ── 主入口 ──

    def evaluate(self, inputs: Dict) -> Dict:
        """
        评估安全状态并输出控制约束。
        输入来自感知和 DWA 的输出汇总。

        输入: {overall_confidence, max_risk, boundary_distance,
                worker_distance, has_feasible_path}
        输出: {safety_state, speed_limit_ratio, brake, reason}
        """
        conf = inputs.get("overall_confidence", 0.85)
        max_risk = inputs.get("max_risk", 0.0)
        bound_dist = inputs.get("boundary_distance", 99.0)
        worker_dist = inputs.get("worker_distance", 99.0)
        has_path = inputs.get("has_feasible_path", True)

        # ── 确定目标状态 ──
        target, reason = self._determine_target(
            conf, max_risk, bound_dist, worker_dist, has_path)

        # ── 状态转换 ──
        self._transition(target, conf)

        # ── 输出 ──
        speed_limit = float(self.speed_limits.get(self._state.name, 1.0))
        return {
            "safety_state": self._state.name,
            "speed_limit_ratio": speed_limit,
            "brake": self._state >= SafetyState.S3_STOP,
            "reason": self._transition_reason,
        }

    def apply_to_dwa(self, dwa_output: Dict, safety_output: Dict) -> Dict:
        """
        将 DWA 输出经安全状态机修正, 生成最终控制命令。

        核心规则:
          - STOP / MANUAL_TAKEOVER → speed 强制为 0
          - CAUTIOUS / SLOWDOWN → speed 乘上限系数
          - NORMAL → 保留 DWA 输出
          - 所有状态下保留 DWA 的 steering (即使在 CAUTIOUS, 微调仍有效)
        """
        state_name = safety_output["safety_state"]
        speed_limit = safety_output["speed_limit_ratio"]

        target_speed = dwa_output.get("target_speed", 0.0)
        target_steering = dwa_output.get("target_steering", 0.0)


        # STOP / TAKEOVER: 强制归零
        if state_name in ("S3_STOP", "S4_MANUAL_TAKEOVER"):
            target_speed = 0.0
            target_steering = 0.0  # 停止时方向盘回正
            brake = True
        else:
            target_speed = target_speed * speed_limit
            brake = safety_output.get("brake", False)

        return {
            "target_speed": target_speed,
            "target_steering": target_steering,
            "brake": brake,
            "safety_state": state_name,
            "reason": safety_output["reason"],
        }

    # ── 内部: 目标状态判断 ──

    def _determine_target(self, conf: float, max_risk: float,
                          bound_dist: float, worker_dist: float,
                          has_path: bool) -> Tuple[SafetyState, str]:
        """根据当前感知数据确定应该进入哪个安全状态。"""
        # 紧急: 工人极近或硬边界极近
        if worker_dist < self.worker_emergency_m or bound_dist < 0.3:
            return SafetyState.S3_STOP, (
                f"EMERGENCY: worker={worker_dist:.1f}m, boundary={bound_dist:.1f}m")

        # STOP 条件
        if conf < self.conf_slowdown_min:
            return SafetyState.S3_STOP, f"STOP: confidence too low ({conf:.2f})"
        if max_risk >= self.risk_stop:
            return SafetyState.S3_STOP, f"STOP: max_risk={max_risk:.2f}"
        if not has_path:
            return SafetyState.S3_STOP, "STOP: no feasible path"
        if worker_dist < self.worker_stop_m:
            return SafetyState.S3_STOP, f"STOP: worker too close ({worker_dist:.1f}m)"

        # SLOWDOWN 条件
        if conf < self.conf_cautious_min:
            return SafetyState.S2_SLOWDOWN, f"SLOWDOWN: confidence={conf:.2f}"
        if max_risk >= self.risk_slowdown:
            return SafetyState.S2_SLOWDOWN, f"SLOWDOWN: max_risk={max_risk:.2f}"

        # CAUTIOUS 条件
        if conf < self.conf_normal_min:
            return SafetyState.S1_CAUTIOUS, f"CAUTIOUS: confidence={conf:.2f}"
        if max_risk >= self.risk_cautious:
            return SafetyState.S1_CAUTIOUS, f"CAUTIOUS: max_risk={max_risk:.2f}"
        if bound_dist < self.bound_cautious_m:
            return SafetyState.S1_CAUTIOUS, f"CAUTIOUS: near hard boundary ({bound_dist:.1f}m)"

        # 正常
        return SafetyState.S0_NORMAL, "NORMAL: all clear"

    # ── 内部: 状态转换 ──

    def _transition(self, target: SafetyState, conf: float):
        """状态转换逻辑: 升级立即, 降级需连续确认。"""
        self._transition_reason = ""

        if target > self._state:
            # 安全升级: 立即生效
            self._transition_reason = f"UP: {self._state.name} → {target.name}"
            self._state = target
            self._recovery_counter = 0
        elif target < self._state:
            # 恢复: 需连续 N 帧确认
            if target == self._prev_target:
                self._recovery_counter += 1
            else:
                self._recovery_counter = 1
            if self._recovery_counter >= self.recovery_frames:
                self._transition_reason = (
                    f"DOWN: {self._state.name} → {target.name} "
                    f"(recovery={self._recovery_counter}/{self.recovery_frames})")
                self._state = target
                self._recovery_counter = 0
            else:
                self._transition_reason = (
                    f"PENDING: {self._state.name} → {target.name} "
                    f"({self._recovery_counter}/{self.recovery_frames})")
        else:
            self._recovery_counter = 0

        self._prev_target = target

        # 计数器
        if self._state >= SafetyState.S3_STOP:
            self._stop_counter += 1
        else:
            self._stop_counter = 0

        if conf < self.conf_cautious_min:
            self._anomaly_counter += 1
        else:
            self._anomaly_counter = 0

        # 触发接管
        if self._stop_counter >= self.takeover_stop_frames:
            self._state = SafetyState.S4_MANUAL_TAKEOVER
            self._transition_reason += (
                f"; TAKEOVER: stop_counter={self._stop_counter}")
        elif self._anomaly_counter >= self.takeover_anomaly_frames:
            self._state = SafetyState.S4_MANUAL_TAKEOVER
            self._transition_reason += (
                f"; TAKEOVER: anomaly_counter={self._anomaly_counter}")

    def reset(self):
        """重置状态 (用于测试或紧急恢复后)。"""
        self._state = SafetyState.S0_NORMAL
        self._stop_counter = 0
        self._anomaly_counter = 0
        self._recovery_counter = 0
        self._prev_target = SafetyState.S0_NORMAL
