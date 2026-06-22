"""
决策模块。
基于可行驶区域 + 障碍状态 → 输出控制指令。
工程化规则引擎：空值保护 + 帧间限幅 + 置信度传导 + 线程安全 + 显式状态机。
"""
import math
import threading
from collections import deque
from enum import Enum

import config
from utils import Decision, FreeSpaceState, ObstacleState


class DriveState(Enum):
    """车辆行驶状态枚举。优先级: EMERGENCY > EVASIVE > CAUTION > NORMAL"""
    NORMAL    = "NORMAL"
    CAUTION   = "CAUTION"
    EVASIVE   = "EVASIVE"
    EMERGENCY = "EMERGENCY"

    def __lt__(self, other):
        order = {"NORMAL": 0, "CAUTION": 1, "EVASIVE": 2, "EMERGENCY": 3}
        return order[self.value] < order[other.value]


class DecisionMaker:
    """
    工程化避障决策器。

    决策优先级（由高到低）:
      1. 感知输入丢失        → EMERGENCY STOP
      2. 延迟/FPS 超标       → CAUTION   SLOW_DOWN
      3. 可行驶区域无效       → CAUTION   SLOW_DOWN / STOP
      4. 三区全阻塞          → EMERGENCY STOP
      5. 中心阻塞            → EVASIVE   侧向绕行
      6. 偏离中心            → NORMAL    微调方向
      7. 畅通                → NORMAL    直行

    低置信度决策自动向上一级状态升格（保守策略）。
    """

    def __init__(self):
        self._last_command = "FORWARD"
        self._last_steer = 0.0
        self._cmd_queue = deque(maxlen=config.DECISION_CONSECUTIVE_FRAMES)
        self._lock = threading.Lock()

    # ---- 公开接口 ----

    def decide(self, free_state: FreeSpaceState, obstacle_state: ObstacleState,
               latency_ms: float, fps: float,
               debris_state=None, path_plan=None, lane_boundary=None,
               degradation_status=None) -> Decision:
        """
        线程安全入口。

        新增可选参数:
          degradation_status: DegradationStatus (安全降级状态)
        """
        with self._lock:
            return self._decide_impl(free_state, obstacle_state, latency_ms, fps,
                                     debris_state, path_plan, lane_boundary,
                                     degradation_status)

    # ---- 内部实现 ----

    def _decide_impl(self, free_state, obstacle_state, latency_ms, fps,
                      debris_state=None, path_plan=None, lane_boundary=None,
                      degradation_status=None):
        # ================================================================
        # 规则 -2: 安全降级触发 —— 最高优先级
        # ================================================================
        if degradation_status is not None:
            if degradation_status.needs_takeover:
                return self._make_decision(
                    "STOP", 0.0, 0.0,
                    f"TAKEOVER: {degradation_status.summary}",
                    DriveState.EMERGENCY, 1.0)
            if degradation_status.should_stop:
                return self._make_decision(
                    "STOP", 0.0, 0.0,
                    f"EMERGENCY: level={degradation_status.level_name}",
                    DriveState.EMERGENCY, 0.95)
            if not degradation_status.allow_detour:
                # L2: 限速 + 仅直行
                speed = min(config.SLOW_SPEED,
                           degradation_status.speed_limit_kmh / 30.0 * config.DEFAULT_SPEED)
                return self._make_decision(
                    "SLOW_DOWN", speed, 0.0,
                    f"Conservative: {degradation_status.summary}",
                    DriveState.CAUTION, 0.6)

        # ================================================================
        # 规则 -1: 空值保护 —— 感知模块崩溃时安全停车
        # ================================================================
        if free_state is None or obstacle_state is None:
            return self._make_decision("STOP", 0.0, 0.0, "Perception input lost",
                                       DriveState.EMERGENCY, 1.0)

        # ================================================================
        # 规则 0: 车道不可通行 → STOP (Phase 1 新增 #33)
        # ================================================================
        if path_plan is not None and not path_plan.passable:
            return self._make_decision("STOP", 0.0, 0.0,
                                       f"Lane impassable: {path_plan.blocked_reason}",
                                       DriveState.EMERGENCY, 0.95)

        # ================================================================
        # 规则 0.5: 车道边界违规检查 —— 检测到即将跨越隔离沟 (#33)
        # ================================================================
        if lane_boundary is not None and lane_boundary.is_valid:
            # 检查自由空间评分是否偏向隔离沟对面
            if self._is_crossing_ditch(free_state, lane_boundary):
                return self._make_decision("STOP", 0.0, 0.0,
                                           "Ditch crossing prevented",
                                           DriveState.EMERGENCY, 0.99)

        # ================================================================
        # 规则 1: 延迟/FPS 超标 → SLOW_DOWN（动态比例减速）
        # ================================================================
        excess_lat = min(1.0, max(0.0, latency_ms - config.MAX_LATENCY_MS) / config.MAX_LATENCY_MS)
        excess_fps = min(1.0, max(0.0, config.TARGET_FPS - fps) / config.TARGET_FPS) if fps > 0 else 1.0
        excess = max(excess_lat, excess_fps)
        if excess > 0:
            speed = config.DEFAULT_SPEED - excess * (config.DEFAULT_SPEED - config.SLOW_SPEED)
            return self._make_decision("SLOW_DOWN", speed, 0.0,
                                       f"Perf degraded: lat={latency_ms:.0f}ms fps={fps:.1f}",
                                       DriveState.CAUTION, 0.8)

        # ================================================================
        # 规则 1: 可行驶区域无效 → SLOW_DOWN / STOP
        # ================================================================
        if not free_state.is_valid:
            if self._last_command == "STOP":
                return self._make_decision("STOP", 0.0, 0.0, "Free space persistently invalid",
                                           DriveState.EMERGENCY, 0.95)
            return self._make_decision("SLOW_DOWN", config.SLOW_SPEED, 0.0,
                                       "Free space invalid",
                                       DriveState.CAUTION, max(0.3, free_state.confidence))

        # ================================================================
        # 规则 2: 三区全阻塞 → STOP (现有)
        # ================================================================
        if obstacle_state.blocked_left and obstacle_state.blocked_center and obstacle_state.blocked_right:
            return self._make_decision("STOP", 0.0, 0.0, "All zones blocked",
                                       DriveState.EMERGENCY, 0.95)

        # ================================================================
        # 规则 3: 中心阻塞 → 动态绕行（危险度 × 通行优势 → 比例转向）
        # ================================================================
        if obstacle_state.blocked_center and obstacle_state.danger_level > 0.4:
            direction, advantage = self._pick_detour_direction(free_state, obstacle_state)
            if direction is None:
                return self._make_decision("STOP", 0.0, 0.0, "Center blocked, no detour",
                                           DriveState.EMERGENCY, 0.9)
            urgency = obstacle_state.danger_level * advantage
            steer = config.STEERING_SMALL + urgency * (config.STEERING_LARGE - config.STEERING_SMALL)
            steer = max(config.STEERING_SMALL, min(config.STEERING_LARGE, steer))
            if direction == "TURN_LEFT":
                steer = -steer
            speed = config.DEFAULT_SPEED - obstacle_state.danger_level * (config.DEFAULT_SPEED - config.SLOW_SPEED)
            label = "LEFT" if direction == "TURN_LEFT" else "RIGHT"
            return self._make_decision(direction, speed, steer,
                                       f"Center blocked -> {label} detour (urgency={urgency:.2f})",
                                       DriveState.EVASIVE, 0.85)

        # ================================================================
        # 规则 4: 偏离可行驶区域中心 → tanh 平滑微调
        # ================================================================
        offset = free_state.center_offset
        if abs(offset) > config.CENTER_OFFSET_THRESHOLD:
            steer = self._smooth_steering(offset)
            if abs(steer) > 0.05:
                return self._make_decision("FORWARD", config.DEFAULT_SPEED, steer,
                                           f"Off-center, correcting: offset={offset:+.2f}",
                                           DriveState.NORMAL, 0.8)

        # ================================================================
        # 规则 5: 正常前进
        # ================================================================
        return self._make_decision("FORWARD", config.DEFAULT_SPEED, 0.0, "Center clear",
                                   DriveState.NORMAL, 0.9)

    # ---- 内部辅助 ----

    @staticmethod
    def _is_crossing_ditch(free_state, lane_boundary) -> bool:
        """
        v2.1: 动态判断隔离沟位置，消除"沟永远在右侧"的硬编码。

        检测车辆是否正在/即将跨越隔离沟。
        通过 lane_boundary 中 ditch_px 与两侧隔离带的相对位置
        动态判断隔离沟在车辆左侧还是右侧。
        """
        if free_state is None or lane_boundary is None:
            return False
        if not lane_boundary.is_valid:
            return False

        offset = free_state.center_offset
        if abs(offset) <= config.CENTER_OFFSET_THRESHOLD * 2:
            return False

        # v2.1: 动态判断隔离沟位置
        ditch_on_right = True  # 默认
        if lane_boundary.ditch_px is not None:
            if lane_boundary.left_barrier_px is not None and \
               lane_boundary.right_barrier_px is not None:
                # 沟在右: left_barrier < ditch < right_barrier 不成立
                # 沟在左: 沟的 x 坐标在左侧隔离带附近
                ditch_on_right = lane_boundary.ditch_px > lane_boundary.right_barrier_px * 0.8
            elif lane_boundary.left_barrier_px is not None:
                ditch_on_right = lane_boundary.ditch_px > lane_boundary.left_barrier_px

        # offset > 0 = 右侧更畅通（偏左），offset < 0 = 左侧更畅通（偏右）
        if ditch_on_right and offset > config.CENTER_OFFSET_THRESHOLD * 2:
            # 沟在右，偏向往右 → 危险
            if free_state.right_free_score > free_state.center_free_score + 0.3:
                return True
        elif not ditch_on_right and offset < -config.CENTER_OFFSET_THRESHOLD * 2:
            # 沟在左，偏向往左 → 危险
            if free_state.left_free_score > free_state.center_free_score + 0.3:
                return True

        return False

    @staticmethod
    def _pick_detour_direction(free_state, obstacle_state):
        """选择绕行方向。返回 (direction, advantage) 或 (None, 0)。"""
        left_score = free_state.left_free_score
        right_score = free_state.right_free_score
        if not obstacle_state.blocked_left and left_score > right_score + 0.05:
            return "TURN_LEFT", left_score - right_score
        if not obstacle_state.blocked_right and right_score > left_score + 0.05:
            return "TURN_RIGHT", right_score - left_score
        return None, 0.0

    @staticmethod
    def _smooth_steering(offset):
        """tanh 平滑转向映射。"""
        steer = math.tanh(offset * config.STEERING_TANH_GAIN) * config.STEERING_TANH_MAX
        return max(-1.0, min(1.0, steer))

    def _make_decision(self, command, speed, steer, reason, drive_state, confidence):
        """
        统一构造 Decision，同时应用帧间转向限幅和置信度衰减。
        低置信度 → 自动升格为更保守的状态并降速。
        """
        # 帧间转向限幅
        steer = self._clamp_steer_rate(steer)

        # 置信度传导：低置信度 → 降速 + 状态升格
        if confidence < 0.5 and drive_state == DriveState.NORMAL:
            drive_state = DriveState.CAUTION
            speed = max(config.SLOW_SPEED, speed * 0.7)
        elif confidence < 0.3:
            drive_state = DriveState.EVASIVE
            speed = max(config.SLOW_SPEED, speed * 0.5)

        decision = Decision(
            command=command, speed=speed, steering=steer,
            reason=reason, confidence=confidence,
            drive_state=drive_state.value,
        )
        self._last_steer = steer
        # EMERGENCY 命令跳过防抖，立即生效
        if drive_state == DriveState.EMERGENCY:
            self._last_command = command
            self._cmd_queue.clear()
            return decision
        return self._apply_hysteresis(decision)

    def _clamp_steer_rate(self, target_steer):
        """帧间转向变化率限制，防止突变。"""
        delta = target_steer - self._last_steer
        delta = max(-config.STEERING_MAX_RATE, min(config.STEERING_MAX_RATE, delta))
        return self._last_steer + delta

    def _apply_hysteresis(self, decision):
        """
        连续帧确认防抖：
        - 同命令 → 直接返回
        - 新命令 → 需连续 DECISION_CONSECUTIVE_FRAMES 帧才切换
        - 切换期间沿用旧命令，steering 可微调（避免转向滞后）
        """
        if decision.command == self._last_command:
            self._cmd_queue.clear()
            return decision

        self._cmd_queue.append(decision.command)
        if len(self._cmd_queue) == self._cmd_queue.maxlen and \
           all(c == decision.command for c in self._cmd_queue):
            self._last_command = decision.command
            self._cmd_queue.clear()
            return decision

        # 待确认：保留旧命令但允许 steering 微调
        return Decision(
            command=self._last_command,
            speed=config.SLOW_SPEED if self._last_command != "FORWARD" else decision.speed,
            steering=decision.steering,
            reason=f"(pending) {decision.reason}",
            confidence=decision.confidence * 0.7,
            drive_state=decision.drive_state,
        )
