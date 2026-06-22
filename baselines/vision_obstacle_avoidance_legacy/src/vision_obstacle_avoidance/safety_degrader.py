"""
安全降级系统。
实现五级降级链 + Fail-Safe 机制，保证系统在任意模块失效时向安全状态退化。

设计参考:
  ISO 26262 Fail-Operational 架构
  Bosch 2024 动态安全架构重构
  ISO 21448 SOTIF (预期功能安全)

降级链:
  L0 正常    → 完整避障, ≤30 km/h
  L1 谨慎    → 限速10km/h, 安全间距2×
  L2 保守    → 限速5km/h, 仅直行
  L3 紧急    → 停车, 等待恢复
  L4 接管    → 物理抱死 + 报警, 需人工介入

v2.1 修复:
  - 迟滞效应: 恢复阈值高于降级阈值，需连续 RECOVERY_FRAMES 帧确认
  - L3 计时器: 只在 L0 恢复时重置，防止 L2/L3 间震荡导致超时永不触发
"""

import time
from enum import Enum


class DegradationLevel(Enum):
    """降级等级"""
    L0_NORMAL = 0       # 正常行驶
    L1_CAUTION = 1      # 谨慎: 限速, 增大安全间距
    L2_CONSERVATIVE = 2 # 保守: 大幅限速, 仅直行
    L3_EMERGENCY = 3    # 紧急: 停车等待
    L4_TAKEOVER = 4     # 接管: 物理抱死, 报警


class SafetyDegrader:
    """
    安全降级状态机。

    v2.1 迟滞机制:
      - 降级: dl_confidence < 降级阈值 或 连续异常帧 ≥ 阈值 → 立即降级
      - 恢复: dl_confidence ≥ 恢复阈值(高于降级阈值) 且连续 RECOVERY_FRAMES 帧 → 逐级恢复
      - 防止在阈值边界震荡导致的"急刹↔全速"抽搐

    输入:
      - HybridNets 置信度 (0-1)
      - 曝光状态 (ExposureState)
      - 连续异常帧数
      - 模块健康状态 (camera, inference, control)

    输出:
      - 当前降级等级
      - 速度上限 (km/h)
      - 安全间距乘数
      - 是否允许绕行
    """

    def __init__(self):
        # ── 降级阈值 (进入阈值，较低) ──
        self.l1_confidence = 0.50
        self.l2_confidence = 0.30
        self.l3_confidence = 0.10

        # ── 恢复阈值 (退出阈值，较高，实现迟滞) ──
        self.l1_recovery_confidence = 0.70
        self.l2_recovery_confidence = 0.50
        self.l3_recovery_confidence = 0.30

        # ── 连续异常帧阈值 ──
        self.l1_abnormal_frames = 5   # 连续5帧异常 → L1
        self.l2_abnormal_frames = 15  # 连续15帧异常 → L2
        self.l3_abnormal_frames = 30  # 连续30帧异常 → L3

        # ── 恢复确认帧数 ──
        self.recovery_frames = 10     # 需连续满足恢复阈值10帧才降回低级别

        # ── L3 超时 ──
        self.l3_timeout_sec = 10.0    # L3 持续10秒 → L4

        # ── 速度上限 (km/h) ──
        self.speed_limits = {
            DegradationLevel.L0_NORMAL: 30.0,
            DegradationLevel.L1_CAUTION: 10.0,
            DegradationLevel.L2_CONSERVATIVE: 5.0,
            DegradationLevel.L3_EMERGENCY: 0.0,
            DegradationLevel.L4_TAKEOVER: 0.0,
        }

        # ── 安全间距乘数 ──
        self.clearance_multipliers = {
            DegradationLevel.L0_NORMAL: 1.0,
            DegradationLevel.L1_CAUTION: 2.0,
            DegradationLevel.L2_CONSERVATIVE: 3.0,
            DegradationLevel.L3_EMERGENCY: 999.0,
            DegradationLevel.L4_TAKEOVER: 999.0,
        }

        # ── 状态 ──
        self._level = DegradationLevel.L0_NORMAL
        self._abnormal_frame_count: int = 0
        self._recovery_frame_count: int = 0  # v2.1: 恢复计数器
        self._l3_entry_time: float | None = None

        # 模块健康
        self._camera_ok: bool = True
        self._inference_ok: bool = True
        self._control_ok: bool = True

    # ── 公开接口 ──────────────────────────────────────────────────────────

    def evaluate(self,
                 dl_confidence: float,
                 exposure_state,
                 cv_lane_valid: bool = True) -> DegradationLevel:
        """
        评估当前降级等级。v2.1 支持迟滞恢复。

        参数:
          dl_confidence: HybridNets 总体置信度 (0-1)
          exposure_state: ExposureState 枚举
          cv_lane_valid: 传统CV车道检测是否有效（回退用）

        返回:
          当前 DegradationLevel
        """
        # ── 模块健康: 硬件故障直接 L4 ──
        if not self._camera_ok or not self._control_ok:
            self._level = DegradationLevel.L4_TAKEOVER
            self._recovery_frame_count = 0
            return self._level

        if not self._inference_ok:
            if cv_lane_valid:
                self._level = DegradationLevel.L2_CONSERVATIVE
            else:
                self._level = DegradationLevel.L4_TAKEOVER
            self._recovery_frame_count = 0
            return self._level

        # ── 曝光状态: 直接降级（不受迟滞影响） ──
        from exposure_detector import ExposureState
        if exposure_state == ExposureState.AE_ADJUSTING:
            self._level = DegradationLevel.L2_CONSERVATIVE
            self._abnormal_frame_count = 0
            self._recovery_frame_count = 0
            return self._level
        if exposure_state in (ExposureState.OVEREXPOSED, ExposureState.UNDEREXPOSED):
            self._level = DegradationLevel.L1_CAUTION
            self._abnormal_frame_count = 0
            self._recovery_frame_count = 0
            return self._level

        # ── 置信度评估 ────────────────────────────────────────────────
        is_abnormal = dl_confidence < self.l1_confidence

        if is_abnormal:
            self._abnormal_frame_count += 1
            self._recovery_frame_count = 0  # 打断恢复
        else:
            self._abnormal_frame_count = 0

        # 计算当前帧对应的"即时降级等级"
        instant_level = DegradationLevel.L0_NORMAL
        if dl_confidence < self.l1_confidence or self._abnormal_frame_count >= self.l1_abnormal_frames:
            instant_level = DegradationLevel.L1_CAUTION
        if dl_confidence < self.l2_confidence or self._abnormal_frame_count >= self.l2_abnormal_frames:
            instant_level = DegradationLevel.L2_CONSERVATIVE
        if dl_confidence < self.l3_confidence or self._abnormal_frame_count >= self.l3_abnormal_frames:
            instant_level = DegradationLevel.L3_EMERGENCY

        # ── 降级方向: 立即恶化 ──
        if instant_level.value > self._level.value:
            self._level = instant_level
            self._recovery_frame_count = 0
            if self._level == DegradationLevel.L3_EMERGENCY and self._l3_entry_time is None:
                self._l3_entry_time = time.time()
            return self._level

        # ── L3 超时 → L4 ──
        if self._level == DegradationLevel.L3_EMERGENCY:
            if self._l3_entry_time is not None and \
               time.time() - self._l3_entry_time > self.l3_timeout_sec:
                self._level = DegradationLevel.L4_TAKEOVER
                return self._level

        # ── 恢复方向: 迟滞 — 需要高于恢复阈值 + 连续 N 帧 ──
        if self._level != DegradationLevel.L0_NORMAL:
            recovery_threshold = self._get_recovery_threshold(self._level)
            if dl_confidence >= recovery_threshold:
                self._recovery_frame_count += 1
            else:
                self._recovery_frame_count = 0

            if self._recovery_frame_count >= self.recovery_frames:
                # 逐级恢复
                self._level = self._step_down_level(self._level)
                self._recovery_frame_count = 0
                # v2.1 修复: 只在完全恢复到 L0 时重置 L3 计时器
                if self._level == DegradationLevel.L0_NORMAL:
                    self._l3_entry_time = None
            return self._level

        # 正常 L0
        self._level = DegradationLevel.L0_NORMAL
        self._l3_entry_time = None  # 完全正常时重置
        return self._level

    def _get_recovery_threshold(self, level: DegradationLevel) -> float:
        """获取当前等级对应的恢复阈值（高于降级阈值）。"""
        mapping = {
            DegradationLevel.L1_CAUTION: self.l1_recovery_confidence,
            DegradationLevel.L2_CONSERVATIVE: self.l2_recovery_confidence,
            DegradationLevel.L3_EMERGENCY: self.l3_recovery_confidence,
        }
        return mapping.get(level, 0.70)

    @staticmethod
    def _step_down_level(level: DegradationLevel) -> DegradationLevel:
        """逐级恢复。"""
        step = {
            DegradationLevel.L1_CAUTION: DegradationLevel.L0_NORMAL,
            DegradationLevel.L2_CONSERVATIVE: DegradationLevel.L1_CAUTION,
            DegradationLevel.L3_EMERGENCY: DegradationLevel.L2_CONSERVATIVE,
        }
        return step.get(level, level)

    # ── 查询接口 ──────────────────────────────────────────────────────────

    @property
    def level(self) -> DegradationLevel:
        return self._level

    @property
    def speed_limit_kmh(self) -> float:
        return self.speed_limits[self._level]

    @property
    def clearance_multiplier(self) -> float:
        return self.clearance_multipliers[self._level]

    @property
    def allow_detour(self) -> bool:
        """是否允许绕行（L2+ 不允许）"""
        return self._level in (DegradationLevel.L0_NORMAL,
                               DegradationLevel.L1_CAUTION)

    @property
    def should_stop(self) -> bool:
        """是否应该停车"""
        return self._level in (DegradationLevel.L3_EMERGENCY,
                               DegradationLevel.L4_TAKEOVER)

    @property
    def needs_takeover(self) -> bool:
        """是否需要人工接管"""
        return self._level == DegradationLevel.L4_TAKEOVER

    # ── 模块健康上报 ──────────────────────────────────────────────────────

    def report_camera(self, ok: bool):
        self._camera_ok = ok

    def report_inference(self, ok: bool):
        self._inference_ok = ok

    def report_control(self, ok: bool):
        self._control_ok = ok

    # ── 手动控制 ──────────────────────────────────────────────────────────

    def force_level(self, level: DegradationLevel):
        """手动设置降级等级（调试/接管用）"""
        self._level = level
        self._abnormal_frame_count = 0
        self._recovery_frame_count = 0
        self._l3_entry_time = None

    def reset(self):
        """重置降级状态"""
        self._level = DegradationLevel.L0_NORMAL
        self._abnormal_frame_count = 0
        self._recovery_frame_count = 0
        self._l3_entry_time = None
        self._camera_ok = True
        self._inference_ok = True
        self._control_ok = True

    # ── 诊断 ──────────────────────────────────────────────────────────────

    def status_summary(self) -> str:
        """返回可读状态摘要"""
        parts = [
            f"Level: {self._level.name}",
            f"SpeedLimit: {self.speed_limit_kmh:.0f}km/h",
            f"AbnormalFrames: {self._abnormal_frame_count}",
            f"RecoveryFrames: {self._recovery_frame_count}/{self.recovery_frames}",
            f"Camera: {'OK' if self._camera_ok else 'FAIL'}",
            f"Inference: {'OK' if self._inference_ok else 'FAIL'}",
            f"Control: {'OK' if self._control_ok else 'FAIL'}",
        ]
        if self._l3_entry_time is not None:
            elapsed = time.time() - self._l3_entry_time
            parts.append(f"L3_elapsed: {elapsed:.1f}s")
        return " | ".join(parts)
