"""
曝光状态感知模块 — v2.2 增强版。
检测当前帧的曝光状态，支持分区域分析（近/中/远），为安全降级系统提供输入。
纯像素统计方法，不依赖深度学习，计算量 ~1ms。
"""

import numpy as np
from enum import Enum
from dataclasses import dataclass


class ExposureState(Enum):
    """全局曝光状态枚举"""
    NORMAL = "normal"
    OVEREXPOSED = "overexposed"
    UNDEREXPOSED = "underexposed"
    AE_ADJUSTING = "ae_adjusting"


class ZoneExposure(Enum):
    """单区域曝光状态"""
    NORMAL = "normal"
    OVEREXPOSED = "overexposed"
    UNDEREXPOSED = "underexposed"


@dataclass
class ZonedExposureResult:
    """分区域曝光检测结果"""
    global_state: ExposureState         # 全局状态
    near: ZoneExposure                  # 近区 (ROI 下 1/3)
    mid: ZoneExposure                   # 中区 (ROI 中 1/3)
    far: ZoneExposure                   # 远区 (ROI 上 1/3)
    far_overexposed_ratio: float        # 远区过曝像素比例 (0~1)
    is_far_blind: bool                  # 远区是否严重过曝 (>50%)

    def __repr__(self):
        return (f"ZonedExposure(global={self.global_state.value}, "
                f"near={self.near.value}, mid={self.mid.value}, far={self.far.value}, "
                f"far_blind={self.is_far_blind}, far_ratio={self.far_overexposed_ratio:.2f})")


class ExposureStateDetector:
    """
    曝光状态感知器 — v2.2。

    新增 zoned analysis:
      - 将 ROI 水平切分为近/中/远三区
      - 隧道出口场景：远区可能过曝而近/中区正常
      - 远区过曝 → 决策层减速/停车，因为看不见远端障碍物

    算法:
      1. 全局统计过/欠曝像素比例
      2. 分区域统计（近/中/远各占 ROI 高度的 1/3）
      3. 区域曝光判定（阈值可独立配置，远区更敏感）
      4. 帧间亮度变化检测（AE 调整中）
      5. 连续帧确认防抖
    """

    def __init__(self,
                 # 全局阈值
                 overexposed_ratio: float = 0.30,
                 underexposed_ratio: float = 0.50,
                 ae_delta_threshold: float = 50.0,
                 overexposed_value: int = 250,
                 underexposed_value: int = 20,
                 # 分区域阈值（远区更敏感）
                 far_overexposed_ratio: float = 0.25,
                 far_blind_ratio: float = 0.50,
                 mid_overexposed_ratio: float = 0.35,
                 near_overexposed_ratio: float = 0.40):
        # 全局阈值
        self.overexposed_ratio = overexposed_ratio
        self.underexposed_ratio = underexposed_ratio
        self.ae_delta_threshold = ae_delta_threshold
        self.overexposed_value = overexposed_value
        self.underexposed_value = underexposed_value

        # 分区域阈值
        self.far_overexposed_ratio = far_overexposed_ratio
        self.far_blind_ratio = far_blind_ratio
        self.mid_overexposed_ratio = mid_overexposed_ratio
        self.near_overexposed_ratio = near_overexposed_ratio

        # 状态
        self._prev_brightness: float | None = None
        self._overexposed_frames: int = 0
        self._underexposed_frames: int = 0
        self._ae_frames: int = 0
        self._confirm_frames: int = 2

        # 最近一次的分区结果（供外部查询）
        self._last_zoned: ZonedExposureResult | None = None

        # 分区曝光连续帧确认
        self._far_over_frames: int = 0
        self._far_normal_frames: int = 0
        self._zone_confirm: int = 2      # 区域状态切换需连续 N 帧确认

    # ── 公开接口 ──────────────────────────────────────────────────────────

    def detect(self, gray: np.ndarray) -> ExposureState:
        """检测全局曝光状态（保持向后兼容）。"""
        result = self.detect_zones(gray)
        return result.global_state if result else ExposureState.NORMAL

    def detect_zones(self, gray: np.ndarray) -> ZonedExposureResult | None:
        """
        检测分区域曝光状态。

        参数:
          gray: 灰度图 (0-255)，通常为 ROI 裁剪后的图像

        返回:
          ZonedExposureResult，若 gray 无效则返回 None
        """
        if gray is None or gray.size == 0:
            return None

        h, w = gray.shape[:2]
        total_pixels = gray.size

        # ── 全局统计 ──
        over_count = int((gray > self.overexposed_value).sum())
        under_count = int((gray < self.underexposed_value).sum())
        over_ratio = over_count / total_pixels
        under_ratio = under_count / total_pixels

        # ── 分区域统计 ──
        zone_h = h // 3
        if zone_h < 1:
            # 图像太小，回退到全局判断
            zoned = ZonedExposureResult(
                global_state=self._classify_global(over_ratio, under_ratio),
                near=ZoneExposure.NORMAL, mid=ZoneExposure.NORMAL,
                far=ZoneExposure.NORMAL,
                far_overexposed_ratio=0.0, is_far_blind=False,
            )
            self._last_zoned = zoned
            return zoned

        near_zone = gray[h - zone_h:, :]       # 下 1/3 = 近
        mid_zone  = gray[h - 2*zone_h : h - zone_h, :]  # 中 1/3
        far_zone  = gray[:h - 2*zone_h, :]     # 上 ~1/3 = 远

        # 各区过曝比例
        near_over = float((near_zone > self.overexposed_value).sum() / max(near_zone.size, 1))
        mid_over  = float((mid_zone > self.overexposed_value).sum() / max(mid_zone.size, 1))
        far_over  = float((far_zone > self.overexposed_value).sum() / max(far_zone.size, 1))

        # ── 帧间亮度变化 ──
        current_brightness = float(gray.mean())
        brightness_delta = 0.0
        if self._prev_brightness is not None:
            brightness_delta = abs(current_brightness - self._prev_brightness)
        self._prev_brightness = current_brightness

        # ── 全局状态判定（连续帧确认）──
        global_state = ExposureState.NORMAL
        if over_ratio > self.overexposed_ratio:
            self._overexposed_frames += 1
            self._underexposed_frames = 0
            self._ae_frames = 0
            if self._overexposed_frames > self._confirm_frames:
                global_state = ExposureState.OVEREXPOSED
        elif under_ratio > self.underexposed_ratio:
            self._underexposed_frames += 1
            self._overexposed_frames = 0
            self._ae_frames = 0
            if self._underexposed_frames > self._confirm_frames:
                global_state = ExposureState.UNDEREXPOSED
        elif brightness_delta > self.ae_delta_threshold:
            self._ae_frames += 1
            self._overexposed_frames = 0
            self._underexposed_frames = 0
            if self._ae_frames > self._confirm_frames:
                global_state = ExposureState.AE_ADJUSTING
        else:
            self._overexposed_frames = 0
            self._underexposed_frames = 0
            self._ae_frames = 0

        # ── 分区域判定 ──
        near_state = self._zone_state(near_over, self.near_overexposed_ratio)
        mid_state  = self._zone_state(mid_over, self.mid_overexposed_ratio)
        far_state  = self._zone_state(far_over, self.far_overexposed_ratio)

        # 远区盲区：连续超过盲区阈值
        is_blind = False
        if far_over > self.far_blind_ratio:
            self._far_over_frames += 1
            self._far_normal_frames = 0
            if self._far_over_frames > self._zone_confirm:
                is_blind = True
        else:
            self._far_normal_frames += 1
            if self._far_normal_frames > self._zone_confirm:
                self._far_over_frames = 0

        zoned = ZonedExposureResult(
            global_state=global_state,
            near=near_state,
            mid=mid_state,
            far=far_state,
            far_overexposed_ratio=far_over,
            is_far_blind=is_blind,
        )
        self._last_zoned = zoned
        return zoned

    # ── 辅助接口 ──────────────────────────────────────────────────────────

    @property
    def last_zoned(self) -> ZonedExposureResult | None:
        """最近一次的分区曝光结果"""
        return self._last_zoned

    def reset(self):
        """重置历史状态"""
        self._prev_brightness = None
        self._overexposed_frames = 0
        self._underexposed_frames = 0
        self._ae_frames = 0
        self._far_over_frames = 0
        self._far_normal_frames = 0
        self._last_zoned = None

    # ── 内部 ──────────────────────────────────────────────────────────────

    def _classify_global(self, over_ratio: float, under_ratio: float) -> ExposureState:
        """单帧全局分类（无历史确认）。"""
        if over_ratio > self.overexposed_ratio:
            return ExposureState.OVEREXPOSED
        if under_ratio > self.underexposed_ratio:
            return ExposureState.UNDEREXPOSED
        return ExposureState.NORMAL

    @staticmethod
    def _zone_state(ratio: float, threshold: float) -> ZoneExposure:
        if ratio > threshold:
            return ZoneExposure.OVEREXPOSED
        return ZoneExposure.NORMAL
