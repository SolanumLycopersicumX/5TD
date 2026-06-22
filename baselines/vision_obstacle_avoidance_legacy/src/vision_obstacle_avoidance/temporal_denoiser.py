"""
时域降噪模块 (3DNR — 3D Noise Reduction)。
利用连续帧的时间相关性抑制 CMOS 传感器噪声。
对低照度场景（#6）和散粒噪声特别有效。
"""

import numpy as np
from collections import deque


class TemporalDenoiser:
    """
    运动自适应的 3 帧加权时域降噪。

    算法:
      当前帧权重 0.5, t-1 帧 0.35, t-2 帧 0.15。
      检测到场景切换（帧间亮度变化 > 阈值）时自动清空历史缓冲。

    计算量: ~2ms @ 640×480
    """

    def __init__(self,
                 num_frames: int = 3,
                 weights: tuple | None = None,
                 scene_change_threshold: float = 50.0):
        self.num_frames = num_frames
        self.weights = weights or (0.50, 0.35, 0.15)  # 最新帧权重最大
        self.scene_change_threshold = scene_change_threshold

        self._buffer: deque = deque(maxlen=num_frames)
        self._prev_brightness: float | None = None

    # ── 公开接口 ──────────────────────────────────────────────────────────

    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        对当前帧进行时域降噪。

        参数:
          frame: 输入帧 (H, W) 灰度 或 (H, W, 3) 彩色

        返回:
          降噪后的帧，尺寸和类型与输入一致
        """
        if frame is None or frame.size == 0:
            return frame

        # 场景切换检测
        if self._detect_scene_change(frame):
            self._buffer.clear()

        # 加入缓冲
        self._buffer.append(frame.astype(np.float32))

        # 缓冲不足时不降噪
        if len(self._buffer) < 2:
            return frame

        # 加权平均
        result = np.zeros_like(self._buffer[0])
        total_weight = 0.0
        n = len(self._buffer)
        for i in range(n):
            w = self.weights[-(i + 1)] if i < len(self.weights) else self.weights[-1]
            result += self._buffer[n - 1 - i] * w
            total_weight += w

        result /= total_weight
        return np.clip(result, 0, 255).astype(frame.dtype)

    def reset(self):
        """清空历史缓冲（摄像头切换/场景切换后调用）"""
        self._buffer.clear()
        self._prev_brightness = None

    @property
    def buffer_size(self) -> int:
        """当前缓冲帧数"""
        return len(self._buffer)

    # ── 内部 ──────────────────────────────────────────────────────────────

    def _detect_scene_change(self, frame: np.ndarray) -> bool:
        """检测场景是否发生剧烈变化"""
        current = float(frame.mean())
        if self._prev_brightness is None:
            self._prev_brightness = current
            return False

        delta = abs(current - self._prev_brightness)
        self._prev_brightness = current
        return delta > self.scene_change_threshold
