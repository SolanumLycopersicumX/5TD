"""
纵向结构检测模块 (#31-32)。
检测隧道场景中的中央隔离沟 + 两侧凸起隔离带，
输出 LaneBoundaryState 作为车道硬边界供下游模块使用。

算法:
  #31 隔离沟 = ROI 下半部的纵向暗色带 → 列投影 + V 通道最小值
  #32 隔离带 = 纵向边缘簇 → HoughLinesP + 消失点筛选 + 聚类
  两者均使用时序中值滤波抑制单帧误检。
"""

import math
from collections import deque

import cv2
import numpy as np

import config
from utils import LaneBoundaryState


class LaneBoundaryDetector:
    """
    纵向结构检测器。

    输入: roi_frame (BGR), edges (Canny), enhanced (CLAHE 增强灰度图),
          calibrator (GroundCalibrator, 可选)

    输出: LaneBoundaryState

    容错: 任一结构未检测到时，对应字段为 None，is_valid 仅需至少检测到隔离沟。
    """

    def __init__(self):
        # 时序中值滤波缓冲
        self._ditch_history = deque(maxlen=config.BOUNDARY_HISTORY_FRAMES)
        self._left_barrier_history = deque(maxlen=config.BOUNDARY_HISTORY_FRAMES)
        self._right_barrier_history = deque(maxlen=config.BOUNDARY_HISTORY_FRAMES)

    # ── 公开接口 ──────────────────────────────────────────────────────────

    def detect(self, roi_frame: np.ndarray, edges: np.ndarray,
               enhanced: np.ndarray, calibrator=None) -> LaneBoundaryState:
        """检测所有纵向结构。"""
        state = LaneBoundaryState()

        if roi_frame is None or roi_frame.size == 0:
            return state
        if edges is None or edges.size == 0:
            return state

        h, w = edges.shape[:2]
        debug_img = roi_frame.copy()
        vp = calibrator.vanishing_point if calibrator and calibrator.calibrated else None
        state.vanishing_point = vp

        # ---- #31: 中央隔离沟 (v2.1: 跳变抑制) ----
        ditch_px = self._detect_ditch(roi_frame, edges, w, h)
        if ditch_px is not None:
            if self._innovation_check(self._ditch_history, ditch_px, config.BOUNDARY_MAX_JUMP_PX):
                self._ditch_history.append(ditch_px)
            state.ditch_px = int(np.median(self._ditch_history)) if self._ditch_history else None

        # ---- #32: 两侧隔离带 (v2.1: 跳变抑制) ----
        left_px, right_px = self._detect_barriers(edges, w, h, vp)
        if left_px is not None:
            if self._innovation_check(self._left_barrier_history, left_px, config.BOUNDARY_MAX_JUMP_PX):
                self._left_barrier_history.append(left_px)
            state.left_barrier_px = int(np.median(self._left_barrier_history)) if self._left_barrier_history else None
        if right_px is not None:
            if self._innovation_check(self._right_barrier_history, right_px, config.BOUNDARY_MAX_JUMP_PX):
                self._right_barrier_history.append(right_px)
            state.right_barrier_px = int(np.median(self._right_barrier_history)) if self._right_barrier_history else None

        # ---- 置信度 ----
        confidence = 1.0
        if state.ditch_px is None:
            confidence -= 0.4
        if state.left_barrier_px is None:
            confidence -= 0.3
        if state.right_barrier_px is None:
            confidence -= 0.3
        state.confidence = max(0.0, confidence)
        state.is_valid = state.ditch_px is not None  # 隔离沟是核心

        # ---- 车道宽度估算 ----
        if calibrator and calibrator.calibrated and state.left_barrier_px is not None \
                and state.ditch_px is not None:
            m_per_px = calibrator.meters_at_row(
                int(h * config.DITCH_STRIP_RATIO_BOTTOM), h)
            if m_per_px:
                state.lane_width_m = (state.ditch_px - state.left_barrier_px) * m_per_px

        # ---- 可视化 ----
        self._draw_boundaries(debug_img, state, h)
        state.debug_image = debug_img

        return state

    # ── #31: 隔离沟检测 ───────────────────────────────────────────────────

    def _detect_ditch(self, roi_frame: np.ndarray, edges: np.ndarray,
                      w: int, h: int) -> int | None:
        """
        检测中央隔离沟。
        v3.0: 三特征综合评分 — 暗度 + V梯度 + Canny边缘密度。
        适配隧道暗区和亮区两种场景：
        - 暗区：隔离沟是明显的暗色条带（高暗度）
        - 亮区：隔离沟是亮度/纹理突变处（高V梯度+边缘密度）
        """
        # 取 ROI 下半部
        strip_top = int(h * config.DITCH_STRIP_RATIO_BOTTOM)
        roi_strip = roi_frame[strip_top:, :]
        if roi_strip.size == 0:
            return None

        # 转 HSV，取 V 通道
        hsv_strip = cv2.cvtColor(roi_strip, cv2.COLOR_BGR2HSV)
        v_strip = hsv_strip[:, :, 2]

        # 列投影（每列 V 均值）
        col_means = v_strip.mean(axis=0)

        # 搜索范围
        left = int(w * config.DITCH_SEARCH_LEFT)
        right = int(w * config.DITCH_SEARCH_RIGHT)
        if right <= left:
            return None

        # 平滑
        kernel = max(5, w // 30)
        smoothed = np.convolve(col_means, np.ones(kernel) / kernel, mode='same')
        mean_val = np.median(smoothed[left:right])

        # 预计算V梯度（每列的水平亮度变化率）
        v_gradients = np.zeros(w, dtype=np.float32)
        for x in range(5, w - 5):
            left_v = smoothed[x - 5:x].mean()
            right_v = smoothed[x:x + 5].mean()
            v_gradients[x] = abs(right_v - left_v) / 255.0

        # ---- 三特征综合评分 ----
        best_idx = None
        best_score = 0.0

        for abs_idx in range(left, right):
            # 特征1: 暗度 (归一化)
            darkness = (mean_val - smoothed[abs_idx]) / 255.0
            darkness = max(0.0, darkness)

            # 特征2: V梯度
            v_grad = v_gradients[abs_idx]

            # 特征3: Canny边缘密度
            col_slice = edges[strip_top:, max(0, abs_idx - 5):min(w, abs_idx + 5)]
            edge_density = 0.0
            if col_slice.size > 0:
                edge_density = np.count_nonzero(col_slice) / col_slice.size

            # 综合评分: 暗度(30%) + V梯度(40%) + 边缘密度(30%)
            # V梯度权重最高，因为在亮场景下隔离沟主要表现为亮度突变
            score = 0.3 * darkness + 0.4 * v_grad + 0.3 * edge_density

            # 附加：暗度加分（暗区特征更可靠）
            if darkness * 255 >= config.DITCH_V_DARK_THRESHOLD:
                score += 0.01  # 暗度达标的小额加分

            if score > best_score:
                best_score = score
                best_idx = abs_idx

        if best_idx is None:
            return None
        if best_score < config.DITCH_EDGE_DENSITY_MIN:
            return None

        return best_idx

    # ── #32: 隔离带检测 ───────────────────────────────────────────────────

    def _detect_barriers(self, edges: np.ndarray, w: int, h: int,
                         vp: tuple | None) -> tuple:
        """
        检测两侧凸起隔离带。
        方法: HoughLinesP → 筛选纵向直线 → 按消失点约束过滤 → 左右聚类。
        返回 (left_px, right_px)，任一可为 None。
        """
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180,
            threshold=config.HOUGH_THRESHOLD,
            minLineLength=config.HOUGH_MIN_LINE_LEN,
            maxLineGap=config.HOUGH_MAX_LINE_GAP,
        )
        if lines is None:
            return None, None

        # 筛选纵向 + 消失点约束
        left_intersections = []
        right_intersections = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(math.atan2(y2 - y1, x2 - x1 + 1e-9))
            if not (config.LINE_ANGLE_MIN <= angle <= config.LINE_ANGLE_MAX):
                continue

            # 消失点约束
            if vp is not None:
                dist = self._line_to_point_dist(x1, y1, x2, y2, vp[0], vp[1])
                if dist > config.LINE_VP_DISTANCE_THRESHOLD:
                    continue

            # 用线段底部的 x 坐标作为截距（底部 = 较大的 y）
            x_bottom = x2 if y2 > y1 else x1
            # 分类到左/右
            if x_bottom < w * config.BARRIER_LEFT_MAX:
                left_intersections.append(x_bottom)
            elif x_bottom > w * config.BARRIER_RIGHT_MIN:
                right_intersections.append(x_bottom)

        left_px = int(np.median(left_intersections)) if left_intersections else None
        right_px = int(np.median(right_intersections)) if right_intersections else None

        return left_px, right_px

    # ── 可视化 ────────────────────────────────────────────────────────────

    @staticmethod
    def _draw_boundaries(debug_img: np.ndarray, state: LaneBoundaryState, h: int):
        """在 debug 图像上绘制检测到的纵向结构。"""
        # 绿色竖线 = 隔离沟
        if state.ditch_px is not None:
            cv2.line(debug_img, (state.ditch_px, 0),
                     (state.ditch_px, h), (0, 255, 0), 2)
            cv2.putText(debug_img, "DITCH", (state.ditch_px + 5, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 黄色竖线 = 隔离带
        for name, x in [("L_BAR", state.left_barrier_px),
                         ("R_BAR", state.right_barrier_px)]:
            if x is not None:
                cv2.line(debug_img, (x, 0), (x, h), (0, 255, 255), 1)
                cv2.putText(debug_img, name, (x + 5, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        # 蓝色虚线 = 车道中心线 (隔离沟与左侧隔离带的中点)
        if state.left_barrier_px is not None and state.ditch_px is not None:
            center = (state.left_barrier_px + state.ditch_px) // 2
            for y in range(0, h, 10):
                cv2.line(debug_img, (center, y), (center, y + 5),
                         (255, 0, 0), 1)

        # 置信度
        cv2.putText(debug_img, f"Conf:{state.confidence:.2f}",
                    (5, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (200, 200, 200), 1)

    # ── 跳变抑制 (v2.1) ─────────────────────────────────────────────────

    @staticmethod
    def _innovation_check(history: deque, new_value: float,
                          max_jump_px: float = 50) -> bool:
        """
        新息检验: 如果历史已有足够样本且新值偏离中位数过大，拒绝加入。
        防止暗色井盖、水渍等野值污染时序滤波器。

        返回 True 表示通过，可加入历史；False 表示拒绝。
        """
        if len(history) < 3:
            return True  # 样本不足，全部接受

        median = np.median(history)
        if abs(new_value - median) > max_jump_px:
            return False
        return True

    # ── 工具 ──────────────────────────────────────────────────────────────

    @staticmethod
    def _line_to_point_dist(x1, y1, x2, y2, px, py) -> float:
        """点到直线的距离。"""
        return abs((x2 - x1) * (y1 - py) - (x1 - px) * (y2 - y1)) / \
               (math.hypot(x2 - x1, y2 - y1) + 1e-9)
