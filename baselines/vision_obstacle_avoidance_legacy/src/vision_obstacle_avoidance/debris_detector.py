"""
碎石/碎渣检测模块 (#18)。
通过局部边缘密度异常检测路面上的碎石/碎渣。
碎石与水泥地面同色，但破坏路面纹理一致性 → 边缘密度出现局部峰值。

算法:
  1. 将车道内边缘图划分为网格
  2. 计算每单元边缘密度
  3. 滑动窗口统计局部均值/标准差 → 标记异常单元
  4. 合并相邻异常单元 → 碎石斑块
  5. 像素尺寸 → 厘米 (需要 calibrator)
  6. 分类: <5cm 忽略, 5-15cm 记录, >15cm 标记为需避让
"""

import cv2
import numpy as np

import config
from utils import DebrisState


class DebrisDetector:
    """
    碎石/碎渣检测器。

    输入: roi_gray (灰度图), roi_edges (Canny 边缘),
          lane_boundary (LaneBoundaryState, 限定搜索范围),
          calibrator (GroundCalibrator, 可选, 用于像素→厘米)

    输出: DebrisState
    """

    def __init__(self):
        pass

    # ── 公开接口 ──────────────────────────────────────────────────────────

    def detect(self, roi_gray: np.ndarray, roi_edges: np.ndarray,
               lane_boundary=None, calibrator=None) -> DebrisState:
        state = DebrisState()

        if roi_edges is None or roi_edges.size == 0:
            return state

        h, w = roi_edges.shape[:2]
        debug_img = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2BGR) \
            if roi_gray is not None else np.zeros((h, w, 3), dtype=np.uint8)

        # ---- 1. 限定搜索范围（车道内）----
        x_start, x_end = 0, w
        if lane_boundary is not None and lane_boundary.is_valid:
            if lane_boundary.left_barrier_px is not None:
                x_start = lane_boundary.left_barrier_px
            if lane_boundary.ditch_px is not None:
                x_end = lane_boundary.ditch_px
        if x_end <= x_start:
            return state

        # ---- 2. 网格划分 + 边缘密度 ----
        cell = config.DEBRIS_GRID_CELL
        cols = (x_end - x_start) // cell
        rows = h // cell
        if cols < 2 or rows < 2:
            return state

        density_map = np.zeros((rows, cols), dtype=np.float32)
        for r in range(rows):
            for c in range(cols):
                y1, y2 = r * cell, min((r + 1) * cell, h)
                x1 = x_start + c * cell
                x2 = min(x_start + (c + 1) * cell, x_end)
                patch = roi_edges[y1:y2, x1:x2]
                if patch.size > 0:
                    density_map[r, c] = np.count_nonzero(patch) / patch.size

        # ---- 3. 异常检测 ----
        anomalies = np.zeros((rows, cols), dtype=np.uint8)
        mult = config.DEBRIS_EDGE_DENSITY_STD_MULT
        half_win = 1  # 3x3 邻域

        for r in range(rows):
            for c in range(cols):
                r1, r2 = max(0, r - half_win), min(rows, r + half_win + 1)
                c1, c2 = max(0, c - half_win), min(cols, c + half_win + 1)
                neighborhood = density_map[r1:r2, c1:c2]
                if neighborhood.size < 3:
                    continue
                mu = neighborhood.mean()
                sigma = neighborhood.std()
                if density_map[r, c] > mu + mult * sigma:
                    anomalies[r, c] = 255

        if not np.any(anomalies):
            state.debug_image = debug_img
            return state

        # ---- 4. 合并相邻异常 ----
        num_labels, labels = cv2.connectedComponents(anomalies, connectivity=8)
        debris_patches = []

        for label_id in range(1, num_labels):
            mask = (labels == label_id).astype(np.uint8)
            ys, xs = np.where(mask)
            if len(ys) < 2:  # 至少 2 个单元
                continue

            # 斑块在 ROI 中的边界
            y1 = ys.min() * cell
            y2 = min((ys.max() + 1) * cell, h)
            x1 = x_start + xs.min() * cell
            x2 = min(x_start + (xs.max() + 1) * cell, x_end)
            area_px = len(ys) * cell * cell

            # 像素尺寸 → 厘米
            size_cm = 0.0
            if calibrator and calibrator.calibrated:
                m_per_px = calibrator.meters_at_row((y1 + y2) // 2, h)
                if m_per_px:
                    size_cm = np.sqrt(area_px) * m_per_px * 100  # 对角线长度 (cm)

            debris_patches.append((x1, y1, x2 - x1, y2 - y1, size_cm))

        # ---- 5. 分类 ----
        state.debris_boxes = [(x, y, bw, bh, sz) for (x, y, bw, bh, sz) in debris_patches
                              if sz >= config.DEBRIS_SMALL_CM]
        state.has_debris = len(state.debris_boxes) > 0
        state.has_large_debris = any(
            sz >= config.DEBRIS_LARGE_CM for (_, _, _, _, sz) in state.debris_boxes)

        # ---- 6. 可视化 ----
        for x, y, bw, bh, sz in state.debris_boxes:
            color = (0, 0, 255) if sz >= config.DEBRIS_LARGE_CM else (255, 0, 0)
            cv2.rectangle(debug_img, (x, y), (x + bw, y + bh), color, 1)
            label = f"{sz:.0f}cm"
            cv2.putText(debug_img, label, (x, y - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

        # 搜索范围边界
        cv2.line(debug_img, (x_start, 0), (x_start, h), (100, 100, 100), 1)
        cv2.line(debug_img, (x_end, 0), (x_end, h), (100, 100, 100), 1)

        state.debug_image = debug_img
        return state
