"""
可行驶区域识别模块。
不依赖标准车道线——通过区域占用率（边缘密度 + 二值 mask）判断各区域可通行程度。
"""

import cv2
import numpy as np
import config
from utils import FreeSpaceState, draw_zone_lines, draw_zone_lines_dynamic


class FreeSpaceDetector:
    """
    可行驶区域检测器。

    算法思路：
      将 ROI 分成左、中、右三区 → 统计各区的边缘像素密度和二值 mask 占比
      → 密度低的区域 = 可行驶（像平坦地面）；密度高的区域 = 可能有障碍/边界 → 综合评分
    """

    def __init__(self):
        if config.ZONE_LEFT_RATIO >= config.ZONE_RIGHT_RATIO:
            raise ValueError(
                f"ZONE_LEFT_RATIO ({config.ZONE_LEFT_RATIO}) 必须小于 "
                f"ZONE_RIGHT_RATIO ({config.ZONE_RIGHT_RATIO})"
            )

    def detect(self, roi_frame: np.ndarray, edges: np.ndarray,
               binary_mask: np.ndarray,
               lane_boundary=None) -> FreeSpaceState:
        state = FreeSpaceState()

        if roi_frame is None or edges is None or binary_mask is None:
            return state

        h, w = edges.shape[:2]
        debug_img = roi_frame.copy()

        # ---- 1. 区域划分 ----
        # 优先使用动态车道边界 (#33)，否则使用固定比例
        if lane_boundary is not None and lane_boundary.is_valid:
            left_edge = lane_boundary.left_barrier_px or 0
            right_edge = lane_boundary.ditch_px or w
            lane_w = right_edge - left_edge

            # 在车道内划分三区
            left_x = left_edge + lane_w // 3
            right_x = right_edge - lane_w // 3

            left_zone   = (left_edge, 0, left_x,  h)
            center_zone = (left_x,   0, right_x, h)
            right_zone  = (right_x,  0, right_edge, h)
        else:
            # 回退：使用 config 中的固定比例（向后兼容）
            left_x   = int(w * config.ZONE_LEFT_RATIO)
            right_x  = int(w * config.ZONE_RIGHT_RATIO)
            left_zone   = (0,      0, left_x,       h)
            center_zone = (left_x, 0, right_x,      h)
            right_zone  = (right_x,0, w,            h)

        # ---- 2. 计算各区域边缘密度 ----
        # 边缘越密集 → 地面纹理越复杂 → 可能有障碍/不可通行
        left_edge_density   = self._edge_density(edges, *left_zone)
        center_edge_density = self._edge_density(edges, *center_zone)
        right_edge_density  = self._edge_density(edges, *right_zone)

        # ---- 3. 计算各区域地面 mask 占比 ----
        # mask 中白色=地面，mask 占比低 = 被非地面物体占据
        left_mask_ratio   = self._mask_ratio(binary_mask, *left_zone)
        center_mask_ratio = self._mask_ratio(binary_mask, *center_zone)
        right_mask_ratio  = self._mask_ratio(binary_mask, *right_zone)

        # ---- 4. 综合评分 ----
        # 综合 = 30% 边缘得分 + 70% mask 得分
        def combined_score(edge_density, mask_ratio):
            edge_score = max(0, 1 - edge_density / config.FREE_SPACE_EDGE_THRESHOLD)
            return 0.3 * edge_score + 0.7 * mask_ratio

        state.left_free_score   = combined_score(left_edge_density, left_mask_ratio)
        state.center_free_score = combined_score(center_edge_density, center_mask_ratio)
        state.right_free_score  = combined_score(right_edge_density, right_mask_ratio)

        # ---- 5. 计算 center_offset ----
        # 比较左右两侧可行驶程度，判断车辆应该往哪边修正
        left_weight  = state.left_free_score + 0.001
        right_weight = state.right_free_score + 0.001
        state.center_offset = (right_weight - left_weight) / (left_weight + right_weight)
        # center_offset > 0 → 右侧更畅通 → 车辆偏左了 → 需要往右修

        # ---- 6. 置信度 ----
        state.confidence = state.center_free_score
        state.is_valid = (
            state.center_free_score > 0.2 or
            state.left_free_score > 0.3 or
            state.right_free_score > 0.3
        )

        # ---- 7. 可视化 ----
        # 使用实际区域边界绘制
        if lane_boundary is not None and lane_boundary.is_valid:
            draw_zone_lines_dynamic(debug_img, left_zone, center_zone, right_zone, h)
        else:
            draw_zone_lines(debug_img, w, h)

        # 在各区域中心标注评分
        font = cv2.FONT_HERSHEY_SIMPLEX
        for name, x1, x2, score in [
            ("L", 0, left_x, state.left_free_score),
            ("C", left_x, right_x, state.center_free_score),
            ("R", right_x, w, state.right_free_score),
        ]:
            cx = (x1 + x2) // 2
            cv2.putText(debug_img, f"{name}:{score:.2f}", (cx - 25, h // 2),
                        font, 0.45, (0, 255, 255), 1)

        # 绘制 center_offset 指示条
        bar_y = h - 15
        cv2.line(debug_img, (w // 2, bar_y - 8), (w // 2, bar_y + 8), (0, 255, 0), 1)
        offset_x = int(w // 2 + state.center_offset * w * 0.3)
        cv2.circle(debug_img, (offset_x, bar_y), 5, (0, 255, 255), -1)

        state.debug_image = debug_img
        return state

    # ---- 内部方法 ----

    @staticmethod
    def _edge_density(edges: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> float:
        """计算区域内边缘像素密度。"""
        roi = edges[y1:y2, x1:x2]
        if roi.size == 0:
            return 0.0
        return np.count_nonzero(roi) / roi.size

    @staticmethod
    def _mask_ratio(binary_mask: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> float:
        """计算区域内地面 mask（白色像素）占比。"""
        roi = binary_mask[y1:y2, x1:x2]
        if roi.size == 0:
            return 0.0
        return np.count_nonzero(roi) / roi.size
