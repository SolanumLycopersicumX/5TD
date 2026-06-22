"""
障碍检测模块。
通过轮廓分析 + 面积过滤 + 位置判断识别 ROI 内的障碍物。
"""

import cv2
import numpy as np
import config
from utils import ObstacleState, draw_zone_lines


class ObstacleDetector:
    """
    障碍检测器。

    算法思路：
      在边缘图上找轮廓 → 过滤面积/宽高比异常值 → 按左中右分区归类
      → 判断最危险区域 → 输出结构化 ObstacleState
    """

    def __init__(self):
        pass

    def detect(self, roi_frame: np.ndarray, edges: np.ndarray,
               binary_mask: np.ndarray) -> ObstacleState:
        state = ObstacleState()

        if roi_frame is None or edges is None or binary_mask is None:
            return state

        h, w = edges.shape[:2]
        debug_img = roi_frame.copy()
        draw_zone_lines(debug_img, w, h)

        # ---- 1. 找轮廓 ----
        # RETR_EXTERNAL: 只取最外层轮廓，减少内部嵌套干扰
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # ---- 2. 过滤轮廓 ----
        valid_boxes = []
        left_x   = int(w * config.ZONE_LEFT_RATIO)
        right_x  = int(w * config.ZONE_RIGHT_RATIO)
        danger_y = int(h * config.DANGER_ZONE_TOP_RATIO)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < config.OBSTACLE_MIN_AREA or area > config.OBSTACLE_MAX_AREA:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect = bw / (bh + 1e-5)

            # 过滤过于细长的假阳性（地面裂缝、反光条等）
            if aspect < config.OBSTACLE_MIN_ASPECT or aspect > config.OBSTACLE_MAX_ASPECT:
                continue

            valid_boxes.append((x, y, bw, bh, area))

        state.obstacle_boxes = [(x, y, bw, bh) for (x, y, bw, bh, _) in valid_boxes]

        # ---- 3. 按区域分类 + 计算危险等级 ----
        blocked_left = blocked_center = blocked_right = False
        danger_scores = []
        closest_zone = "NONE"
        closest_dist = float("inf")

        for x, y, bw, bh, area in valid_boxes:
            cx = x + bw // 2
            cy = y + bh // 2
            bottom = y + bh

            # 区域判断
            if cx < left_x:
                blocked_left = True
                zone = "LEFT"
            elif cx > right_x:
                blocked_right = True
                zone = "RIGHT"
            else:
                blocked_center = True
                zone = "CENTER"

            # 危险等级：越靠底部（离车越近）+ 面积越大 = 越危险
            # bottom 越接近 ROI 底部 → 距离车越近
            proximity = bottom / h
            area_score = min(1.0, area / config.OBSTACLE_AREA_REFERENCE)
            danger = 0.5 * proximity + 0.5 * area_score
            danger_scores.append(danger)

            # 最近障碍物
            if proximity > 0.3 and bottom > closest_dist:
                closest_dist = bottom
                closest_zone = zone

            # 绘制障碍框
            color = (0, 0, 255) if danger > 0.6 else (0, 200, 255)
            cv2.rectangle(debug_img, (x, y), (x + bw, y + bh), color, 2)
            cv2.putText(debug_img, f"D:{danger:.1f}", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

        # ---- 4. 组装状态 ----
        state.has_obstacle = len(valid_boxes) > 0
        state.blocked_left = blocked_left
        state.blocked_center = blocked_center
        state.blocked_right = blocked_right
        state.largest_obstacle_area = max([a for _, _, _, _, a in valid_boxes], default=0)
        state.danger_level = max(danger_scores) if danger_scores else 0.0
        state.closest_obstacle_zone = closest_zone
        state.debug_image = debug_img

        # ---- 5. 危险区域可视化 ----
        cv2.line(debug_img, (0, danger_y), (w, danger_y), (0, 0, 255), 1)
        cv2.putText(debug_img, "DANGER ZONE", (5, danger_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        return state
