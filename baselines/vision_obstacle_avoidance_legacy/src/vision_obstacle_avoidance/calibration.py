"""
像素→度量空间标定模块 (#49)。
利用消失点 + 地面平面假设，建立图像坐标到地面坐标的映射。
标定未收敛时，所有米制计算不可用，上游模块应回退固定区域行为。
"""

import math
from collections import deque

import cv2
import numpy as np

import config


class GroundCalibrator:
    """
    像素→地面平面单应性标定。

    算法:
      1. 从边缘图中提取纵向 Hough 直线
      2. RANSAC 求消失点 (vanishing point)
      3. 多帧累积 + 方差收敛判定
      4. 地面平面假设: depth = H / tan(θ + α)，推算每行像素对应的米数

    容错:
      - 标定未收敛时 calibrated=False，所有米制查询返回 None
      - Hough 直线不足时 estimate_vp 返回 None，不影响主循环
    """

    def __init__(self, camera_height_m=None, camera_pitch_deg=None):
        self.camera_height_m = camera_height_m or config.CAMERA_HEIGHT_M
        self.camera_pitch_deg = camera_pitch_deg or config.CAMERA_PITCH_DEG
        self.camera_pitch_rad = math.radians(self.camera_pitch_deg)

        # ── 消失点累积 ──
        self._vp_buffer = deque(maxlen=config.VP_ACCUMULATE_FRAMES)
        self.vanishing_point: tuple | None = None   # (vx, vy) 图像坐标
        self.calibrated: bool = False

        # ── 相机内参近似 ──
        # 假设主点在图像中心，焦距由 pitch 角 + 消失点位置反推
        self._focal_length_px: float | None = None   # 估算焦距 (像素)
        self._principal_point: tuple | None = None    # (cx, cy) 主点

    # ── 公开接口 ──────────────────────────────────────────────────────────

    def estimate_vp_from_edges(self, edges: np.ndarray,
                               roi_offset_x: int = 0,
                               roi_offset_y: int = 0) -> tuple | None:
        """
        从边缘图中提取消失点。

        参数:
          edges: Canny 边缘二值图
          roi_offset_x, roi_offset_y: 边缘图在原图中的偏移（ROI 左上角坐标）

        返回: (vx, vy) 在原图坐标系中，或 None
        """
        if edges is None or edges.size == 0:
            return None

        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180,
            threshold=config.HOUGH_THRESHOLD,
            minLineLength=config.HOUGH_MIN_LINE_LEN,
            maxLineGap=config.HOUGH_MAX_LINE_GAP,
        )
        if lines is None or len(lines) < config.VP_MIN_LINES:
            return None

        # 筛选纵向直线
        longitudinal = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(math.atan2(y2 - y1, x2 - x1 + 1e-9))
            if config.LINE_ANGLE_MIN <= angle <= config.LINE_ANGLE_MAX:
                # 转回原图坐标
                longitudinal.append((
                    x1 + roi_offset_x, y1 + roi_offset_y,
                    x2 + roi_offset_x, y2 + roi_offset_y,
                ))

        if len(longitudinal) < config.VP_MIN_LINES:
            return None

        # RANSAC 求所有直线对的交点
        vp = self._ransac_intersection(longitudinal)
        return vp

    def accumulate(self, vp: tuple) -> bool:
        """
        累积消失点观测。返回 True 表示刚完成收敛。

        收敛条件: 窗口内 VP 的 x,y 标准差均低于阈值。
        """
        if vp is None:
            return False
        self._vp_buffer.append(vp)
        if len(self._vp_buffer) < config.VP_ACCUMULATE_FRAMES:
            return False

        pts = np.array(self._vp_buffer)
        std_x, std_y = np.std(pts[:, 0]), np.std(pts[:, 1])
        if std_x < 5.0 and std_y < 5.0:   # 5像素标准差=稳定
            self.vanishing_point = (float(np.median(pts[:, 0])),
                                    float(np.median(pts[:, 1])))
            self._compute_focal()
            self.calibrated = True
            return True
        return False

    def meters_at_row(self, row: int, image_height: int) -> float | None:
        """
        返回指定行每像素对应多少米（水平方向）。
        row: 图像行号（原图坐标系）
        image_height: 原图高度
        """
        if not self.calibrated or self._focal_length_px is None:
            return None
        if self._principal_point is None:
            return None
        cy = self._principal_point[1]
        fy = self._focal_length_px
        # 该行的俯角
        alpha = math.atan2(row - cy, fy)
        theta_eff = self.camera_pitch_rad + alpha
        if theta_eff <= 0:
            return None   # 看向地平线以上
        depth = self.camera_height_m / math.tan(theta_eff)
        # depth 米对应 fy / cos(alpha) 像素
        px_per_m_vertical = fy / (depth * math.cos(alpha))
        return 1.0 / max(px_per_m_vertical, 1e-9)

    def pixel_to_ground(self, x_px: int, y_px: int,
                         image_width: int, image_height: int) -> tuple | None:
        """
        将图像坐标转换为地面坐标 (x_m 横向, y_m 纵向)。
        原点: 车辆正下方地面点。
        """
        m_per_px = self.meters_at_row(y_px, image_height)
        if m_per_px is None or self._principal_point is None:
            return None
        cx = self._principal_point[0]
        cy = self._principal_point[1]
        fy = self._focal_length_px
        # 横向: 像素偏移 × 每像素米数
        x_m = (x_px - cx) * m_per_px
        # 纵向: 深度
        alpha = math.atan2(y_px - cy, fy)
        theta_eff = self.camera_pitch_rad + alpha
        if theta_eff <= 0:
            return None
        y_m = self.camera_height_m / math.tan(theta_eff)
        return x_m, y_m

    # ── 内部 ──────────────────────────────────────────────────────────────

    def _ransac_intersection(self, lines: list, max_iter=100,
                             inlier_thresh=None) -> tuple | None:
        """RANSAC 求多条直线的交点（消失点）。"""
        if inlier_thresh is None:
            inlier_thresh = config.VP_RANSAC_THRESHOLD

        best_vp = None
        best_inliers = 0
        n = len(lines)

        for _ in range(max_iter):
            # 随机选两条线
            i, j = np.random.choice(n, 2, replace=False)
            vp = self._line_intersection(lines[i], lines[j])
            if vp is None:
                continue
            # 计数内点
            inliers = 0
            for k in range(n):
                dist = self._point_to_line_distance(vp, lines[k])
                if dist < inlier_thresh:
                    inliers += 1
            if inliers > best_inliers:
                best_inliers = inliers
                best_vp = vp

        # 用所有内点精炼
        if best_vp is not None and best_inliers >= config.VP_MIN_LINES:
            inlier_pts = []
            for k in range(n):
                if self._point_to_line_distance(best_vp, lines[k]) < inlier_thresh:
                    inlier_pts.append(self._line_midpoint(lines[k]))
            if inlier_pts:
                pts = np.array(inlier_pts)
                return (float(np.median(pts[:, 0])), float(np.median(pts[:, 1])))
        return best_vp

    @staticmethod
    def _line_intersection(l1, l2) -> tuple | None:
        """两条线段的交点 (延长线)。"""
        x1, y1, x2, y2 = l1
        x3, y3, x4, y4 = l2
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-9:
            return None
        px = ((x1 * y2 - y1 * x2) * (x3 - x4) -
              (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
        py = ((x1 * y2 - y1 * x2) * (y3 - y4) -
              (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
        return px, py

    @staticmethod
    def _point_to_line_distance(pt, line) -> float:
        """点到直线（延长线）的距离。"""
        x0, y0 = pt
        x1, y1, x2, y2 = line
        return abs((x2 - x1) * (y1 - y0) - (x1 - x0) * (y2 - y1)) / \
               (math.hypot(x2 - x1, y2 - y1) + 1e-9)

    @staticmethod
    def _line_midpoint(line) -> tuple:
        x1, y1, x2, y2 = line
        return (x1 + x2) / 2, (y1 + y2) / 2

    def _compute_focal(self):
        """利用消失点 + pitch 角反推焦距。"""
        if self.vanishing_point is None:
            return
        vx, vy = self.vanishing_point
        # 假设主点在图像中心
        cx, cy = 640.0, 480.0   # 默认，调用方可覆盖
        self._principal_point = (cx, cy)
        # vy = cy - f * tan(pitch)  (地平线在图像中的位置)
        # → f = (cy - vy) / tan(pitch)
        if abs(math.tan(self.camera_pitch_rad)) > 1e-9:
            self._focal_length_px = (cy - vy) / math.tan(self.camera_pitch_rad)
        else:
            self._focal_length_px = 800.0  # 合理默认值

    def set_principal_point(self, cx: float, cy: float):
        """手动设置主点（ROI 偏移后调用）。"""
        self._principal_point = (cx, cy)
        if self.vanishing_point is not None:
            self._compute_focal()
