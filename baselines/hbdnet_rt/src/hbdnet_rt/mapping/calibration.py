"""
相机标定模块。
支持手动标定参数输入 (高度/俯仰/FOV) → 生成 image-to-ground homography。
也支持直接传入 homography 矩阵。
"""
import numpy as np
from typing import Optional, Tuple


class CameraCalibration:
    """
    相机标定器。
    从安装参数计算图像平面到地面平面的单应性矩阵。

    坐标系:
      图像: (u, v) 左上角原点
      地面: (X, Y) 车辆正下方为原点, X=右, Y=前
    """

    def __init__(self, config):
        calib_cfg = config.model.get("calibration", {})
        scene_input = config.scene.get("input", {})
        model_input = scene_input.get("model_input", {})

        self.camera_height_m = calib_cfg.get("camera_height_m", 3.5)
        self.camera_pitch_deg = calib_cfg.get("camera_pitch_deg", 12.0)
        self.camera_fov_h_deg = calib_cfg.get("camera_fov_h_deg", 90.0)
        self.image_w = model_input.get("width", 640)
        self.image_h = model_input.get("height", 384)

        self._homography: Optional[np.ndarray] = None
        self._focal_px: Optional[float] = None
        self._principal_point: Optional[Tuple[float, float]] = None

        # 从参数计算内参和 homography
        self._compute_intrinsics()
        self._compute_homography()

    # ── 公开属性 ──

    @property
    def homography(self) -> Optional[np.ndarray]:
        return self._homography

    @property
    def is_calibrated(self) -> bool:
        return self._homography is not None

    # ── 投影 ──

    def image_to_ground(self, u: float, v: float) -> Tuple[float, float]:
        """像素坐标 → 地面坐标 (X_m, Y_m)。使用 homography 或简化模型。"""
        if self._homography is not None:
            pt = np.array([u, v, 1.0])
            ground = self._homography @ pt
            ground /= ground[2]
            return float(ground[0]), float(ground[1])

        # 回退: 简化几何模型
        return self._simple_image_to_ground(u, v)

    def set_homography(self, H: np.ndarray):
        """直接设置 homography 矩阵 (3×3)，覆盖自动计算。"""
        assert H.shape == (3, 3)
        self._homography = H

    def set_manual_params(self, height_m: float, pitch_deg: float, fov_h_deg: float):
        """手动设置相机参数并重新计算。"""
        self.camera_height_m = height_m
        self.camera_pitch_deg = pitch_deg
        self.camera_fov_h_deg = fov_h_deg
        self._compute_intrinsics()
        self._compute_homography()

    # ── 内部 ──

    def _compute_intrinsics(self):
        """从 FOV 和图像尺寸计算近似内参。"""
        fov_h_rad = np.radians(self.camera_fov_h_deg)
        self._focal_px = (self.image_w / 2) / np.tan(fov_h_rad / 2)
        self._principal_point = (self.image_w / 2.0, self.image_h / 2.0)

    def _compute_homography(self):
        """
        计算图像平面 → 地面平面的 homography。
        基于: 地面平面假设 + 已知相机高度和俯仰角。

        四个对应点:
          - 图像底部中心 → 地面近处 (0, y_near)
          - 图像底部左/右 → 地面近处左/右
          - 图像顶部中心 → 地面远处
        用 DLT 求解 homography。
        """
        if self._focal_px is None:
            return

        pitch_rad = np.radians(self.camera_pitch_deg)
        fx = self._focal_px
        fy = self._focal_px
        cx, cy = self._principal_point
        H = self.camera_height_m

        # 图像中的关键行 → 地面深度
        def row_to_depth(v):
            """图像行 v → 地面纵向距离 Y (m)。"""
            alpha = np.arctan2(v - cy, fy)
            theta = pitch_rad + alpha
            if theta <= 0.01:
                return 50.0  # 地平线 → 远距离
            return H / np.tan(theta)

        # 选 4 组对应点 (图像四角 → 地面)
        img_pts = []
        grd_pts = []

        for (u, v) in [(0, self.image_h), (self.image_w, self.image_h),
                        (0, 0), (self.image_w, 0)]:
            Y = row_to_depth(v)
            X = (u - cx) * Y / fx
            img_pts.append([u, v])
            grd_pts.append([X, Y])

        img_pts = np.array(img_pts, dtype=np.float32)
        grd_pts = np.array(grd_pts, dtype=np.float32)

        # 用最小二乘求 homography
        A = []
        for (u, v), (X, Y) in zip(img_pts, grd_pts):
            A.append([-u, -v, -1, 0, 0, 0, u*X, v*X, X])
            A.append([0, 0, 0, -u, -v, -1, u*Y, v*Y, Y])
        A = np.array(A, dtype=np.float64)
        _, _, Vt = np.linalg.svd(A)
        h = Vt[-1] / Vt[-1, -1]
        self._homography = h.reshape(3, 3)

    def _simple_image_to_ground(self, u: float, v: float) -> Tuple[float, float]:
        """简化几何投影 (无 homography 时的回退)。"""
        if self._focal_px is None:
            return 0.0, 0.0
        pitch_rad = np.radians(self.camera_pitch_deg)
        cx, cy = self._principal_point
        fx = self._focal_px
        alpha = np.arctan2(v - cy, fx)
        theta = pitch_rad + alpha
        if theta <= 0.01:
            return 0.0, 50.0
        Y = self.camera_height_m / np.tan(theta)
        X = (u - cx) * Y / fx
        return X, Y
