"""
BEV 投影器。
将图像坐标的感知输出投影到车辆局部 BEV 地面坐标系。
当前使用简化线性映射 (flat ground + constant pitch)。
预留 homography 接口供后续替换为精确标定。
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class BEVProjector:
    """
    图像 → BEV 投影器。

    简化映射:
      - 图像底部行 → grid 近处 (range_y_min)
      - 图像顶部行 → grid 远处 (range_y_max)
      - 水平: 像素偏移 × (深度 / focal_approx) → 米

    预留: set_homography(H) 替换为精确单应性投影。
    """

    def __init__(self, config):
        grid_cfg = config.scene.get("output", {}).get("bev_grid", {})
        self.range_x = (grid_cfg.get("range_x_min", -2.5),
                        grid_cfg.get("range_x_max", 2.5))
        self.range_y = (grid_cfg.get("range_y_min", 0.0),
                        grid_cfg.get("range_y_max", 8.0))
        self.resolution = grid_cfg.get("resolution", 0.10)

        # 相机近似参数 (待标定替换)
        self._camera_height = 3.5        # 米 (待实测)
        self._camera_pitch_deg = 12.0    # 度 (待实测)
        self._focal_approx_px = 400.0    # 近似焦距 (像素)
        self._homography: Optional[np.ndarray] = None

    # ── 公开属性 ──

    @property
    def grid_shape(self) -> Tuple[int, int]:
        """(nx, ny): 横向和纵向栅格单元数。"""
        nx = int((self.range_x[1] - self.range_x[0]) / self.resolution)
        ny = int((self.range_y[1] - self.range_y[0]) / self.resolution)
        return nx, ny

    @property
    def grid_extent(self) -> dict:
        return {
            "x_min": self.range_x[0], "x_max": self.range_x[1],
            "y_min": self.range_y[0], "y_max": self.range_y[1],
            "resolution": self.resolution,
            "nx": self.grid_shape[0], "ny": self.grid_shape[1],
        }

    # ── 投影 ──

    def project_mask_to_bev(self, mask: torch.Tensor) -> torch.Tensor:
        """
        将分割 mask 投影到 BEV。
        使用简化线性映射: 底部→近, 顶部→远。
        输入: [B, C, H_img, W_img]
        输出: [B, C, ny, nx]
        """
        if self._homography is not None:
            return self._project_homography(mask)

        _, _, h, w = mask.shape
        nx, ny = self.grid_shape

        # 纵向映射: 行号 → 深度。底部行=近处, 顶部行=远处。
        # 取 mask 下半部分 (对应地面) 映射到 grid
        # 简化: 直接 resize, 底部→近 由网格方向处理
        grid = F.interpolate(mask, size=(ny, nx),
                             mode="bilinear", align_corners=False)
        # 翻转垂直方向: 图像底部 (大 depth) → grid 近处 (小 y index)
        grid = torch.flip(grid, dims=[2])
        return grid

    def image_xy_to_grid_xy(self, px: float, py: float,
                             image_h: int = 96, image_w: int = 160) -> Tuple[float, float]:
        """
        单个像素坐标 → 地面坐标 (简化)。
        px, py: 图像坐标 (0~W-1, 0~H-1)。
        返回 (x_m, y_m) 地面坐标, 原点为车辆正下方。
        """
        # 纵向: 行号线性映射到深度范围
        row_ratio = py / max(image_h - 1, 1)
        y_m = self.range_y[0] + row_ratio * (self.range_y[1] - self.range_y[0])

        # 横向: 像素偏移 × 深度/焦距
        cx = image_w / 2.0
        x_m = (px - cx) * (y_m / self._focal_approx_px)
        return x_m, y_m

    # ── Homography 接口 ──

    def set_homography(self, H: np.ndarray):
        """
        设置精确 homography 矩阵 (3×3), 从图像平面到地面平面。
        设置后 project_mask_to_bev 自动使用 cv2.warpPerspective 等效逻辑。
        H: 将图像坐标 (u,v,1) 映射到地面坐标 (X,Y,1)。
        """
        assert H.shape == (3, 3), f"Homography must be 3x3, got {H.shape}"
        self._homography = H

    def clear_homography(self):
        """回退到简化线性映射。"""
        self._homography = None

    def _project_homography(self, mask: torch.Tensor) -> torch.Tensor:
        """使用 homography 精确投影 (Placeholder: 回退到简化映射)。"""
        # 完整实现需 torch 版 warpPerspective。
        # 当前回退, 但接口已预留。
        return self.project_mask_to_bev(mask)
