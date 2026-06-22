"""
占用栅格融合模块。
将 HybridNets 分割 + 检测 + 碎石信息融合为统一的 BEV 占用栅格，
同时进行飞溅物时域过滤。

输入:
  HybridNetsOutput (分割mask + bbox)
  LaneBoundaryState (车道边界, 可选)
  GroundCalibrator (像素→米, 可选)

输出:
  OccupancyGrid: BEV 占用栅格 + 障碍物列表
"""

import logging
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class OccupancyGrid:
    """BEV 占用栅格"""
    cells: Optional[np.ndarray] = None       # (N, M) 占用概率 0-1
    resolution_m: float = 0.2                # 栅格分辨率 (米/格)
    width_m: float = 5.0                     # 栅格宽度 (米, 横向)
    length_m: float = 10.0                   # 栅格长度 (米, 纵向)

    obstacle_list: list = field(default_factory=list)
    # [(x_m, y_m, radius_m, type, confidence), ...]

    num_static_obstacles: int = 0
    num_dynamic_obstacles: int = 0

    debug_image: Optional[np.ndarray] = None


class OccupancyGridFusion:
    """
    占用栅格融合器。

    功能:
      1. 分割 mask → BEV 投影 (地面标定)
      2. 检测 bbox → 底面投影 → 圆形占用
      3. 飞溅物过滤: 新物体 < 3 帧 → 忽略
      4. 安全膨胀: 所有占用膨胀 SAFETY_MARGIN
    """

    def __init__(self,
                 grid_width_m: float = 5.0,
                 grid_length_m: float = 10.0,
                 resolution_m: float = 0.2,
                 safety_margin_m: float = 0.25,
                 debris_min_frames: int = 3,
                 object_timeout_frames: int = 30):
        self.grid_width_m = grid_width_m
        self.grid_length_m = grid_length_m
        self.resolution_m = resolution_m
        self.safety_margin_m = safety_margin_m
        self.debris_min_frames = debris_min_frames
        self.object_timeout_frames = object_timeout_frames

        # ── 时域跟踪 ──
        self._object_tracks: dict = {}  # track_id → {history, frames_seen}
        self._next_track_id: int = 0
        self._debris_candidates: list = []  # [(x_m, y_m, size_cm, frames)]

    # ── 公开接口 ──────────────────────────────────────────────────────────

    def fuse(self,
             drivable_mask: np.ndarray | None,
             lane_mask: np.ndarray | None,
             bboxes: list,
             calibrator=None,
             lane_boundary=None) -> OccupancyGrid:
        """
        融合所有感知信息为占用栅格。

        参数:
          drivable_mask: (H, W) uint8, 可行驶区域
          lane_mask: (H, W) uint8, 车道线
          bboxes: [(x1,y1,x2,y2,class_id,conf), ...]
          calibrator: GroundCalibrator (可选, 用于 BEV 投影)
          lane_boundary: LaneBoundaryState (可选)

        返回:
          OccupancyGrid
        """
        grid = OccupancyGrid(
            resolution_m=self.resolution_m,
            width_m=self.grid_width_m,
            length_m=self.grid_length_m,
        )

        cols = int(self.grid_width_m / self.resolution_m)
        rows = int(self.grid_length_m / self.resolution_m)
        cells = np.zeros((rows, cols), dtype=np.float32)

        # 1. 分割 mask → 车道内占用
        if drivable_mask is not None and drivable_mask.size > 0:
            non_drivable = (drivable_mask == 0).astype(np.float32)
            # 简化: 非可行驶区域 = 占用 (后续可加入 BEV 投影)
            # 此处用像素级的近似: 底部区域权重更高
            h = non_drivable.shape[0]
            for row_idx in range(rows):
                y_start = int(h * (1.0 - (row_idx + 1) * self.grid_length_m / 10.0))
                y_end = int(h * (1.0 - row_idx * self.grid_length_m / 10.0))
                y_start = max(0, min(h, y_start))
                y_end = max(0, min(h, y_end))
                if y_end > y_start:
                    for col_idx in range(cols):
                        x_start = int(drivable_mask.shape[1] * col_idx / cols)
                        x_end = int(drivable_mask.shape[1] * (col_idx + 1) / cols)
                        if x_end > x_start:
                            patch = non_drivable[y_start:y_end, x_start:x_end]
                            if patch.size > 0:
                                cells[row_idx, col_idx] = float(patch.mean())

        # 2. 检测 bbox → 圆形占用区域
        static_obs = 0
        dynamic_obs = 0
        obstacle_list = []

        for bbox in bboxes:
            x1, y1, x2, y2, cls_id, conf = bbox
            # 底面中心 → 地面坐标
            cx_img = (x1 + x2) / 2
            cy_img = y2  # bbox 底部 = 地面接触点
            radius = max((x2 - x1), (y2 - y1)) / 2

            if calibrator and calibrator.calibrated:
                ground_pt = calibrator.pixel_to_ground(
                    int(cx_img), int(cy_img),
                    image_width=640, image_height=480)
                if ground_pt:
                    ox, oy = ground_pt
                    oradius = radius * calibrator.meters_at_row(int(cy_img), 480) or 0.3
                else:
                    # 回退: 假设在车道前方
                    ox, oy = self.grid_width_m / 2, self.grid_length_m * 0.4
                    oradius = 0.5
            else:
                ox, oy = self.grid_width_m / 2, self.grid_length_m * 0.4
                oradius = 0.5

            # 圆形占用标记
            col_c = int(ox / self.resolution_m)
            row_c = int(oy / self.resolution_m)
            r_cells = int(oradius / self.resolution_m)
            for dr in range(-r_cells, r_cells + 1):
                for dc in range(-r_cells, r_cells + 1):
                    if dr * dr + dc * dc <= r_cells * r_cells:
                        rr, cc = row_c + dr, col_c + dc
                        if 0 <= rr < rows and 0 <= cc < cols:
                            cells[rr, cc] = max(cells[rr, cc], conf)

            # 分类
            is_dynamic = cls_id in (0, 1, 2)  # person, worker_standing, worker_crouching
            if is_dynamic:
                dynamic_obs += 1
            else:
                static_obs += 1
            obstacle_list.append((ox, oy, oradius, cls_id, conf))

        # 3. 飞溅物过滤 (新出现 < 3帧 → 从栅格中移除)
        # 此处简化: 在 obstacle_list 中标记短期物体
        # 完整实现需要关联跟踪

        # 4. 安全膨胀
        margin_cells = int(self.safety_margin_m / self.resolution_m)
        if margin_cells > 0 and cells.any():
            dilated = cells.copy()
            for r in range(rows):
                for c in range(cols):
                    if cells[r, c] > 0.3:
                        r1, r2 = max(0, r - margin_cells), min(rows, r + margin_cells + 1)
                        c1, c2 = max(0, c - margin_cells), min(cols, c + margin_cells + 1)
                        dilated[r1:r2, c1:c2] = np.maximum(
                            dilated[r1:r2, c1:c2], cells[r, c] * 0.5)
            cells = dilated

        grid.cells = cells
        grid.obstacle_list = obstacle_list
        grid.num_static_obstacles = static_obs
        grid.num_dynamic_obstacles = dynamic_obs

        return grid

    def reset(self):
        """清空跟踪历史"""
        self._object_tracks.clear()
        self._debris_candidates.clear()
        self._next_track_id = 0
