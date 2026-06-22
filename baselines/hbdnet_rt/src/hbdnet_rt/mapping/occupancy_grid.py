"""
BEV 占用栅格生成器。
输入感知输出 → 投影融合 → 二值占用 (0=可通行, 1=占用)。
"""
import torch
import torch.nn.functional as F
from typing import Dict, Optional
from .bev_projector import BEVProjector


class OccupancyGrid:
    """
    局部 BEV 二值占用栅格。

    规则 (按优先级):
      1. hard_boundary 区域 → 占用 (1)
      2. ego_passable 外部区域 → 占用 (1)
      3. detection 边界框区域 → 占用 (1), 带安全膨胀
      4. ego_passable 内部区域 → 可通行 (0)

    输出: [1, 1, ny, nx] 二值 tensor
    """

    def __init__(self, config):
        self.projector = BEVProjector(config)
        self.nx, self.ny = self.projector.grid_shape
        planner_cfg = config.planner.get("dwa", {})
        self.vehicle_width = planner_cfg.get("vehicle_width_m", 2.0)
        self.vehicle_length = planner_cfg.get("vehicle_length_m", 3.0)
        self.safety_margin = planner_cfg.get("safety_margin_m", 0.25)

    # ── 主入口 ──

    def generate(self, perception_output: Dict) -> Dict:
        """
        生成占用栅格。
        输入: 感知后处理的输出字典。
        输出: {"occupancy_grid": [1,1,ny,nx], "metadata": {...}}
        """
        ego = perception_output.get("ego_passable_mask")
        hard_b = perception_output.get("hard_boundary_mask")
        detections = perception_output.get("detections", {})

        # 初始化: 全部占用 (最保守)
        occup = torch.ones(1, 1, self.ny, self.nx)

        # 规则 2: ego_passable 内部 → 可通行
        if ego is not None and ego.numel() > 0:
            ego_bev = self.projector.project_mask_to_bev(ego)
            # ego_passable=1 的区域 → 标记为可通行 (occup=0)
            # 注意: ego 已经是二值 mask (后处理做了 threshold)
            passable = (ego_bev > 0.5).float()
            # passable=1 的区域 occup=0
            occup = occup * (1.0 - passable)

        # 规则 1: hard_boundary → 强制占用 (覆盖规则2)
        if hard_b is not None and hard_b.numel() > 0:
            hb_bev = self.projector.project_mask_to_bev(hard_b)
            # 取所有通道的 max (任一边界类别)
            hb_max = hb_bev.max(dim=1, keepdim=True)[0]
            hb_binary = (hb_max > 0.5).float()
            # hard_boundary 区域强制 occup=1
            occup = torch.maximum(occup, hb_binary)

        # 规则 3: detection 边界框 → 占用
        occup = self._add_detection_occupancy(occup, detections)

        return {
            "occupancy_grid": occup,
            "metadata": self.projector.grid_extent,
        }

    # ── Detection → 占用 ──

    def _add_detection_occupancy(self, occup: torch.Tensor,
                                  detections: Dict) -> torch.Tensor:
        """
        将检测框投影为栅格占用。简化版: 按框大小/位置扩张圆形区域。
        """
        boxes = detections.get("boxes")
        if boxes is None or boxes.numel() == 0:
            return occup

        # 将每个 box 底部中心投影到 grid, 膨胀圆形区域
        _, _, gh, gw = occup.shape[0], occup.shape[1], self.ny, self.nx
        dilation_px = int((self.vehicle_width / 2 + self.safety_margin) / self.projector.resolution)

        for i in range(min(boxes.shape[0], 20)):  # 最多处理 20 个框
            x1, y1, x2, y2 = boxes[i].tolist()
            # 底部中心 (地面投影更可靠)
            cx_img = (x1 + x2) / 2
            by_img = y2  # 框底部
            # 投影到 grid 坐标
            gx, gy = self.projector.image_xy_to_grid_xy(
                cx_img * 160 / 640,   # 假设 mask 尺寸 160×96 → 归一化
                by_img * 96 / 384,
                image_h=96, image_w=160)
            # 地面坐标 → grid index
            gx_idx = int((gx - self.projector.range_x[0]) / self.projector.resolution)
            gy_idx = int((gy - self.projector.range_y[0]) / self.projector.resolution)
            # 确保在范围内
            gx_idx = max(0, min(gh - 1, gx_idx))
            gy_idx = max(0, min(gw - 1, gy_idx))
            # 膨胀 (简化: 方形)
            x0 = max(0, gx_idx - dilation_px)
            x1 = min(gh, gx_idx + dilation_px)
            y0 = max(0, gy_idx - dilation_px)
            y1 = min(gw, gy_idx + dilation_px)
            if x0 < x1 and y0 < y1:
                occup[0, 0, x0:x1, y0:y1] = 1.0

        return occup
