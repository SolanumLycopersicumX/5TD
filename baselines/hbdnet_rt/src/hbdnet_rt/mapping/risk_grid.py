"""
BEV 风险栅格生成器。
在占用栅格基础上, 按障碍类别和置信度分配 0~1 连续风险值。
"""
import torch
from typing import Dict, Optional
from .bev_projector import BEVProjector
from .occupancy_grid import OccupancyGrid


# 默认类别风险映射: label index → base_risk
DEFAULT_CLASS_RISK = {
    0: 0.95,  # construction_vehicle → 极高
    1: 0.95,  # worker → 极高 (最优先保护)
    2: 0.80,  # suspended_object → 高
    3: 0.65,  # falling_debris → 中高
}


class RiskGrid:
    """
    局部 BEV 连续风险栅格 (0~1)。

    风险叠加规则 (优先级从高到低):
      1. hard_boundary 区域 → risk = 1.0 (强制, 不可通行)
      2. ego_passable 外部 → risk = 1.0
      3. detection 区域 → 按类别 base_risk, 带安全膨胀
      4. 低置信度 → 全图风险提升 (加性, 体现不确定性)
      5. 其余区域 → risk = 0.0 (可通行)
    """

    def __init__(self, config):
        self.projector = BEVProjector(config)
        self.nx, self.ny = self.projector.grid_shape
        planner_cfg = config.planner.get("dwa", {})
        safety_cfg = config.safety.get("safety", {})
        self.vehicle_width = planner_cfg.get("vehicle_width_m", 2.0)
        self.safety_margin = planner_cfg.get("safety_margin_m", 0.25)
        self.conf_limits = safety_cfg.get("confidence", {})
        self.class_risk = DEFAULT_CLASS_RISK.copy()

    # ── 主入口 ──

    def generate(self, perception_output: Dict,
                 occupancy_grid: Optional[torch.Tensor] = None) -> Dict:
        """
        生成风险栅格。
        输出: {"risk_grid": [1,1,ny,nx], "max_risk": float, "metadata": {...}}
        """
        ego = perception_output.get("ego_passable_mask")
        hard_b = perception_output.get("hard_boundary_mask")
        detections = perception_output.get("detections", {})
        confidence = perception_output.get("confidence", {})

        # 初始化: 基于 occupancy (如果没有传入则从 ego 构建)
        if occupancy_grid is not None:
            risk = occupancy_grid.clone().float()
        else:
            risk = torch.zeros(1, 1, self.ny, self.nx)

        # ---- 规则 1: hard_boundary → risk = 1.0 ----
        if hard_b is not None and hard_b.numel() > 0:
            hb_bev = self.projector.project_mask_to_bev(hard_b)
            hb_max = hb_bev.max(dim=1, keepdim=True)[0]
            risk = torch.maximum(risk, (hb_max > 0.5).float())

        # ---- 规则 2: ego_passable 外部 → risk = 1.0 ----
        if ego is not None and ego.numel() > 0:
            ego_bev = self.projector.project_mask_to_bev(ego)
            passable = (ego_bev > 0.5).float()
            # 不可通行 = 1 - passable
            impassable = 1.0 - passable
            risk = torch.maximum(risk, impassable)

        # ---- 规则 3: detection → 按类别 ----
        risk = self._add_detection_risk(risk, detections)

        # ---- 规则 4: 低置信度 → 全图风险提升 ----
        risk = self._apply_confidence_bias(risk, confidence)

        # clamp
        risk = risk.clamp(0.0, 1.0)
        max_risk = risk.max().item()

        return {
            "risk_grid": risk,
            "max_risk": max_risk,
            "metadata": self.projector.grid_extent,
        }

    # ── Detection 风险 ──

    def _add_detection_risk(self, risk: torch.Tensor,
                             detections: Dict) -> torch.Tensor:
        """将检测框按类别和位置叠加风险值。"""
        boxes = detections.get("boxes")
        labels = detections.get("labels")
        scores = detections.get("scores")

        if boxes is None or boxes.numel() == 0:
            return risk

        _, _, gh, gw = risk.shape
        # 膨胀半径 (基于车宽+余量)
        dilate_px = int((self.vehicle_width / 2 + self.safety_margin) / self.projector.resolution)
        dilate_px = max(2, dilate_px)

        for i in range(min(boxes.shape[0], 20)):
            x1, y1, x2, y2 = boxes[i].tolist()
            # 框底部中心 → 地面投影
            cx_img = (x1 + x2) / 2
            by_img = y2
            gx, gy = self.projector.image_xy_to_grid_xy(
                cx_img * 160 / 640,
                by_img * 96 / 384,
                image_h=96, image_w=160)
            gx_idx = int((gx - self.projector.range_x[0]) / self.projector.resolution)
            gy_idx = int((gy - self.projector.range_y[0]) / self.projector.resolution)
            gx_idx = max(0, min(gh - 1, gx_idx))
            gy_idx = max(0, min(gw - 1, gy_idx))

            # 类别风险
            label = int(labels[i].item()) if labels.numel() > 0 else 0
            score = float(scores[i].item()) if scores.numel() > 0 else 0.5
            base = self.class_risk.get(label, 0.7)
            # 高 score → 接近 base, 低 score → 在 base 基础上增加不确定性
            obj_risk = base * score + 0.4 * (1.0 - score)

            # 膨胀区域取 max(当前, obj_risk)
            x0 = max(0, gx_idx - dilate_px)
            x1 = min(gh, gx_idx + dilate_px)
            y0 = max(0, gy_idx - dilate_px)
            y1 = min(gw, gy_idx + dilate_px)
            if x0 < x1 and y0 < y1:
                patch = risk[0, 0, x0:x1, y0:y1]
                risk[0, 0, x0:x1, y0:y1] = torch.maximum(patch,
                    torch.full_like(patch, obj_risk))

        return risk

    # ── 置信度偏置 ──

    def _apply_confidence_bias(self, risk: torch.Tensor,
                                confidence: Dict) -> torch.Tensor:
        """
        低置信度 → 全图风险提升。
        体现: 感知不可靠时, 即使是"可通行"区域也要提高警惕。
        """
        overall = confidence.get("overall", 0.85)
        # overall < 0.5 → bias up to 0.3
        if overall < 0.5:
            bias = (0.5 - overall) * 0.6  # max ~0.3 at conf=0
            risk = risk + bias
        return risk
