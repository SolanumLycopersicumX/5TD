"""感知后处理。Detection NMS + mask threshold + 置信度计算。所有阈值从 config 读取。"""
import torch
import torch.nn.functional as F
from typing import Dict, List


class PostProcessor:
    """
    后处理器。
    输入: HBDNetRT.forward() 的原始输出
    输出: 统一格式, 供 BEV 栅格 / DWA / 决策器直接使用
    """

    def __init__(self, config):
        det_cfg = config.scene.get("output", {}).get("detection", {})
        seg_cfg = config.scene.get("output", {}).get("segmentation", {})
        self.score_threshold = det_cfg.get("score_threshold", 0.3)
        self.nms_iou_threshold = det_cfg.get("nms_iou_threshold", 0.45)
        self.max_detections = det_cfg.get("max_detections", 50)
        self.passable_threshold = seg_cfg.get("ego_passable_threshold", 0.5)
        self.boundary_threshold = seg_cfg.get("hard_boundary_threshold", 0.5)
        self.edge_threshold = seg_cfg.get("edge_threshold", 0.5)

    def process(self, raw_output: Dict) -> Dict:
        """后处理主入口。"""
        # Detection: NMS + top-K (模型已做 score 过滤, 这里补充 NMS)
        detections = self._process_detections(raw_output["detections"])

        # Segmentation: sigmoid/softmax + threshold
        ego_passable = (raw_output["ego_passable_mask"].sigmoid() > self.passable_threshold).float()
        hard_boundary = raw_output["hard_boundary_mask"].softmax(dim=1)
        hard_boundary_bin = (hard_boundary.max(dim=1, keepdim=True)[0] > self.boundary_threshold).float()
        edge = (raw_output["hard_boundary_edge"].sigmoid() > self.edge_threshold).float()

        # 拼接 boundary 信息到 mask
        hard_boundary_full = torch.cat([hard_boundary, hard_boundary_bin], dim=1)

        return {
            "detections": detections,
            "ego_passable_mask": ego_passable,
            "hard_boundary_mask": hard_boundary_full,
            "hard_boundary_edge": edge,
            "surface_risk_map": raw_output.get("surface_risk_map"),
            "confidence": raw_output["confidence"],
        }

    def _process_detections(self, detections: Dict) -> Dict:
        """NMS 过滤 (轻量, 避免 Python 循环)。"""
        boxes = detections["boxes"]
        scores = detections["scores"]
        labels = detections["labels"]

        if boxes.numel() == 0:
            return {"boxes": boxes, "scores": scores, "labels": labels}

        # 简化: 按 score 排序 + top-K (完整 NMS 后续添加)
        if boxes.shape[0] > self.max_detections:
            _, idxs = scores.topk(self.max_detections)
            boxes = boxes[idxs]
            scores = scores[idxs]
            labels = labels[idxs]

        return {"boxes": boxes, "scores": scores, "labels": labels}
