"""
HBD-Net-RT v1.0 感知模型。
RepVGG-lite Backbone + Lightweight FPN + 多任务 Head。
第一版输出: detections, ego_passable_mask, hard_boundary_mask,
           hard_boundary_edge, surface_risk_map。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple


# ═══════════════════════════════════════════════════════════════
#  RepVGG-lite Backbone
# ═══════════════════════════════════════════════════════════════

class RepVGGBlock(nn.Module):
    """
    RepVGG 基础模块。
    训练时: 3×3 conv + 1×1 conv + identity (可选) → 三条分支融合。
    推理时: 重参数化为单个 3×3 conv。当前实现直接使用融合后的单路形式。
    """
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1,
                 use_residual: bool = False):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=stride,
                              padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.use_residual = use_residual and (stride == 1 and in_ch == out_ch)

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        if self.use_residual:
            out = out + x
        return out


class RepVGGBackbone(nn.Module):
    """
    RepVGG-lite Backbone。
    输出 P3/P4/P5 三级特征, strides [8, 16, 32]。
    各 stage 的通道数和层数可配置, 默认为轻量版 (~3M 参数)。
    """
    def __init__(self, in_channels: int = 3,
                 stages_config: List[Tuple[int, int, int]] = None):
        """
        stages_config: [(out_ch, num_blocks, stride_first_block), ...]
        默认: stage0=stem, stage1-4 对应 1/4, 1/8, 1/16, 1/32
        """
        super().__init__()
        if stages_config is None:
            # 轻量配置: ~3M 参数
            stages_config = [
                (48,  2, 2),   # /4  → 作为 stem 的输出, 不进入 FPN
                (64,  2, 2),   # /8  → P3
                (128, 3, 2),   # /16 → P4
                (256, 2, 2),   # /32 → P5
            ]

        self.out_channels = []
        self._strides = []
        in_ch = in_channels

        # Stem: 3×3 conv /2
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, stages_config[0][0], 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stages_config[0][0]),
            nn.ReLU(inplace=True),
        )

        # Stages
        self.stages = nn.ModuleList()
        prev_ch = stages_config[0][0]
        current_stride = 4

        for i, (out_ch, num_blocks, stride) in enumerate(stages_config):
            blocks = []
            # 第一块做下采样
            blocks.append(RepVGGBlock(prev_ch, out_ch, stride=stride))
            # 其余块保持分辨率
            for _ in range(num_blocks - 1):
                blocks.append(RepVGGBlock(out_ch, out_ch, stride=1,
                                          use_residual=True))
            self.stages.append(nn.Sequential(*blocks))
            prev_ch = out_ch
            current_stride *= stride
            if current_stride >= 8:
                self.out_channels.append(out_ch)
                self._strides.append(current_stride)

        # 只保留最后 3 个 stage (P3/P4/P5)
        self.out_channels = self.out_channels[-3:]
        self._strides = self._strides[-3:]

    @property
    def strides(self) -> List[int]:
        return self._strides

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.stem(x)
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        # 返回最后 3 级 (P3, P4, P5)
        return features[-3:]


# ═══════════════════════════════════════════════════════════════
#  Neck
# ═══════════════════════════════════════════════════════════════

class LightweightFPN(nn.Module):
    """轻量 FPN。自顶向下 + 横向连接, 输出统一通道数的多尺度特征。"""
    def __init__(self, in_channels: List[int] = None, fpn_channels: int = 128):
        super().__init__()
        if in_channels is None:
            in_channels = [64, 128, 256]
        self.fpn_channels = fpn_channels
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(ch, fpn_channels, 1, bias=False) for ch in in_channels
        ])
        self.output_convs = nn.ModuleList([
            nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1, bias=False)
            for _ in in_channels
        ])
        self.bns = nn.ModuleList([
            nn.BatchNorm2d(fpn_channels) for _ in in_channels
        ])

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]
        for i in range(len(laterals) - 1, 0, -1):
            up = F.interpolate(laterals[i], size=laterals[i-1].shape[2:],
                              mode='nearest')
            laterals[i-1] = laterals[i-1] + up
        return [bn(conv(lat))
                for conv, lat, bn in zip(self.output_convs, laterals, self.bns)]


# ═══════════════════════════════════════════════════════════════
#  Detection Head (Anchor-free)
# ═══════════════════════════════════════════════════════════════

class AnchorFreeDetectionHead(nn.Module):
    """Anchor-free 检测头。每层: 2 层共享卷积 → 分类 + 回归预测。"""
    def __init__(self, fpn_channels: int = 128, num_classes: int = 4):
        super().__init__()
        self.num_classes = num_classes
        self.cls_convs = nn.ModuleList([
            nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1, bias=False)
            for _ in range(2)])
        self.reg_convs = nn.ModuleList([
            nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1, bias=False)
            for _ in range(2)])
        self.cls_pred = nn.Conv2d(fpn_channels, num_classes, 1)
        self.reg_pred = nn.Conv2d(fpn_channels, 4, 1)
        nn.init.constant_(self.cls_pred.bias, -4.0)

    def forward(self, features: List[torch.Tensor]) -> Tuple[List, List]:
        cls_outs, reg_outs = [], []
        for feat in features:
            cf, rf = feat, feat
            for cc, rc in zip(self.cls_convs, self.reg_convs):
                cf, rf = F.relu(cc(cf)), F.relu(rc(rf))
            cls_outs.append(self.cls_pred(cf))
            reg_outs.append(self.reg_pred(rf))
        return cls_outs, reg_outs


# ═══════════════════════════════════════════════════════════════
#  Segmentation Heads
# ═══════════════════════════════════════════════════════════════

class SegmentationHead(nn.Module):
    """通用分割头: 多尺度融合 → 2 层卷积 → 预测。"""
    def __init__(self, fpn_channels: int = 128, num_classes: int = 1):
        super().__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(fpn_channels)
        self.conv2 = nn.Conv2d(fpn_channels, fpn_channels // 2, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(fpn_channels // 2)
        self.pred = nn.Conv2d(fpn_channels // 2, num_classes, 1)
        nn.init.constant_(self.pred.bias, 0.0)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        out = features[0]
        for feat in features[1:]:
            up = F.interpolate(feat, size=out.shape[2:],
                              mode='bilinear', align_corners=False)
            out = out + up
        x = F.relu(self.bn1(self.conv1(out)))
        x = F.relu(self.bn2(self.conv2(x)))
        return self.pred(x)


# ═══════════════════════════════════════════════════════════════
#  Surface Risk Head (New)
# ═══════════════════════════════════════════════════════════════

class SurfaceRiskHead(nn.Module):
    """
    路面风险估计头。
    输出低分辨率 risk map (1/16 输入尺寸), 估计路面凹凸/碎石/积水等
    非语义障碍物的风险。与 segmentation mask 互补:
    - segmentation: "这是什么类别"
    - surface_risk: "这里能不能安全通过"
    """
    def __init__(self, fpn_channels: int = 128):
        super().__init__()
        # 使用 P4 (1/16) 作为主特征层，兼顾语义和空间细节
        self.conv1 = nn.Conv2d(fpn_channels, fpn_channels // 2, 3,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(fpn_channels // 2)
        self.conv2 = nn.Conv2d(fpn_channels // 2, 32, 3,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.pred = nn.Conv2d(32, 1, 1)
        nn.init.constant_(self.pred.bias, -1.0)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        # 取 P4 (index 1), 分辨率 1/16
        x = features[min(1, len(features) - 1)]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return self.pred(x)


# ═══════════════════════════════════════════════════════════════
#  Anchor-Free Decoder
# ═══════════════════════════════════════════════════════════════

class AnchorFreeDecoder:
    """将多尺度检测头输出解码为 boxes/scores/labels。"""

    def __init__(self, strides: List[int], score_threshold: float = 0.3,
                 max_detections: int = 50):
        self.strides = strides
        self.score_threshold = score_threshold
        self.max_detections = max_detections

    def __call__(self, cls_outs, reg_outs) -> Dict:
        all_boxes, all_scores, all_labels = [], [], []
        device = cls_outs[0].device

        for stride, cls_out, reg_out in zip(self.strides, cls_outs, reg_outs):
            B, C, H, W = cls_out.shape
            grid_y, grid_x = torch.meshgrid(
                torch.arange(H, device=device, dtype=torch.float32),
                torch.arange(W, device=device, dtype=torch.float32),
                indexing='ij')
            cx = (grid_x + 0.5) * stride
            cy = (grid_y + 0.5) * stride

            cls_scores, cls_labels = cls_out.flatten(2).permute(0, 2, 1).max(dim=-1)
            reg = reg_out.permute(0, 2, 3, 1)
            x1 = cx - reg[..., 0]
            y1 = cy - reg[..., 1]
            x2 = cx + reg[..., 2]
            y2 = cy + reg[..., 3]
            boxes = torch.stack([x1, y1, x2, y2], dim=-1)

            all_boxes.append(boxes.reshape(B, -1, 4))
            all_scores.append(cls_scores.sigmoid())
            all_labels.append(cls_labels)

        boxes = torch.cat(all_boxes, dim=1)[0]
        scores = torch.cat(all_scores, dim=1)[0]
        labels = torch.cat(all_labels, dim=1)[0]

        keep = scores > self.score_threshold
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
        if boxes.shape[0] > self.max_detections:
            _, topk = scores.topk(self.max_detections)
            boxes, scores, labels = boxes[topk], scores[topk], labels[topk]

        return {"boxes": boxes, "scores": scores, "labels": labels}


# ═══════════════════════════════════════════════════════════════
#  HBD-Net-RT v1.0
# ═══════════════════════════════════════════════════════════════

class HBDNetRT(nn.Module):
    """
    HBD-Net-RT v1.0 — 隧道施工半幅通行感知模型。

    结构: RepVGG-lite Backbone → Lightweight FPN → 5 Heads
    输入:  B × 3 × 384 × 640
    输出:  detections, ego_passable_mask, hard_boundary_mask,
           hard_boundary_edge, surface_risk_map, confidence
    """

    def __init__(self, num_classes: int = 4, fpn_channels: int = 128):
        super().__init__()
        self.num_classes = num_classes
        self._input_h, self._input_w = 384, 640

        # 1. Backbone
        self.backbone = RepVGGBackbone(in_channels=3)

        # 2. Neck
        self.neck = LightweightFPN(
            in_channels=self.backbone.out_channels,
            fpn_channels=fpn_channels)

        # 3. Multi-task Heads
        self.detection_head = AnchorFreeDetectionHead(
            fpn_channels=fpn_channels, num_classes=num_classes)
        self.passable_head = SegmentationHead(
            fpn_channels=fpn_channels, num_classes=1)
        self.boundary_head = SegmentationHead(
            fpn_channels=fpn_channels, num_classes=4)
        self.edge_head = SegmentationHead(
            fpn_channels=fpn_channels, num_classes=1)
        self.surface_risk_head = SurfaceRiskHead(
            fpn_channels=fpn_channels)

        # 4. Decoder
        self.detection_decoder = AnchorFreeDecoder(
            strides=self.backbone.strides)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                       nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @property
    def input_shape(self) -> Tuple[int, int, int]:
        return (3, self._input_h, self._input_w)

    @property
    def output_shape(self) -> Tuple[int, int]:
        return (self._input_h // 4, self._input_w // 4)

    def forward(self, x: torch.Tensor) -> Dict:
        B = x.shape[0]
        out_h, out_w = self.output_shape

        # Backbone + Neck
        features = self.neck(self.backbone(x))

        # Detection
        cls_outs, reg_outs = self.detection_head(features)
        detections = self.detection_decoder(cls_outs, reg_outs)

        # Segmentation — 上采样到 1/4
        ego_passable = F.interpolate(
            self.passable_head(features), size=(out_h, out_w),
            mode='bilinear', align_corners=False)
        hard_boundary = F.interpolate(
            self.boundary_head(features), size=(out_h, out_w),
            mode='bilinear', align_corners=False)
        edge = F.interpolate(
            self.edge_head(features), size=(out_h, out_w),
            mode='bilinear', align_corners=False)

        # Surface Risk — 保持低分辨率 (1/16), 不放大
        surface_risk = self.surface_risk_head(features)

        # Confidence
        det_conf = (detections["scores"].mean().item()
                    if detections["scores"].numel() > 0 else 0.5)
        passable_conf = (ego_passable.sigmoid() > 0.5).float().mean().item()
        boundary_conf = (hard_boundary.softmax(dim=1).max(dim=1)[0] > 0.5).float().mean().item()
        risk_conf = surface_risk.sigmoid().mean().item()
        overall = 0.3 * det_conf + 0.25 * passable_conf + 0.25 * boundary_conf + 0.2 * risk_conf

        return {
            "detections": detections,
            "ego_passable_mask": ego_passable,
            "hard_boundary_mask": hard_boundary,
            "hard_boundary_edge": edge,
            "surface_risk_map": surface_risk,
            "confidence": {
                "detection": round(det_conf, 4),
                "passable": round(passable_conf, 4),
                "boundary": round(boundary_conf, 4),
                "surface_risk": round(risk_conf, 4),
                "overall": round(overall, 4),
            }
        }

    def get_model_info(self) -> Dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "name": "HBD-Net-RT",
            "version": "1.0",
            "backbone": "RepVGG-lite",
            "input_shape": self.input_shape,
            "output_shape": self.output_shape,
            "num_classes": self.num_classes,
            "total_params": total,
            "trainable_params": trainable,
            "status": "skeleton (random weights, RepVGG-lite backbone)",
        }
