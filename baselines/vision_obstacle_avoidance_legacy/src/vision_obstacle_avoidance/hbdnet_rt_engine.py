"""
HBD-Net-RT 推理引擎 wrapper。
将 HBD-Net-RT (RepVGG-lite + 5 Head) 封装为与 HybridNetsEngine 兼容的接口，
供 main.py 作为 DL 感知层直接调用。

HBD-Net-RT 输出 → 主循环期望的 HybridNetsOutput 映射:
  - ego_passable_mask → drivable_mask (可行驶区域)
  - hard_boundary_mask → lane_mask (硬边界/车道线)
  - detections → bboxes (检测框)
  - confidence.overall → confidence (综合置信度)

当前使用随机权重 — 训练数据到位后替换为训练好的权重即可。
"""

import logging
from dataclasses import dataclass, field
from typing import Optional
import sys
import os

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# HBDNet-RT 路径
_HBDNET_ROOT = os.path.join(os.path.dirname(__file__), '..', 'hbdnet_rt')
_HBDNET_SRC = os.path.join(_HBDNET_ROOT, 'src')

# ── 输出数据结构 (与 HybridNetsOutput 兼容) ──────────────────────────────


@dataclass
class HBDNetOutput:
    """HBD-Net-RT 推理结果，字段与 HybridNetsOutput 对齐"""
    bboxes: list = field(default_factory=list)
    drivable_mask: Optional[np.ndarray] = None      # (H, W) uint8, 0/255
    lane_mask: Optional[np.ndarray] = None           # (H, W) uint8
    confidence: float = 0.0

    # HBDNet 特有: 额外输出供高级模块使用
    hard_boundary_mask: Optional[np.ndarray] = None  # 硬边界多类 mask
    surface_risk_map: Optional[np.ndarray] = None    # 路面风险图
    raw_detections: Optional[dict] = None            # 原始检测结果


class HBDNetRTEngine:
    """
    HBD-Net-RT 推理封装，接口与 HybridNetsEngine 保持一致。

    使用方式:
      engine = HBDNetRTEngine(use_gpu=True)
      output = engine.infer(frame)   # 返回 HBDNetOutput
    """

    # 模型输入尺寸
    INPUT_WIDTH = 640
    INPUT_HEIGHT = 384

    def __init__(self, use_gpu: bool = True,
                 conf_threshold: float = 0.3,
                 iou_threshold: float = 0.45):
        """
        初始化 HBD-Net-RT 引擎。

        参数:
          use_gpu: 是否使用 GPU 推理 (需要 PyTorch CUDA)
          conf_threshold: 检测置信度阈值
          iou_threshold: NMS IoU 阈值
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self._model = None
        self._preprocessor = None
        self._postprocessor = None
        self._device = None

        try:
            import torch
            self._torch = torch

            # 添加 HBDNet src 到 path
            if _HBDNET_SRC not in sys.path:
                sys.path.insert(0, _HBDNET_SRC)

            from hbdnet_rt.perception.model import HBDNetRT
            from hbdnet_rt.perception.postprocess import PostProcessor
            from hbdnet_rt.utils.config import load_config

            # 设备选择
            if use_gpu and torch.cuda.is_available():
                self._device = torch.device('cuda')
            else:
                self._device = torch.device('cpu')
                if use_gpu:
                    logger.warning("GPU 不可用，回退到 CPU")

            # 加载模型 (随机权重)
            self._model = HBDNetRT()
            self._model.to(self._device)
            self._model.eval()

            # 后处理器
            hbd_cfg = load_config()
            self._postprocessor = PostProcessor(hbd_cfg)

            param_count = sum(p.numel() for p in self._model.parameters())
            logger.info("HBD-Net-RT 模型已加载 (%.1fM 参数, device=%s, 随机权重)",
                        param_count / 1e6, self._device)

        except ImportError as e:
            logger.error("HBD-Net-RT 依赖缺失: %s。请确保 PyTorch 已安装。", e)
            raise
        except Exception as e:
            logger.error("HBD-Net-RT 加载失败: %s", e)
            raise

    # ── 公开接口 ──────────────────────────────────────────────────────────

    def infer(self, frame: np.ndarray) -> HBDNetOutput:
        """
        对单帧执行 HBD-Net-RT 推理。

        参数:
          frame: BGR 图像 (H, W, 3), uint8, 任意尺寸

        返回:
          HBDNetOutput
        """
        if self._model is None:
            return HBDNetOutput()

        import torch

        orig_h, orig_w = frame.shape[:2]

        # 1. 预处理: resize + CLAHE + normalize → tensor
        input_tensor = self._preprocess(frame)

        # 2. 模型推理
        with torch.no_grad():
            raw = self._model(input_tensor.to(self._device))

        # 3. 后处理
        processed = self._postprocessor.process(raw)

        # 4. 转换为 HybridNets 兼容格式
        return self._to_output(processed, orig_w, orig_h)

    # ── 内部 ──────────────────────────────────────────────────────────────

    def _preprocess(self, frame: np.ndarray):
        """预处理: letterbox resize + CLAHE + normalize。"""
        h, w = frame.shape[:2]

        # Letterbox resize (保持宽高比)
        scale = min(self.INPUT_WIDTH / w, self.INPUT_HEIGHT / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        canvas = np.full((self.INPUT_HEIGHT, self.INPUT_WIDTH, 3), 128, dtype=np.uint8)
        y_off = (self.INPUT_HEIGHT - new_h) // 2
        x_off = (self.INPUT_WIDTH - new_w) // 2
        canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized

        # CLAHE 增强 (LAB L 通道)
        lab = cv2.cvtColor(canvas, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # BGR → RGB → tensor [0, 1]
        rgb = enhanced[..., ::-1].astype(np.float32) / 255.0
        tensor = self._torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
        return tensor

    def _to_output(self, processed: dict, orig_w: int, orig_h: int) -> HBDNetOutput:
        """将 HBDNet 后处理结果转为 HybridNets 兼容格式。"""
        import torch

        output = HBDNetOutput()

        # ── 置信度 ──
        output.confidence = float(processed["confidence"].get("overall", 0.5))

        # ── 可行驶区域 mask (ego_passable) ──
        ego = processed.get("ego_passable_mask")
        if ego is not None:
            if isinstance(ego, torch.Tensor):
                ego = ego.squeeze().cpu().numpy()
            ego_resized = cv2.resize(
                (ego * 255).astype(np.uint8), (orig_w, orig_h),
                interpolation=cv2.INTER_NEAREST)
            output.drivable_mask = ego_resized

        # ── 车道线/硬边界 mask (hard_boundary) ──
        hb = processed.get("hard_boundary_mask")
        if hb is not None:
            if isinstance(hb, torch.Tensor):
                # hb shape: (1, 5, H/4, W/4) — 4类 + binary
                hb = hb.squeeze(0)  # (5, H4, W4)
                # 取二值化层（最后一层）
                hb_bin = hb[-1].cpu().numpy()  # (H4, W4)
            else:
                hb_bin = hb
            hb_resized = cv2.resize(
                (hb_bin * 255).astype(np.uint8), (orig_w, orig_h),
                interpolation=cv2.INTER_NEAREST)
            output.lane_mask = hb_resized
            output.hard_boundary_mask = hb_resized

        # ── 检测框 ──
        detections = processed.get("detections", {})
        boxes = detections.get("boxes")
        scores = detections.get("scores")
        labels = detections.get("labels")

        if boxes is not None and boxes.numel() > 0:
            if isinstance(boxes, torch.Tensor):
                boxes_np = boxes.cpu().numpy()
                scores_np = scores.cpu().numpy() if isinstance(scores, torch.Tensor) else scores
                labels_np = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels
            else:
                boxes_np, scores_np, labels_np = boxes, scores, labels

            # 坐标缩放: 模型输出是 640×384 坐标系 → 原图坐标系
            scale_x = orig_w / self.INPUT_WIDTH
            scale_y = orig_h / self.INPUT_HEIGHT

            bboxes = []
            for i in range(min(len(boxes_np), 50)):
                x1, y1, x2, y2 = boxes_np[i]
                score = float(scores_np[i])
                if score < self.conf_threshold:
                    continue
                bboxes.append((
                    float(x1 * scale_x), float(y1 * scale_y),
                    float(x2 * scale_x), float(y2 * scale_y),
                    int(labels_np[i]), score,
                ))
            output.bboxes = bboxes
            output.raw_detections = {"boxes": boxes_np, "scores": scores_np, "labels": labels_np}

        # ── 路面风险图 ──
        srm = processed.get("surface_risk_map")
        if srm is not None:
            if isinstance(srm, torch.Tensor):
                srm = srm.squeeze().cpu().numpy()
            output.surface_risk_map = srm

        return output
