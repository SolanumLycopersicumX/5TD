"""
HybridNets ONNX 推理引擎。
加载 ONNX 模型，执行多任务推理（检测 + 分割 + 车道线），输出结构化结果。

后处理参考:
  datvuthanh/HybridNets utils/utils.py (BBoxTransform, Anchors, postprocess)
  适配 ONNX Runtime, 无需 PyTorch 依赖。

输入:  640×384 BGR (resized from camera)
输出:  HybridNetsOutput (bboxes + drivable_mask + lane_mask + confidence)
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ============================================================================
# 输出数据结构
# ============================================================================


@dataclass
class HybridNetsOutput:
    """HybridNets 多任务推理结果"""
    # 目标检测
    bboxes: list = field(default_factory=list)
    # [(x1, y1, x2, y2, class_id, confidence), ...] 原图坐标

    # 可行驶区域分割
    drivable_mask: Optional[np.ndarray] = None  # (H, W) uint8, 0=背景, 1=可行驶

    # 车道线检测
    lane_mask: Optional[np.ndarray] = None  # (H, W) uint8, 0=背景, >0=车道线类

    # 置信度 (分割分支的平均概率)
    confidence: float = 0.0

    # 原始输出 (调试用)
    raw_regression: Optional[np.ndarray] = None
    raw_classification: Optional[np.ndarray] = None
    raw_segmentation: Optional[np.ndarray] = None


# ============================================================================
# ONNX 推理引擎
# ============================================================================


class HybridNetsEngine:
    """
    HybridNets ONNX 推理封装。

    使用方式:
      engine = HybridNetsEngine("hybridnets.onnx")
      output = engine.infer(frame)
    """

    # 模型输入尺寸 (HybridNets 默认)
    INPUT_WIDTH = 640
    INPUT_HEIGHT = 384

    # ImageNet 归一化参数
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    # 检测参数
    CONF_THRESHOLD = 0.3
    IOU_THRESHOLD = 0.45

    # 锚框参数 (HybridNets EfficientNet-B3 默认)
    ANCHOR_SCALE = 4.0
    PYRAMID_LEVELS = [3, 4, 5, 6, 7]
    ANCHOR_SCALES = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
    ANCHOR_RATIOS = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]

    def __init__(self, onnx_path: str, use_gpu: bool = False,
                 conf_threshold: float = 0.3, iou_threshold: float = 0.45):
        """
        初始化推理引擎。

        参数:
          onnx_path: ONNX 模型文件路径
          use_gpu: 是否使用 GPU (需要 onnxruntime-gpu)
          conf_threshold: 检测置信度阈值
          iou_threshold: NMS IoU 阈值
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self._session = None
        self._anchors: Optional[np.ndarray] = None

        try:
            import onnxruntime as ort
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_gpu \
                else ['CPUExecutionProvider']
            self._session = ort.InferenceSession(onnx_path, providers=providers)
            logger.info("HybridNets ONNX 模型已加载: %s (providers=%s)",
                        onnx_path, self._session.get_providers())
        except ImportError:
            logger.error("onnxruntime 未安装。请运行: pip install onnxruntime")
            raise
        except Exception as e:
            logger.error("加载 ONNX 模型失败: %s", e)
            raise

        # 预生成锚框
        self._anchors = self._generate_anchors()

    # ── 公开接口 ──────────────────────────────────────────────────────────

    def infer(self, frame: np.ndarray) -> HybridNetsOutput:
        """
        对单帧执行 HybridNets 推理。

        参数:
          frame: BGR 图像 (H, W, 3), uint8, 任意尺寸

        返回:
          HybridNetsOutput
        """
        if self._session is None:
            return HybridNetsOutput()

        orig_h, orig_w = frame.shape[:2]

        # 1. 预处理
        input_tensor, meta = self._preprocess(frame)

        # 2. ONNX 推理
        outputs = self._session.run(None, {self._session.get_inputs()[0].name: input_tensor})

        # HybridNets ONNX 输出:
        #   [0] regression    (1, N_anchors, 4)  bbox 偏移
        #   [1] classification (1, N_anchors, N_classes) 类别概率
        #   [2] segmentation  (1, seg_classes+1, 384, 640) 分割 logits
        regression = outputs[0]
        classification = outputs[1]
        segmentation = outputs[2]

        # 3. 检测后处理
        bboxes = self._postprocess_detection(regression, classification, meta, orig_w, orig_h)

        # 4. 分割后处理
        drivable_mask, lane_mask, seg_conf = self._postprocess_segmentation(
            segmentation, orig_w, orig_h)

        # 5. 综合置信度
        detection_conf = np.mean([b[5] for b in bboxes]) if bboxes else 1.0
        confidence = 0.4 * detection_conf + 0.6 * seg_conf

        return HybridNetsOutput(
            bboxes=bboxes,
            drivable_mask=drivable_mask,
            lane_mask=lane_mask,
            confidence=float(confidence),
        )

    # ── 预处理 ─────────────────────────────────────────────────────────────

    def _preprocess(self, frame: np.ndarray) -> tuple:
        """预处理: resize → normalize → NCHW"""
        h, w = frame.shape[:2]

        # Resize to 640×384
        resized = cv2.resize(frame, (self.INPUT_WIDTH, self.INPUT_HEIGHT),
                             interpolation=cv2.INTER_AREA)

        # BGR → RGB → normalize
        rgb = resized[..., ::-1].astype(np.float32) / 255.0
        normalized = (rgb - self.MEAN) / self.STD

        # HWC → NCHW
        input_tensor = np.transpose(normalized, (2, 0, 1))[np.newaxis, ...].astype(np.float32)

        meta = {
            'orig_w': w, 'orig_h': h,
            'new_w': self.INPUT_WIDTH, 'new_h': self.INPUT_HEIGHT,
        }
        return input_tensor, meta

    # ── 检测后处理 ─────────────────────────────────────────────────────────

    def _postprocess_detection(self, regression: np.ndarray,
                                classification: np.ndarray,
                                meta: dict,
                                orig_w: int, orig_h: int) -> list:
        """
        检测后处理: 解码 + NMS + 坐标缩放。
        返回 [(x1, y1, x2, y2, class_id, confidence), ...] 原图坐标
        """
        if self._anchors is None:
            return []

        # NumPy 版本的 BBoxTransform
        anchors = self._anchors  # (1, N, 4)
        regression = regression[0]  # (N, 4)
        classification = classification[0]  # (N, C)

        # 解码 bbox
        y_centers_a = (anchors[0, :, 0] + anchors[0, :, 2]) / 2
        x_centers_a = (anchors[0, :, 1] + anchors[0, :, 3]) / 2
        ha = anchors[0, :, 2] - anchors[0, :, 0]
        wa = anchors[0, :, 3] - anchors[0, :, 1]

        w = np.exp(regression[:, 3]) * wa
        h = np.exp(regression[:, 2]) * ha
        y_centers = regression[:, 0] * ha + y_centers_a
        x_centers = regression[:, 1] * wa + x_centers_a

        ymin = y_centers - h / 2.0
        xmin = x_centers - w / 2.0
        ymax = y_centers + h / 2.0
        xmax = x_centers + w / 2.0

        boxes = np.stack([xmin, ymin, xmax, ymax], axis=1)

        # Clip
        boxes[:, 0] = np.clip(boxes[:, 0], 0, self.INPUT_WIDTH - 1)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, self.INPUT_HEIGHT - 1)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, self.INPUT_WIDTH - 1)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, self.INPUT_HEIGHT - 1)

        # 置信度 + 类别
        scores = classification.max(axis=1)
        class_ids = classification.argmax(axis=1)
        mask = scores > self.conf_threshold

        if not mask.any():
            return []

        boxes = boxes[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

        # NMS
        keep = self._nms(boxes, scores, self.iou_threshold)
        boxes = boxes[keep]
        scores = scores[keep]
        class_ids = class_ids[keep]

        # 坐标缩放回原图
        scale_x = orig_w / self.INPUT_WIDTH
        scale_y = orig_h / self.INPUT_HEIGHT
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y

        result = []
        for i in range(len(boxes)):
            result.append((
                float(boxes[i, 0]), float(boxes[i, 1]),
                float(boxes[i, 2]), float(boxes[i, 3]),
                int(class_ids[i]), float(scores[i]),
            ))
        return result

    # ── 分割后处理 ─────────────────────────────────────────────────────────

    def _postprocess_segmentation(self, segmentation: np.ndarray,
                                   orig_w: int, orig_h: int) -> tuple:
        """
        分割后处理: softmax → argmax → resize。
        返回 (drivable_mask, lane_mask, confidence)
        """
        # segmentation: (1, C+1, 384, 640)
        seg = segmentation[0]  # (C+1, 384, 640)

        # Softmax
        seg_exp = np.exp(seg - seg.max(axis=0, keepdims=True))
        seg_probs = seg_exp / seg_exp.sum(axis=0, keepdims=True)

        # Argmax
        seg_labels = seg_probs.argmax(axis=0).astype(np.uint8)  # (384, 640)

        # Resize 回原图
        seg_labels = cv2.resize(seg_labels, (orig_w, orig_h),
                                interpolation=cv2.INTER_NEAREST)

        # 分离 drivable 和 lane
        # 类别定义:
        #   0 = 背景
        #   1 = 可行驶路面 (drivable)
        #   2 = 车道线/隔离沟/隔离带 (lane)
        #   ...
        drivable_mask = (seg_labels == 1).astype(np.uint8) * 255
        lane_mask = np.where(seg_labels >= 2, seg_labels, 0).astype(np.uint8)

        # 置信度 = 可行驶区域像素的平均概率
        drivable_prob = seg_probs[1]  # 可行驶类别概率图 (384, 640)
        confidence = float(drivable_prob.mean())

        return drivable_mask, lane_mask, confidence

    # ── 锚框生成 ───────────────────────────────────────────────────────────

    def _generate_anchors(self) -> np.ndarray:
        """预生成所有锚框 (HybridNets Anchors 类的 NumPy 版本)"""
        all_boxes = []
        strides = [2 ** level for level in self.PYRAMID_LEVELS]

        for stride in strides:
            level_boxes = []
            for scale, ratio in zip(self.ANCHOR_SCALES, self.ANCHOR_RATIOS):
                base_size = self.ANCHOR_SCALE * stride * scale
                ax2 = base_size * ratio[0] / 2.0
                ay2 = base_size * ratio[1] / 2.0

                x = np.arange(stride / 2, self.INPUT_WIDTH, stride)
                y = np.arange(stride / 2, self.INPUT_HEIGHT, stride)
                xv, yv = np.meshgrid(x, y)
                xv, yv = xv.ravel(), yv.ravel()

                boxes = np.stack([
                    yv - ay2, xv - ax2,
                    yv + ay2, xv + ax2,
                ], axis=1)
                level_boxes.append(boxes[:, np.newaxis, :])

            level_boxes = np.concatenate(level_boxes, axis=1)
            all_boxes.append(level_boxes.reshape(-1, 4))

        anchors = np.vstack(all_boxes).astype(np.float32)
        return anchors[np.newaxis, ...]  # (1, N, 4)

    # ── NMS ────────────────────────────────────────────────────────────────

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
        """
        NumPy 版本的 Non-Maximum Suppression。
        返回保留的索引。
        """
        if len(boxes) == 0:
            return np.array([], dtype=np.int32)

        x1, y1 = boxes[:, 0], boxes[:, 1]
        x2, y2 = boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)

        order = scores.argsort()[::-1]
        keep = []

        while len(order) > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)

            remain = np.where(iou <= iou_threshold)[0]
            order = order[remain + 1]

        return np.array(keep, dtype=np.int32)
