"""
预处理管线。letterbox → CLAHE → 归一化 → 输出 tensor。
不依赖训练数据，纯 OpenCV + numpy。
"""
import cv2
import numpy as np
import torch
from typing import Tuple


class ImagePreprocessor:
    """
    图像预处理器。
    输入: BGR numpy 图像 (任意分辨率)
    输出: torch.Tensor [1, 3, H, W] 归一化到 [0, 1]
    """

    def __init__(self, config):
        input_cfg = config.scene.get("input", {})
        cam_cfg = input_cfg.get("camera", {})
        model_cfg = input_cfg.get("model_input", {})

        self.src_w = cam_cfg.get("width", 1280)
        self.src_h = cam_cfg.get("height", 720)
        self.target_w = model_cfg.get("width", 640)
        self.target_h = model_cfg.get("height", 384)
        self.letterbox = model_cfg.get("letterbox", True)
        self.letterbox_color = model_cfg.get("letterbox_color", 128)

        # CLAHE (可选, 默认启用)
        model_head = config.model.get("heads", {})
        self.use_clahe = True
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        """
        预处理一帧。
        image: BGR numpy 数组 (H, W, 3), uint8
        返回: [1, 3, target_h, target_w] float32, 值域 [0, 1]
        """
        # 1. Letterbox resize
        if self.letterbox:
            img = self._letterbox(image)
        else:
            img = cv2.resize(image, (self.target_w, self.target_h))

        # 2. CLAHE 增强 (在亮度通道上)
        if self.use_clahe:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = self.clahe.apply(l)
            lab = cv2.merge([l, a, b])
            img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # 3. BGR → RGB → tensor → normalize
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        return tensor.unsqueeze(0)

    def _letterbox(self, image: np.ndarray) -> np.ndarray:
        """保持宽高比的缩放, 填充灰边。"""
        h, w = image.shape[:2]
        scale = min(self.target_w / w, self.target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(image, (new_w, new_h))
        canvas = np.full((self.target_h, self.target_w, 3),
                         self.letterbox_color, dtype=np.uint8)
        y_off = (self.target_h - new_h) // 2
        x_off = (self.target_w - new_w) // 2
        canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
        return canvas
