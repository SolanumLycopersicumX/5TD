"""感知推理管线。串联模型 forward + 后处理。"""
import torch
from typing import Dict
from .model import HBDNetRT
from .postprocess import PostProcessor


class PerceptionInference:
    """感知推理统一入口。"""

    def __init__(self, config, device: str = "cpu"):
        self.config = config
        self.device = device
        self.model = HBDNetRT()
        self.model.to(device)
        self.model.eval()
        self.postprocessor = PostProcessor(config)

    @property
    def input_shape(self):
        return self.model.input_shape

    @torch.no_grad()
    def __call__(self, image_tensor: torch.Tensor) -> Dict:
        """推理一帧。输入 [1,3,H,W] 或 [3,H,W], 输出后处理后的统一字典。"""
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(self.device)
        raw = self.model(image_tensor)
        return self.postprocessor.process(raw)

    def get_model_info(self) -> Dict:
        return self.model.get_model_info()
