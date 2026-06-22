#!/usr/bin/env python3
"""运行单帧感知推理。"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import torch
from hbdnet_rt.utils.config import load_config
from hbdnet_rt.utils.logger import get_logger
from hbdnet_rt.perception.inference import PerceptionInference

def main():
    logger = get_logger("inference_demo")
    cfg = load_config()
    infer = PerceptionInference(cfg)
    img = torch.randn(1, 3, 384, 640)
    out = infer(img)
    logger.info(f"Inference done. Keys: {list(out.keys())}")
    logger.info(f"Confidence: {out['confidence']}")
    logger.info("✅ Inference pipeline OK")

if __name__ == "__main__":
    main()
