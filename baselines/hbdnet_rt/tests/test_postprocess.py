"""测试后处理输出格式正确。"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import torch
from hbdnet_rt.perception.model import HBDNetRT
from hbdnet_rt.perception.postprocess import PostProcessor
from hbdnet_rt.utils.config import load_config


def test_postprocess_format():
    cfg = load_config()
    model = HBDNetRT()
    post = PostProcessor(cfg)
    x = torch.randn(1, 3, 384, 640)
    raw = model(x)
    out = post.process(raw)
    assert "detections" in out
    assert "ego_passable_mask" in out
    assert "hard_boundary_mask" in out
    assert "hard_boundary_edge" in out
    assert "confidence" in out
    assert "overall" in out["confidence"]
