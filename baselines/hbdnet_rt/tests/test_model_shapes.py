"""测试模型 forward shape 正确。"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import torch
from hbdnet_rt.perception.model import HBDNetRT


def test_model_forward_shape():
    model = HBDNetRT()
    x = torch.randn(1, 3, 384, 640)
    out = model(x)
    # Check keys
    assert "detections" in out
    assert "ego_passable_mask" in out
    assert "hard_boundary_mask" in out
    assert "hard_boundary_edge" in out
    assert "surface_risk_map" in out
    assert "confidence" in out
    # Check shapes
    B = 1
    assert out["ego_passable_mask"].shape == (B, 1, 96, 160)
    assert out["hard_boundary_mask"].shape == (B, 4, 96, 160)
    assert out["hard_boundary_edge"].shape == (B, 1, 96, 160)


def test_model_info():
    model = HBDNetRT()
    info = model.get_model_info()
    assert info["name"] == "HBD-Net-RT"
    assert info["backbone"] == "RepVGG-lite"
    assert info["total_params"] > 0
