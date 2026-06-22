"""测试 BEV 占用栅格和风险栅格。"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import torch
from hbdnet_rt.utils.config import load_config
from hbdnet_rt.mapping.bev_projector import BEVProjector
from hbdnet_rt.mapping.occupancy_grid import OccupancyGrid
from hbdnet_rt.mapping.risk_grid import RiskGrid


# ── Fixtures ──

def _make_config():
    return load_config()

def _make_empty_perception(ego_val=0.0, hb_val=0.0):
    """构造感知输出 dict。"""
    return {
        "ego_passable_mask": torch.full((1, 1, 96, 160), ego_val),
        "hard_boundary_mask": torch.full((1, 4, 96, 160), hb_val),
        "hard_boundary_edge": torch.zeros(1, 1, 96, 160),
        "detections": {"boxes": torch.zeros(0, 4), "scores": torch.zeros(0), "labels": torch.zeros(0, dtype=torch.long)},
        "confidence": {"detection": 0.9, "passable": 0.9, "boundary": 0.9, "overall": 0.9},
    }


# ── BEVProjector ──

def test_bev_grid_shape():
    cfg = _make_config()
    proj = BEVProjector(cfg)
    nx, ny = proj.grid_shape
    assert nx > 0 and ny > 0
    # 5m width / 0.1 = 50, 8m / 0.1 = 80
    assert nx == 50
    assert ny == 80


def test_bev_mask_projection():
    cfg = _make_config()
    proj = BEVProjector(cfg)
    mask = torch.ones(1, 1, 96, 160)
    proj_mask = proj.project_mask_to_bev(mask)
    assert proj_mask.shape[0] == 1
    assert proj_mask.shape[1] == 1
    assert proj_mask.shape[2:] == (proj.grid_shape[1], proj.grid_shape[0])


def test_bev_grid_extent():
    cfg = _make_config()
    proj = BEVProjector(cfg)
    ext = proj.grid_extent
    assert ext["x_min"] == -2.5
    assert ext["x_max"] == 2.5
    assert ext["y_min"] == 0.0
    assert ext["y_max"] == 8.0
    assert ext["resolution"] == 0.10


def test_bev_homography_interface():
    cfg = _make_config()
    proj = BEVProjector(cfg)
    import numpy as np
    H = np.eye(3)
    proj.set_homography(H)
    assert proj._homography is not None
    proj.clear_homography()
    assert proj._homography is None


# ── OccupancyGrid ──

def test_occupancy_all_passable():
    """全可通行 → occupancy 全 0。"""
    cfg = _make_config()
    grid = OccupancyGrid(cfg)
    perc = _make_empty_perception(ego_val=1.0, hb_val=0.0)
    result = grid.generate(perc)
    occ = result["occupancy_grid"]
    assert occ.shape == (1, 1, grid.ny, grid.nx)
    # 全部可通行 → max 应该很低 (允许少量边界误差)
    assert occ.max() < 0.5, f"Expected near-zero occupancy, got max={occ.max().item()}"


def test_occupancy_all_blocked():
    """无可通行区域 → occupancy 全 1。"""
    cfg = _make_config()
    grid = OccupancyGrid(cfg)
    perc = _make_empty_perception(ego_val=0.0)
    result = grid.generate(perc)
    occ = result["occupancy_grid"]
    assert occ.min() > 0.5, f"Expected near-full occupancy, got min={occ.min().item()}"


def test_occupancy_empty_input():
    """空输入 → 全部占用 (保守)。"""
    cfg = _make_config()
    grid = OccupancyGrid(cfg)
    result = grid.generate({})
    occ = result["occupancy_grid"]
    assert occ.shape == (1, 1, grid.ny, grid.nx)
    assert occ.min() > 0.5


def test_occupancy_metadata():
    cfg = _make_config()
    grid = OccupancyGrid(cfg)
    result = grid.generate(_make_empty_perception())
    assert "metadata" in result
    assert result["metadata"]["resolution"] == 0.10


# ── RiskGrid ──

def test_risk_hard_boundary_max():
    """hard_boundary 区域 → risk = 1.0。"""
    cfg = _make_config()
    rg = RiskGrid(cfg)
    perc = _make_empty_perception(ego_val=1.0, hb_val=1.0)
    result = rg.generate(perc)
    risk = result["risk_grid"]
    # hard_boundary=1 且 ego_passable=1 → hard boundary 优先，risk=1
    assert risk.max() >= 0.99, f"Hard boundary should give max risk, got {risk.max().item()}"


def test_risk_ego_passable_region_safe():
    """ego_passable 且无 hard_boundary → risk ≈ 0。"""
    cfg = _make_config()
    rg = RiskGrid(cfg)
    # 大部分区域可通行, 无 hard boundary
    ego = torch.full((1, 1, 96, 160), 1.0)
    hb = torch.zeros(1, 4, 96, 160)
    dets = {"boxes": torch.zeros(0, 4), "scores": torch.zeros(0), "labels": torch.zeros(0, dtype=torch.long)}
    perc = {"ego_passable_mask": ego, "hard_boundary_mask": hb,
            "detections": dets, "confidence": {"overall": 0.9}}
    result = rg.generate(perc)
    risk = result["risk_grid"]
    # 全可通行区域 → 大部分应该 risk < 0.1
    assert risk.mean() < 0.3, f"Passable region should be low risk, got mean={risk.mean().item()}"


def test_risk_output_range():
    """风险值在 [0, 1] 范围内。"""
    cfg = _make_config()
    rg = RiskGrid(cfg)
    perc = _make_empty_perception(ego_val=1.0, hb_val=0.5)
    result = rg.generate(perc)
    risk = result["risk_grid"]
    assert risk.min() >= 0.0
    assert risk.max() <= 1.0


def test_risk_max_risk_field():
    """max_risk 字段正确。"""
    cfg = _make_config()
    rg = RiskGrid(cfg)
    # 全部 hard_boundary → max_risk ≈ 1.0
    perc = _make_empty_perception(ego_val=0.0, hb_val=1.0)
    result = rg.generate(perc)
    assert "max_risk" in result
    assert result["max_risk"] >= 0.99


def test_risk_low_confidence_bias():
    """低置信度 → 风险升高。"""
    cfg = _make_config()
    rg = RiskGrid(cfg)
    # 高置信度
    perc_high = _make_empty_perception(ego_val=1.0, hb_val=0.0)
    perc_high["confidence"]["overall"] = 0.9
    r_high = rg.generate(perc_high)["risk_grid"].mean().item()

    # 低置信度
    perc_low = _make_empty_perception(ego_val=1.0, hb_val=0.0)
    perc_low["confidence"]["overall"] = 0.2
    r_low = rg.generate(perc_low)["risk_grid"].mean().item()

    # 低置信度应比高置信度风险更高
    assert r_low > r_high, f"Low conf ({r_low}) should be > high conf ({r_high})"
