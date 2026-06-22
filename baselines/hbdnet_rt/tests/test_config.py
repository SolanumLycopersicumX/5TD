"""测试配置文件可正常加载。"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from hbdnet_rt.utils.config import load_config


def test_load_all_configs():
    cfg = load_config()
    assert cfg.scene is not None
    assert cfg.model is not None
    assert cfg.planner is not None
    assert cfg.safety is not None
    # 关键字段存在
    assert "backbone" in cfg.model
    assert "dwa" in cfg.planner
    assert "safety" in cfg.safety
