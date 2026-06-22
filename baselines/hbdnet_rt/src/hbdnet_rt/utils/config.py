"""配置加载模块。所有参数从 YAML 文件读取，不硬编码。"""

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict


@dataclass
class Config:
    """统一配置对象，从 YAML 文件加载。"""
    scene: Dict[str, Any] = field(default_factory=dict)
    model: Dict[str, Any] = field(default_factory=dict)
    planner: Dict[str, Any] = field(default_factory=dict)
    safety: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, config_dir: str) -> "Config":
        config_dir = Path(config_dir)
        cfg = cls()
        for name in ["fixed_scene", "model", "planner", "safety"]:
            path = config_dir / f"{name}.yaml"
            if path.exists():
                with open(path) as f:
                    setattr(cfg, name if name != "fixed_scene" else "scene",
                            yaml.safe_load(f))
        return cfg


def load_config(config_dir: str = None) -> Config:
    if config_dir is None:
        config_dir = Path(__file__).parent.parent.parent.parent / "configs"
    return Config.from_yaml(str(config_dir))
