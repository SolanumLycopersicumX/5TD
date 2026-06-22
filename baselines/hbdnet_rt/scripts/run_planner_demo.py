#!/usr/bin/env python3
"""运行 DWA 路径规划 Demo (不依赖模型)。"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
from hbdnet_rt.utils.config import load_config
from hbdnet_rt.utils.logger import get_logger
from hbdnet_rt.planning.dwa import DWAPlanner

def main():
    logger = get_logger("planner_demo")
    cfg = load_config()
    planner = DWAPlanner(cfg)
    # 空旷环境
    result = planner.plan({"current_pose": [0,0,0], "current_velocity": 0.5})
    logger.info(f"Empty: speed={result['target_speed']:.2f}, steer={result['target_steering']:.2f}, status={result['planner_status']}")
    # 全堵 (无 risk_grid, 应仍可输出)
    result2 = planner.plan({})
    logger.info(f"NoInput: status={result2['planner_status']}")
    logger.info("✅ Planner demo OK")

if __name__ == "__main__":
    main()
