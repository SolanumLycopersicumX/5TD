#!/usr/bin/env python3
"""
端到端管线: 图像 → 预处理 → 模型推理 → 后处理 → BEV 栅格 → DWA → 安全修正 → 控制命令。
训练后替换模型权重即可直接使用。当前使用随机权重验证管线。
"""
import sys, os, time, argparse
import cv2
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from hbdnet_rt.utils.config import load_config
from hbdnet_rt.utils.logger import get_logger
from hbdnet_rt.utils.timing import Timer
from hbdnet_rt.perception.preprocessor import ImagePreprocessor
from hbdnet_rt.perception.inference import PerceptionInference
from hbdnet_rt.mapping.occupancy_grid import OccupancyGrid
from hbdnet_rt.mapping.risk_grid import RiskGrid
from hbdnet_rt.planning.dwa import DWAPlanner
from hbdnet_rt.safety.state_machine import SafetyStateMachine
from hbdnet_rt.control.command import ControlCommand


def run_pipeline(image, cfg, timer=None):
    """完整一帧: image → control command。"""
    if timer is None:
        timer = Timer()

    # ── 1. 预处理 ──
    with timer.measure("preprocessing"):
        preprocessor = ImagePreprocessor(cfg)
        tensor = preprocessor(image)

    # ── 2. 模型推理 ──
    with timer.measure("model_inference"):
        infer = PerceptionInference(cfg)
        perc_out = infer(tensor)

    # ── 3. BEV 栅格 ──
    with timer.measure("grid_generation"):
        occup = OccupancyGrid(cfg)
        risk_gen = RiskGrid(cfg)
        occ = occup.generate(perc_out)
        risk = risk_gen.generate(perc_out, occ["occupancy_grid"])

    # ── 4. DWA ──
    with timer.measure("dwa"):
        planner = DWAPlanner(cfg)
        risk_arr = risk["risk_grid"].numpy() if torch.is_tensor(risk["risk_grid"]) else risk["risk_grid"]
        dwa = planner.plan({
            "current_pose": [0, 0, 0],
            "current_velocity": 0.5,
            "risk_grid": risk_arr,
            "grid_extent": risk["metadata"],
        })

    # ── 5. 安全修正 ──
    with timer.measure("safety"):
        safety = SafetyStateMachine(cfg)
        safety_out = safety.evaluate({
            "overall_confidence": perc_out["confidence"]["overall"],
            "max_risk": risk["max_risk"],
            "boundary_distance": 5.0,
            "worker_distance": 10.0,
            "has_feasible_path": dwa["planner_status"] == "OK",
        })
        cmd = safety.apply_to_dwa(dwa, safety_out)

    return cmd, perc_out, risk, dwa, safety_out


def main():
    parser = argparse.ArgumentParser(description="HBD-Net-RT 端到端管线")
    parser.add_argument("--input", "-i", help="输入图片路径")
    parser.add_argument("--frames", "-n", type=int, default=1, help="重复帧数 (测试用)")
    args = parser.parse_args()

    logger = get_logger("pipeline")
    cfg = load_config()
    timer = Timer()

    if args.input:
        image = cv2.imread(args.input)
        if image is None:
            logger.error(f"无法读取: {args.input}")
            return
    else:
        # 生成随机测试图
        image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

    logger.info(f"端到端管线启动 (输入: {image.shape})")

    for i in range(args.frames):
        cmd, perc_out, risk, dwa, safety_out = run_pipeline(image, cfg, timer)

        if i == 0 or i == args.frames - 1:
            logger.info(
                f"Frame {i}: speed={cmd['target_speed']:.2f} "
                f"steer={cmd['target_steering']:.2f} "
                f"state={cmd['safety_state']} "
                f"brake={cmd['brake']} "
                f"reason={cmd['reason'][:60]}")

    stats = timer.stats()
    logger.info(f"延迟统计 (avg ms): total={timer.total_mean_ms():.1f}")
    for name, s in stats.items():
        logger.info(f"  {name:<20s}: mean={s['mean_ms']:.2f} max={s['max_ms']:.2f}")

    budget_ok = timer.total_mean_ms() < 100
    logger.info(f"{'✅' if budget_ok else '❌'} 总延迟 {'< 100ms' if budget_ok else '>= 100ms'}")


if __name__ == "__main__":
    main()
