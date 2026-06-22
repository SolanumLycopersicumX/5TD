#!/usr/bin/env python3
"""
延迟 Benchmark 工具。
分别统计 preprocessing / model inference / postprocess /
grid generation / DWA / safety / total 各阶段耗时。
"""
import sys, os, time, argparse
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from hbdnet_rt.utils.config import load_config
from hbdnet_rt.utils.logger import get_logger
from hbdnet_rt.perception.model import HBDNetRT
from hbdnet_rt.perception.postprocess import PostProcessor
from hbdnet_rt.mapping.occupancy_grid import OccupancyGrid
from hbdnet_rt.mapping.risk_grid import RiskGrid
from hbdnet_rt.planning.dwa import DWAPlanner
from hbdnet_rt.safety.state_machine import SafetyStateMachine


def benchmark(num_frames=100, warmup=10, device="cpu"):
    """运行 benchmark, 返回各阶段耗时统计 (ms)。"""
    logger = get_logger("benchmark")
    cfg = load_config()

    # ── 初始化模块 ──
    model = HBDNetRT().to(device).eval()
    postproc = PostProcessor(cfg)
    occup_grid = OccupancyGrid(cfg)
    risk_grid_gen = RiskGrid(cfg)
    planner = DWAPlanner(cfg)
    safety = SafetyStateMachine(cfg)

    # ── 计时存储 ──
    timings = {k: [] for k in [
        "preprocessing", "model_inference", "postprocess",
        "grid_generation", "risk_generation", "dwa", "safety", "total"]}

    dummy = torch.randn(1, 3, 384, 640).to(device)

    for i in range(num_frames + warmup):
        t_total_start = time.perf_counter()

        # 1. Preprocessing (模拟: resize + CLAHE + Canny 等, 用简单随机操作近似)
        t0 = time.perf_counter()
        preprocessed = torch.randn(1, 3, 384, 640)  # 模拟预处理输出
        t_pre = (time.perf_counter() - t0) * 1000

        # 2. Model Inference
        t0 = time.perf_counter()
        with torch.no_grad():
            raw = model(dummy)
        if device == "cuda":
            torch.cuda.synchronize()
        t_inf = (time.perf_counter() - t0) * 1000

        # 3. Postprocess
        t0 = time.perf_counter()
        perc_out = postproc.process(raw)
        t_post = (time.perf_counter() - t0) * 1000

        # 4. Grid Generation (occupancy + risk)
        t0 = time.perf_counter()
        occ = occup_grid.generate(perc_out)
        risk = risk_grid_gen.generate(perc_out, occ["occupancy_grid"])
        t_grid = (time.perf_counter() - t0) * 1000

        # 5. DWA
        t0 = time.perf_counter()
        dwa_out = planner.plan({
            "current_pose": [0, 0, 0],
            "current_velocity": 0.5,
            "risk_grid": risk["risk_grid"].numpy() if torch.is_tensor(risk["risk_grid"]) else risk["risk_grid"],
            "grid_extent": risk["metadata"],
        })
        t_dwa = (time.perf_counter() - t0) * 1000

        # 6. Safety
        t0 = time.perf_counter()
        safety_out = safety.evaluate({
            "overall_confidence": perc_out["confidence"]["overall"],
            "max_risk": risk["max_risk"],
            "boundary_distance": 5.0,
            "worker_distance": 10.0,
            "has_feasible_path": dwa_out["planner_status"] == "OK",
        })
        cmd = safety.apply_to_dwa(dwa_out, safety_out)
        t_safety = (time.perf_counter() - t0) * 1000

        t_total = (time.perf_counter() - t_total_start) * 1000

        if i >= warmup:
            timings["preprocessing"].append(t_pre)
            timings["model_inference"].append(t_inf)
            timings["postprocess"].append(t_post)
            timings["grid_generation"].append(t_grid)
            timings["dwa"].append(t_dwa)
            timings["safety"].append(t_safety)
            timings["total"].append(t_total)

    return timings


def print_stats(timings: dict):
    """打印统计结果。"""
    print("\n" + "=" * 72)
    print(f"{'Module':<24} {'Mean (ms)':>10} {'Min (ms)':>10} {'Max (ms)':>10} {'P95 (ms)':>10}")
    print("-" * 72)

    total_mean = 0
    for name in ["preprocessing", "model_inference", "postprocess",
                  "grid_generation", "dwa", "safety", "total"]:
        vals = timings.get(name, [])
        if not vals:
            continue
        arr = np.array(vals)
        mean = np.mean(arr)
        if name != "total":
            total_mean += mean
        p95 = np.percentile(arr, 95)
        print(f"{name:<24} {mean:10.2f} {arr.min():10.2f} {arr.max():10.2f} {p95:10.2f}")

    print("-" * 72)
    print(f"{'Sum (excl. total)':<24} {total_mean:10.2f}")
    total_p95 = np.percentile(np.array(timings["total"]), 95)
    total_max = np.array(timings["total"]).max()
    print(f"\n{'P95 Total':<24} {total_p95:10.2f} ms")
    print(f"{'Max Total':<24} {total_max:10.2f} ms")
    budget_ok = total_p95 < 100
    print(f"\n{'✅ P95 < 100ms' if budget_ok else '❌ P95 >= 100ms'}")
    print("=" * 72 + "\n")


def main():
    parser = argparse.ArgumentParser(description="HBD-Net-RT 延迟 Benchmark")
    parser.add_argument("-n", "--frames", type=int, default=100, help="测试帧数 (默认100)")
    parser.add_argument("-w", "--warmup", type=int, default=10, help="预热帧数 (默认10)")
    parser.add_argument("-d", "--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    logger = get_logger("benchmark")
    logger.info(f"Benchmark: {args.frames} frames, {args.warmup} warmup, device={args.device}")

    timings = benchmark(num_frames=args.frames, warmup=args.warmup, device=args.device)
    print_stats(timings)


if __name__ == "__main__":
    main()
