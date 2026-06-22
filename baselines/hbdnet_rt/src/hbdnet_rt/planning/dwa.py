"""
Risk-Adaptive DWA 路径规划器。
速度/转角采样 → 自行车模型前向模拟 → 四因子 Risk-Adaptive 评分 → 选最优。
硬边界不可穿越, ego_passable 外 = 出界, 无可行轨迹输出 STOP。
"""
import numpy as np
from typing import Dict, List, Optional
from .trajectory import simulate_trajectory
from .cost_functions import compute_costs


class DWAPlanner:
    """
    Risk-Adaptive DWA。

    输入: current_pose, current_velocity, risk_grid,
          goal_direction, vehicle_profile
    输出: target_speed, target_steering_angle, selected_trajectory, safety_state
    """

    def __init__(self, config):
        dwa_cfg = config.planner.get("dwa", {})
        cost_w = dwa_cfg.get("cost_weights", {})

        # 采样参数
        self.min_vel = dwa_cfg.get("min_velocity_ms", 0.2)
        self.max_vel = dwa_cfg.get("max_velocity_ms", 1.5)
        self.n_vel = dwa_cfg.get("velocity_samples", 5)
        self.min_steer = dwa_cfg.get("min_steering_rad", -0.5)
        self.max_steer = dwa_cfg.get("max_steering_rad", 0.5)
        self.n_steer = dwa_cfg.get("steering_samples", 9)
        self.predict_time = dwa_cfg.get("predict_time_s", 2.0)
        self.dt = dwa_cfg.get("dt_s", 0.1)

        # 车辆参数 (可从 vehicle_profile 覆盖)
        self.wheelbase = dwa_cfg.get("wheelbase_m", 2.0)
        self.vehicle_width = dwa_cfg.get("vehicle_width_m", 2.0)
        self.safety_margin = dwa_cfg.get("safety_margin_m", 0.25)
        self.min_clearance_m = dwa_cfg.get("min_clearance_m", 0.30)
        self.hard_boundary_threshold = dwa_cfg.get("hard_boundary_risk_threshold", 0.99)

        # Risk-Adaptive 代价权重
        self.cost_weights = {
            "clearance": cost_w.get("clearance", 0.35),
            "risk_cost": cost_w.get("risk_cost", 0.25),
            "smoothness": cost_w.get("smoothness", 0.25),
            "progress": cost_w.get("progress", 0.15),
        }

        # Risk-Adaptive 速度调节阈值
        self.risk_slow_threshold = dwa_cfg.get("risk_slow_threshold", 0.5)
        self.risk_stop_threshold = dwa_cfg.get("risk_stop_threshold", 0.85)

        # 帧间状态
        self._last_steering = 0.0
        self._last_velocity = 0.0

    # ── 主入口 ──

    def plan(self, inputs: Dict) -> Dict:
        pose = inputs.get("current_pose", [0.0, 0.0, 0.0])
        curr_vel = inputs.get("current_velocity", 0.5)
        risk_grid = inputs.get("risk_grid", None)
        grid_extent = inputs.get("grid_extent", None)

        # vehicle_profile: 可选, 覆盖默认车辆参数
        vp = inputs.get("vehicle_profile", {})
        wheelbase = vp.get("wheelbase_m", self.wheelbase)
        veh_width = vp.get("vehicle_width_m", self.vehicle_width)
        safety_m = vp.get("safety_margin_m", self.safety_margin)

        # ── 采样 ──
        velocities, steerings = self._sample_window(curr_vel)
        steps = int(self.predict_time / self.dt)

        # ── 评估所有候选 ──
        candidates = []
        for v in velocities:
            for s in steerings:
                traj = simulate_trajectory(pose, v, s, wheelbase, self.dt, steps)
                costs = compute_costs(
                    traj, risk_grid, grid_extent,
                    self._last_steering, s,
                    weights=self.cost_weights,
                    hard_boundary_threshold=self.hard_boundary_threshold,
                    vehicle_width_m=veh_width,
                    safety_margin_m=safety_m,
                )
                candidates.append({
                    "velocity": v,
                    "steering": s,
                    "trajectory": traj,
                    "costs": costs,
                })

        # ── 筛选可行轨迹 ──
        # 1. 排除碰撞
        feasible = [c for c in candidates if c["costs"]["collision"] == 0]

        # 2. Risk-Adaptive: 排除高风险轨迹 (avg_risk >= 0.85 → 视为不安全)
        feasible = [c for c in feasible
                    if c["costs"]["avg_risk"] < self.risk_stop_threshold]

        # ── 无可行路径 → STOP ──
        if not feasible:
            return self._no_path_result()

        # ── 选最优 ──
        feasible.sort(key=lambda c: c["costs"]["total_score"], reverse=True)
        best = feasible[0]

        # Risk-Adaptive 速度: 根据最优轨迹的 avg_risk 调节
        avg_risk = best["costs"]["avg_risk"]
        if avg_risk >= self.risk_slow_threshold:
            best["velocity"] *= 0.5   # 高风险区域减速
        best["velocity"] = max(self.min_vel, best["velocity"])

        # 确定 safety_state
        if avg_risk >= self.risk_stop_threshold:
            safety_state = "S2_SLOWDOWN"
        elif avg_risk >= self.risk_slow_threshold:
            safety_state = "S1_CAUTIOUS"
        else:
            safety_state = "S0_NORMAL"

        # 更新帧间状态
        self._last_steering = best["steering"]
        self._last_velocity = best["velocity"]

        return {
            "target_speed": best["velocity"],
            "target_steering_angle": best["steering"],
            "selected_trajectory": best["trajectory"].tolist(),
            "safety_state": safety_state,
            "planner_status": "OK",
            "cost_breakdown": best["costs"],
            "num_candidates": len(candidates),
            "num_feasible": len(feasible),
        }

    # ── 内部 ──

    def _sample_window(self, curr_vel: float):
        v_range = self.max_vel - self.min_vel
        v_center = np.clip(curr_vel, self.min_vel, self.max_vel)
        v_min = max(self.min_vel, v_center - v_range * 0.3)
        v_max = min(self.max_vel, v_center + v_range * 0.3)
        velocities = np.linspace(v_min, v_max, self.n_vel)
        steerings = np.linspace(self.min_steer, self.max_steer, self.n_steer)
        return velocities, steerings

    def _no_path_result(self) -> Dict:
        return {
            "target_speed": 0.0,
            "target_steering_angle": 0.0,
            "selected_trajectory": None,
            "safety_state": "S3_STOP",
            "planner_status": "NO_PATH",
            "cost_breakdown": {"total_score": -1000.0},
            "num_candidates": 0,
            "num_feasible": 0,
        }
