"""
DWA 代价函数。Risk-Adaptive: 基于 risk_grid 的四因子评分。
clearance / risk_cost / smoothness / progress。
穿越 hard_boundary (risk>=0.99) 或出界 → 直接判为不可行。
"""
import numpy as np
from typing import Dict, Optional, Tuple


def compute_costs(
    trajectory: np.ndarray,
    risk_grid: Optional[np.ndarray],
    grid_extent: Optional[Dict],
    last_steering: float,
    current_steering: float,
    weights: Dict = None,
    hard_boundary_threshold: float = 0.99,
    vehicle_width_m: float = 2.0,
    safety_margin_m: float = 0.25,
    min_clearance_score: float = -1000.0,
) -> Dict:
    """
    Risk-Adaptive 轨迹评估。

    四因子:
      clearance:   1 - max_risk (整体安全性)
      risk_cost:   轨迹的平均风险代价 (鼓励走在低风险区域)
      smoothness:  与上一帧转向角的一致性
      progress:    沿车道方向的前进距离

    碰撞判定 (任一满足即不可行):
      - 轨迹点 risk >= hard_boundary_threshold (硬边界)
      - 轨迹点超出 grid 范围 (出界)
    """
    if weights is None:
        weights = {
            "clearance": 0.35,    # 安全间距
            "risk_cost": 0.25,    # 风险代价 (新增)
            "smoothness": 0.25,   # 转向平滑
            "progress": 0.15,     # 前进进度
        }

    # ── 1. 碰撞与风险采样 ──
    max_risk = 0.0
    risk_values = []
    has_collision = False

    if risk_grid is not None and grid_extent is not None:
        max_risk, risk_values, has_collision = _sample_risks(
            trajectory, risk_grid, grid_extent, hard_boundary_threshold)

    if has_collision:
        return {
            "total_score": min_clearance_score,
            "clearance": 0.0,
            "risk_cost": 0.0,
            "smoothness": 0.0,
            "progress": 0.0,
            "collision": 1,
            "max_risk": max_risk,
            "avg_risk": max_risk,
        }

    avg_risk = float(np.mean(risk_values)) if risk_values else 0.0

    # ── 2. Clearance: 安全余量 ──
    clearance = 1.0 - max_risk

    # ── 3. Risk Cost: 轨迹平均风险 (新增) ──
    # 将 avg_risk 映射到 [0, 1]: 高风险 → 低分
    risk_cost = 1.0 - avg_risk

    # ── 4. Smoothness ──
    max_steer = 0.5
    smoothness = 1.0 - min(1.0, abs(current_steering - last_steering) / max_steer)

    # ── 5. Progress ──
    displacement = trajectory[-1, 1] - trajectory[0, 1]  # y 纵向位移
    progress = min(1.0, max(0.0, abs(displacement) / 5.0))

    # ── 6. 综合 ──
    total = (weights["clearance"] * clearance +
             weights["risk_cost"] * risk_cost +
             weights["smoothness"] * smoothness +
             weights["progress"] * progress)

    return {
        "total_score": round(total, 4),
        "clearance": round(clearance, 4),
        "risk_cost": round(risk_cost, 4),
        "smoothness": round(smoothness, 4),
        "progress": round(progress, 4),
        "collision": 0,
        "max_risk": round(max_risk, 4),
        "avg_risk": round(avg_risk, 4),
    }


def _sample_risks(
    trajectory: np.ndarray,
    risk_grid: np.ndarray,
    grid_extent: Dict,
    hard_bound_threshold: float,
) -> Tuple[float, list, bool]:
    """遍历轨迹点，采样 risk 并检测碰撞。返回 (max_risk, risk_list, has_collision)。"""
    if risk_grid.ndim == 4:
        rg = risk_grid[0, 0]
    elif risk_grid.ndim == 3:
        rg = risk_grid[0]
    else:
        rg = risk_grid

    ny, nx = rg.shape
    x_min, x_max = grid_extent["x_min"], grid_extent["x_max"]
    y_min, y_max = grid_extent["y_min"], grid_extent["y_max"]
    res = grid_extent["resolution"]

    max_risk = 0.0
    risk_list = []
    has_collision = False

    for pt in trajectory:
        px, py = pt[0], pt[1]
        gx = int((px - x_min) / res)
        gy = int((py - y_min) / res)

        # 出界 = 碰撞
        if gx < 0 or gx >= nx or gy < 0 or gy >= ny:
            has_collision = True
            max_risk = 1.0
            risk_list.append(1.0)
            continue

        rv = float(rg[gy, gx])
        risk_list.append(rv)
        max_risk = max(max_risk, rv)

        if rv >= hard_bound_threshold:
            has_collision = True

    return max_risk, risk_list, has_collision
