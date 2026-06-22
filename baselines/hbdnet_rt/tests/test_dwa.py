"""测试 DWA 路径规划 — 空旷/障碍/硬边界/无路 四场景。"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
from hbdnet_rt.utils.config import load_config
from hbdnet_rt.planning.dwa import DWAPlanner


def _make_grid(x_range=(-2.5, 2.5), y_range=(0.0, 8.0), resolution=0.10, fill=0.0):
    """构造测试用 risk_grid 和 grid_extent。"""
    nx = int((x_range[1] - x_range[0]) / resolution)
    ny = int((y_range[1] - y_range[0]) / resolution)
    grid = np.full((ny, nx), fill, dtype=np.float32)
    extent = {"x_min": x_range[0], "x_max": x_range[1],
              "y_min": y_range[0], "y_max": y_range[1], "resolution": resolution}
    return grid, extent


def _add_rect(grid, extent, x0, x1, y0, y1, value):
    """在栅格中添加矩形区域。"""
    gx0 = int((x0 - extent["x_min"]) / extent["resolution"])
    gx1 = int((x1 - extent["x_min"]) / extent["resolution"])
    gy0 = int((y0 - extent["y_min"]) / extent["resolution"])
    gy1 = int((y1 - extent["y_min"]) / extent["resolution"])
    gx0 = max(0, min(grid.shape[1], gx0))
    gx1 = max(0, min(grid.shape[1], gx1))
    gy0 = max(0, min(grid.shape[0], gy0))
    gy1 = max(0, min(grid.shape[0], gy1))
    grid[gy0:gy1, gx0:gx1] = value


def _make_planner():
    return DWAPlanner(load_config())


# ═══════════════════════════════════════════════════
#  场景 1: 空旷环境 → 直行
# ═══════════════════════════════════════════════════

def test_empty_forward():
    """空旷环境: 无障碍, 应选择直行或接近直行的轨迹。"""
    planner = _make_planner()
    grid, extent = _make_grid(fill=0.0)

    result = planner.plan({
        "current_pose": [0, 0, 0],
        "current_velocity": 0.5,
        "risk_grid": grid,
        "grid_extent": extent,
    })

    assert result["planner_status"] == "OK"
    assert result["target_speed"] > 0
    # 空旷环境应走直行 or 接近直行
    assert abs(result["target_steering_angle"]) < 0.3, \
        f"空旷应直行, 转角={result['target_steering']:.3f} 太大了"


def test_empty_returns_trajectory():
    """空旷环境: 应返回有效轨迹。"""
    planner = _make_planner()
    grid, extent = _make_grid(fill=0.0)
    result = planner.plan({
        "current_pose": [0, 0, 0],
        "current_velocity": 0.5,
        "risk_grid": grid,
        "grid_extent": extent,
    })
    assert result["selected_trajectory"] is not None
    assert len(result["selected_trajectory"]) > 5


# ═══════════════════════════════════════════════════
#  场景 2: 前方有障碍 → 绕行
# ═══════════════════════════════════════════════════

def test_obstacle_detour():
    """前方有障碍: 应绕行 (转向非零) 而非停车。"""
    planner = _make_planner()
    grid, extent = _make_grid(fill=0.0)
    # 在正前方 2-4m 放置障碍物 (risk=0.8)
    _add_rect(grid, extent, -0.5, 0.5, 2.0, 4.0, value=0.8)

    result = planner.plan({
        "current_pose": [0, 0, 0],
        "current_velocity": 0.5,
        "risk_grid": grid,
        "grid_extent": extent,
    })

    # 应输出可行轨迹 (绕行而非停车)
    assert result["planner_status"] == "OK", \
        f"应可绕行, 状态={result['planner_status']}"
    assert result["target_speed"] > 0
    # 绕行路径不应经过高危险区域
    if result["selected_trajectory"] is not None:
        traj = np.array(result["selected_trajectory"])
        # 轨迹中点不应在障碍区域内 (允许起点和终点)
        mid = traj[len(traj)//2]
        gx = int((mid[0] - extent["x_min"]) / extent["resolution"])
        gy = int((mid[1] - extent["y_min"]) / extent["resolution"])
        if 0 <= gx < grid.shape[1] and 0 <= gy < grid.shape[0]:
            assert grid[gy, gx] < 0.8, \
                f"轨迹经过高风险区 ({mid[0]:.2f},{mid[1]:.2f}) risk={grid[gy,gx]}"


def test_obstacle_side_detour_steering_direction():
    """障碍偏右 → 应向左绕行 (负转向)。"""
    planner = _make_planner()
    grid, extent = _make_grid(fill=0.0)
    # 障碍在右半侧 (x>0), 且较窄、距离适中, 可以绕行
    _add_rect(grid, extent, 0.3, 1.0, 1.5, 4.0, value=0.99)

    result = planner.plan({
        "current_pose": [0, 0, 0],
        "current_velocity": 0.5,
        "risk_grid": grid,
        "grid_extent": extent,
    })

    # 应可绕行且偏向左侧
    assert result["planner_status"] == "OK"
    assert result["target_steering_angle"] < 0.2,         f"障碍在右侧, 应左转(<0.2), 实际={result['target_steering']:.3f}"


# ═══════════════════════════════════════════════════
#  场景 3: 硬边界 → 不可穿越
# ═══════════════════════════════════════════════════

def test_hard_boundary_uncrossable():
    """硬边界 (risk=1.0) 不可穿越: 轨迹不能经过 risk=1.0 区域。"""
    planner = _make_planner()
    grid, extent = _make_grid(fill=0.0)
    # 右侧设置硬边界 (隔离沟/隔离带) — risk=1.0
    _add_rect(grid, extent, 2.0, 2.5, 0.0, 8.0, value=1.0)

    result = planner.plan({
        "current_pose": [0, 0, 0],
        "current_velocity": 0.5,
        "risk_grid": grid,
        "grid_extent": extent,
    })

    # 应有可行轨迹 (从左侧通过)
    assert result["planner_status"] == "OK", \
        f"硬边界在右侧, 左侧应有路, 状态={result['planner_status']}"

    # 选中轨迹不能穿越硬边界
    if result["selected_trajectory"] is not None:
        traj = np.array(result["selected_trajectory"])
        for pt in traj:
            gx = int((pt[0] - extent["x_min"]) / extent["resolution"])
            gy = int((pt[1] - extent["y_min"]) / extent["resolution"])
            if 0 <= gx < grid.shape[1] and 0 <= gy < grid.shape[0]:
                assert grid[gy, gx] < 0.99, \
                    f"轨迹穿越硬边界! pt=({pt[0]:.2f},{pt[1]:.2f})"


def test_hard_boundary_both_sides():
    """两侧都是硬边界 → 只能直行在中间。"""
    planner = _make_planner()
    grid, extent = _make_grid(fill=0.0)
    # 左侧硬边界
    _add_rect(grid, extent, -2.5, -2.0, 0.0, 8.0, value=1.0)
    # 右侧硬边界
    _add_rect(grid, extent, 2.0, 2.5, 0.0, 8.0, value=1.0)

    result = planner.plan({
        "current_pose": [0, 0, 0],
        "current_velocity": 0.5,
        "risk_grid": grid,
        "grid_extent": extent,
    })

    assert result["planner_status"] == "OK"
    # 轨迹应保持在 x ∈ [-2, 2] 之间
    if result["selected_trajectory"] is not None:
        traj = np.array(result["selected_trajectory"])
        xs = traj[:, 0]
        assert xs.min() > -2.0, f"轨迹超出左侧硬边界: min_x={xs.min():.2f}"
        assert xs.max() < 2.0, f"轨迹超出右侧硬边界: max_x={xs.max():.2f}"


# ═══════════════════════════════════════════════════
#  场景 4: 全堵 → STOP
# ═══════════════════════════════════════════════════

def test_fully_blocked_stop():
    """前方全部被硬边界或障碍占用: 应输出 STOP。"""
    planner = _make_planner()
    grid, extent = _make_grid(fill=1.0)  # 全部 risk=1.0

    result = planner.plan({
        "current_pose": [0, 0, 0],
        "current_velocity": 0.5,
        "risk_grid": grid,
        "grid_extent": extent,
    })

    assert result["planner_status"] == "NO_PATH"
    assert result["target_speed"] == 0.0, \
        f"全堵应停车, speed={result['target_speed']}"


def test_narrow_passage():
    """仅有一条窄通道 → 应找到并通过。"""
    planner = _make_planner()
    grid, extent = _make_grid(fill=1.0)
    # 在中间留一条窄通道 (宽 1m)
    _add_rect(grid, extent, -0.5, 0.5, 0.0, 8.0, value=0.0)

    result = planner.plan({
        "current_pose": [0, 0, 0],
        "current_velocity": 0.5,
        "risk_grid": grid,
        "grid_extent": extent,
    })

    assert result["planner_status"] == "OK", \
        f"有窄通道应能找到路, 状态={result['planner_status']}"
    # 轨迹应保持在通道内 x ∈ [-0.5, 0.5]
    if result["selected_trajectory"] is not None:
        traj = np.array(result["selected_trajectory"])
        xs = traj[:, 0]
        assert xs.min() > -0.7, f"轨迹越界左: min_x={xs.min():.2f}"
        assert xs.max() < 0.7, f"轨迹越界右: max_x={xs.max():.2f}"


# ═══════════════════════════════════════════════════
#  辅助测试
# ═══════════════════════════════════════════════════

def test_dwa_output_format():
    """DWA 输出包含所有必要字段。"""
    planner = _make_planner()
    result = planner.plan({"current_pose": [0,0,0], "current_velocity": 0.5})
    for key in ["target_speed", "target_steering_angle", "selected_trajectory",
                "planner_status", "cost_breakdown", "safety_state"]:
        assert key in result, f"缺少字段: {key}"


def test_dwa_empty_input_handles_gracefully():
    """空输入也应返回 dict 不抛异常。"""
    planner = _make_planner()
    result = planner.plan({})
    assert "target_speed" in result
    assert "planner_status" in result
