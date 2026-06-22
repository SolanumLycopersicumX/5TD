"""
场景决策验证 — 手写 risk_grid, 验证 DWA + 安全状态机决策是否正确。
不依赖模型, 不依赖数据, 现在就能跑。
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
from hbdnet_rt.utils.config import load_config
from hbdnet_rt.planning.dwa import DWAPlanner
from hbdnet_rt.safety.state_machine import SafetyStateMachine


def _make_grid(x_range=(-2.5, 2.5), y_range=(0.0, 8.0), resolution=0.10, fill=0.0):
    """构造 risk_grid [ny, nx] 和 extent。"""
    nx = int((x_range[1] - x_range[0]) / resolution)
    ny = int((y_range[1] - y_range[0]) / resolution)
    return np.full((ny, nx), fill, dtype=np.float32), {
        "x_min": x_range[0], "x_max": x_range[1],
        "y_min": y_range[0], "y_max": y_range[1],
        "resolution": resolution}


def _add_rect(grid, extent, x0, x1, y0, y1, value):
    """在栅格中添加矩形风险区域。"""
    gx0 = max(0, int((x0 - extent["x_min"]) / extent["resolution"]))
    gx1 = min(grid.shape[1], int((x1 - extent["x_min"]) / extent["resolution"]))
    gy0 = max(0, int((y0 - extent["y_min"]) / extent["resolution"]))
    gy1 = min(grid.shape[0], int((y1 - extent["y_min"]) / extent["resolution"]))
    grid[gy0:gy1, gx0:gx1] = value


def _run_scenario(name, grid, extent, pose, velocity, worker_dist=10.0,
                  bound_dist=5.0, overall_conf=0.9):
    """运行一个场景, 返回完整决策链。"""
    cfg = load_config()
    planner = DWAPlanner(cfg)
    safety = SafetyStateMachine(cfg)

    dwa = planner.plan({
        "current_pose": pose, "current_velocity": velocity,
        "risk_grid": grid, "grid_extent": extent,
    })

    max_risk = float(grid.max())
    safety_out = safety.evaluate({
        "overall_confidence": overall_conf,
        "max_risk": max_risk,
        "boundary_distance": bound_dist,
        "worker_distance": worker_dist,
        "has_feasible_path": dwa["planner_status"] == "OK",
    })

    cmd = safety.apply_to_dwa(dwa, safety_out)

    return {
        "scenario": name,
        "dwa": dwa,
        "safety": safety_out,
        "command": cmd,
    }


def _check(result, expected_status, expected_speed_gt_0, desc):
    """验证场景结果并打印。"""
    dwa_status = result["dwa"]["planner_status"]
    safety_state = result["command"]["safety_state"]
    speed = result["command"]["target_speed"]
    steer = result["command"]["target_steering"]

    ok = True
    checks = []
    if dwa_status != expected_status:
        ok = False
        checks.append(f"❌ DWA={dwa_status}(期望{expected_status})")
    else:
        checks.append(f"✅ DWA={dwa_status}")
    if expected_speed_gt_0 and speed == 0:
        ok = False
        checks.append(f"❌ speed=0(应>0)")
    elif not expected_speed_gt_0 and speed > 0:
        ok = False
        checks.append(f"❌ speed={speed:.2f}(应=0)")
    else:
        checks.append(f"✅ speed={speed:.2f}")
    checks.append(f"steer={steer:.3f} state={safety_state}")

    status = "PASS" if ok else "FAIL"
    print(f"\n{'─'*60}")
    print(f"  {desc}")
    print(f"  {' '.join(checks)}")
    print(f"  → {status}")
    return ok


# ═══════════════════════════════════════════════════
#  场景 1: 空旷直道
# ═══════════════════════════════════════════════════
def test_scenario_01_empty_forward():
    """空旷直道: 无障碍, 应直行。"""
    grid, extent = _make_grid(fill=0.0)
    r = _run_scenario("empty", grid, extent, [0, 0, 0], 0.5)
    assert _check(r, "OK", True, "场景1: 空旷直道 → 应直行")


# ═══════════════════════════════════════════════════
#  场景 2: 前方障碍绕行
# ═══════════════════════════════════════════════════
def test_scenario_02_obstacle_detour():
    """前方3m处有障碍物(0.8m宽), 应绕行。"""
    grid, extent = _make_grid(fill=0.0)
    _add_rect(grid, extent, -0.4, 0.4, 3.0, 5.0, value=0.9)
    r = _run_scenario("detour", grid, extent, [0, 0, 0], 0.5)
    assert _check(r, "OK", True, "场景2: 前方3m障碍 → 应绕行(速度>0)")


# ═══════════════════════════════════════════════════
#  场景 3: 右侧硬边界不可穿越
# ═══════════════════════════════════════════════════
def test_scenario_03_ditch_boundary():
    """右侧有隔离沟(risk=1.0), 轨迹不应穿越。"""
    grid, extent = _make_grid(fill=0.0)
    _add_rect(grid, extent, 2.0, 2.5, 0.0, 8.0, value=1.0)  # 隔离沟
    r = _run_scenario("ditch", grid, extent, [0, 0, 0], 0.5)

    assert r["dwa"]["planner_status"] == "OK", "有路应OK"
    if r["dwa"]["selected_trajectory"]:
        traj = np.array(r["dwa"]["selected_trajectory"])
        for pt in traj:
            gx = int((pt[0] - extent["x_min"]) / extent["resolution"])
            if 0 <= gx < grid.shape[1]:
                gy = int((pt[1] - extent["y_min"]) / extent["resolution"])
                if 0 <= gy < grid.shape[0]:
                    assert grid[gy, gx] < 0.99, f"穿越隔离沟! pt=({pt[0]:.2f},{pt[1]:.2f})"
    print(f"\n{'─'*60}")
    print(f"  场景3: 右侧隔离沟 → 轨迹不越界 ✅ PASS")


# ═══════════════════════════════════════════════════
#  场景 4: 前方全堵 → STOP
# ═══════════════════════════════════════════════════
def test_scenario_04_fully_blocked():
    """前方全部risk=1.0, 应停车。"""
    grid, extent = _make_grid(fill=1.0)
    r = _run_scenario("blocked", grid, extent, [0, 0, 0], 0.5)
    assert _check(r, "NO_PATH", False, "场景4: 全堵 → 应STOP(speed=0)")


# ═══════════════════════════════════════════════════
#  场景 5: 工人突然出现在2m内 → 紧急STOP
# ═══════════════════════════════════════════════════
def test_scenario_05_worker_emergency():
    """工人距离1.5m, 即使风险栅格干净也应紧急停车。"""
    grid, extent = _make_grid(fill=0.0)
    r = _run_scenario("worker_emerg", grid, extent, [0, 0, 0], 0.5,
                       worker_dist=1.5)
    assert r["safety"]["safety_state"] == "S3_STOP", \
        f"工人1.5m应STOP, 实际={r['safety']['safety_state']}"
    assert r["command"]["target_speed"] == 0.0
    assert r["command"]["brake"] == True
    print(f"\n{'─'*60}")
    print(f"  场景5: 工人1.5m → S3_STOP speed=0 brake=True ✅ PASS")


# ═══════════════════════════════════════════════════
#  场景 6: 窄通道不够宽 → 不可通行
# ═══════════════════════════════════════════════════
def test_scenario_06_narrow_passage():
    """
    两侧全是risk=1.0, 中间仅留0.8m宽通道。
    车宽2m+余量0.25m×2=2.5m > 0.8m → 不可通行。
    """
    grid, extent = _make_grid(fill=1.0)
    _add_rect(grid, extent, -0.4, 0.4, 0.0, 8.0, value=0.0)  # 仅0.8m宽

    r = _run_scenario("narrow", grid, extent, [0, 0, 0], 0.5)
    # 通道太窄, DWA轨迹全部碰撞 → NO_PATH
    assert r["dwa"]["planner_status"] == "NO_PATH", \
        f"窄通道应NO_PATH, 实际={r['dwa']['planner_status']}"
    assert r["command"]["target_speed"] == 0.0
    print(f"\n{'─'*60}")
    print(f"  场景6: 0.8m窄通道(车宽2m) → NO_PATH speed=0 ✅ PASS")


# ═══════════════════════════════════════════════════
#  场景 7: 中间高风险区 → 减速通过
# ═══════════════════════════════════════════════════
def test_scenario_07_high_risk_slowdown():
    """前方有大片risk=0.7区域, 应减速但不停。"""
    grid, extent = _make_grid(fill=0.0)
    _add_rect(grid, extent, -1.5, 1.5, 1.0, 5.0, value=0.7)
    r = _run_scenario("slowdown", grid, extent, [0, 0, 0], 0.5,
                       overall_conf=0.35)  # 偏低置信度
    # DWA应能找到路(risk=0.7 < 0.85排除阈值), 但安全状态机应降级
    assert r["dwa"]["planner_status"] == "OK"
    assert r["safety"]["safety_state"] in ("S1_CAUTIOUS", "S2_SLOWDOWN"), \
        f"高风险区应谨慎, 实际={r['safety']['safety_state']}"
    assert r["command"]["target_speed"] < 0.5, \
        f"应减速, 实际={r['command']['target_speed']:.2f}"
    print(f"\n{'─'*60}")
    print(f"  场景7: 高风险区+低置信度 → 减速谨慎 ✅ PASS")


# ═══════════════════════════════════════════════════
#  场景 8: 两侧硬边界 + 中间障碍 → 沿中轴绕行
# ═══════════════════════════════════════════════════
def test_scenario_08_tunnel_with_obstacle():
    """
    模拟真实隧道: 左右都是硬边界(risk=1.0), 中间有个障碍物。
    应沿车道中心线绕行, 不越界。
    """
    grid, extent = _make_grid(fill=0.0)
    _add_rect(grid, extent, -2.5, -2.0, 0.0, 8.0, value=1.0)   # 左隔离带
    _add_rect(grid, extent, 2.0, 2.5, 0.0, 8.0, value=1.0)     # 右隔离沟
    _add_rect(grid, extent, -0.3, 0.3, 3.0, 5.0, value=0.8)    # 中间障碍

    r = _run_scenario("tunnel", grid, extent, [0, 0, 0], 0.5)
    assert r["dwa"]["planner_status"] == "OK", "应有可行路径"

    if r["dwa"]["selected_trajectory"]:
        traj = np.array(r["dwa"]["selected_trajectory"])
        xs = traj[:, 0]
        # 不能越界: x必须在 [-2.0, 2.0] 内
        assert xs.min() > -2.0, f"越界左侧: min_x={xs.min():.2f}"
        assert xs.max() < 2.0, f"越界右侧: max_x={xs.max():.2f}"
    print(f"\n{'─'*60}")
    print(f"  场景8: 隧道半幅+中间障碍 → 绕行不越界 ✅ PASS")


# ═══════════════════════════════════════════════════
#  总结
# ═══════════════════════════════════════════════════
if __name__ == "__main__":
    results = []
    for fn in [test_scenario_01_empty_forward, test_scenario_02_obstacle_detour,
               test_scenario_03_ditch_boundary, test_scenario_04_fully_blocked,
               test_scenario_05_worker_emergency, test_scenario_06_narrow_passage,
               test_scenario_07_high_risk_slowdown, test_scenario_08_tunnel_with_obstacle]:
        try:
            fn()
            results.append(("PASS", fn.__doc__))
        except AssertionError as e:
            results.append(("FAIL", f"{fn.__doc__}: {e}"))

    print(f"\n{'='*60}")
    print(f" 场景验证汇总")
    print(f"{'='*60}")
    for status, desc in results:
        print(f"  {status:4s}  {desc}")
    passed = sum(1 for s, _ in results if s == "PASS")
    print(f"\n  {passed}/{len(results)} 通过")
