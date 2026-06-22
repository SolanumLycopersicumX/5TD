"""测试安全状态机 — 状态转换 / 零速强制 / 恢复 / 接管 / DWA 修正。"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from hbdnet_rt.utils.config import load_config
from hbdnet_rt.safety.state_machine import SafetyStateMachine, SafetyState


def _make_sm():
    return SafetyStateMachine(load_config())


def _normal_inputs(**overrides):
    """构造正常行驶的输入。"""
    d = {"overall_confidence": 0.9, "max_risk": 0.0,
         "boundary_distance": 5.0, "worker_distance": 10.0,
         "has_feasible_path": True}
    d.update(overrides)
    return d


# ═══════════════════════════════════════════════════════
#  状态转换: 正常 → 各级别
# ═══════════════════════════════════════════════════════

def test_normal_stays_normal():
    """高置信度 + 无风险 → S0_NORMAL。"""
    sm = _make_sm()
    r = sm.evaluate(_normal_inputs())
    assert r["safety_state"] == "S0_NORMAL"
    assert r["brake"] is False
    assert r["speed_limit_ratio"] == 1.0


def test_normal_to_cautious_on_low_confidence():
    sm = _make_sm()
    r = sm.evaluate(_normal_inputs(overall_confidence=0.4))
    assert r["safety_state"] == "S1_CAUTIOUS"
    assert r["speed_limit_ratio"] == 0.5


def test_normal_to_cautious_on_high_risk():
    sm = _make_sm()
    r = sm.evaluate(_normal_inputs(max_risk=0.4))
    assert r["safety_state"] == "S1_CAUTIOUS"


def test_normal_to_cautious_near_boundary():
    sm = _make_sm()
    r = sm.evaluate(_normal_inputs(boundary_distance=0.5))
    assert r["safety_state"] == "S1_CAUTIOUS"


def test_normal_to_slowdown():
    sm = _make_sm()
    r = sm.evaluate(_normal_inputs(overall_confidence=0.2, max_risk=0.7))
    assert r["safety_state"] == "S2_SLOWDOWN"
    assert r["speed_limit_ratio"] == 0.25


def test_normal_to_stop_on_very_low_conf():
    sm = _make_sm()
    r = sm.evaluate(_normal_inputs(overall_confidence=0.05))
    assert r["safety_state"] == "S3_STOP"
    assert r["brake"] is True
    assert r["speed_limit_ratio"] == 0.0


def test_normal_to_stop_on_no_path():
    sm = _make_sm()
    r = sm.evaluate(_normal_inputs(has_feasible_path=False))
    assert r["safety_state"] == "S3_STOP"


def test_normal_to_stop_on_worker_too_close():
    sm = _make_sm()
    r = sm.evaluate(_normal_inputs(worker_distance=1.5))
    assert r["safety_state"] == "S3_STOP"


def test_emergency_worker_very_close():
    sm = _make_sm()
    r = sm.evaluate(_normal_inputs(worker_distance=0.5))
    assert r["safety_state"] == "S3_STOP"


# ═══════════════════════════════════════════════════════
#  零速强制: STOP / TAKEOVER
# ═══════════════════════════════════════════════════════

def test_stop_forces_zero_speed():
    """STOP 状态下 apply_to_dwa 必须输出 speed=0。"""
    sm = _make_sm()
    safety = sm.evaluate(_normal_inputs(overall_confidence=0.0))
    assert safety["safety_state"] == "S3_STOP"

    dwa = {"target_speed": 1.0, "target_steering": 0.2}
    cmd = sm.apply_to_dwa(dwa, safety)
    assert cmd["target_speed"] == 0.0, f"STOP must force speed=0, got {cmd['target_speed']}"
    assert cmd["brake"] is True


def test_takeover_forces_zero_speed():
    """手动触发 TAKEOVER → apply_to_dwa 输出 speed=0。"""
    sm = _make_sm()
    # 模拟连续 STOP 触发 TAKEOVER
    for _ in range(15):
        safety = sm.evaluate(_normal_inputs(overall_confidence=0.0, has_feasible_path=False))

    assert safety["safety_state"] == "S4_MANUAL_TAKEOVER"

    dwa = {"target_speed": 1.0, "target_steering": 0.5}
    cmd = sm.apply_to_dwa(dwa, safety)
    assert cmd["target_speed"] == 0.0
    assert cmd["brake"] is True


def test_normal_preserves_dwa_steering():
    """NORMAL 状态下保留 DWA 转向角。"""
    sm = _make_sm()
    safety = sm.evaluate(_normal_inputs())
    dwa = {"target_speed": 1.0, "target_steering": 0.3}
    cmd = sm.apply_to_dwa(dwa, safety)
    assert cmd["target_speed"] > 0
    assert cmd["target_steering"] == 0.3


def test_cautious_limits_speed_preserves_steering():
    """CAUTIOUS 状态限速但不影响转向。"""
    sm = _make_sm()
    safety = sm.evaluate(_normal_inputs(overall_confidence=0.4))
    assert safety["safety_state"] == "S1_CAUTIOUS"

    dwa = {"target_speed": 1.0, "target_steering": 0.15}
    cmd = sm.apply_to_dwa(dwa, safety)
    assert cmd["target_speed"] == 0.5  # 50% limit
    assert cmd["target_steering"] == 0.15  # steering preserved


# ═══════════════════════════════════════════════════════
#  恢复逻辑: 降态需连续帧确认
# ═══════════════════════════════════════════════════════

def test_recovery_requires_multiple_frames():
    """从 STOP 恢复到 NORMAL 需要连续 5 帧 (默认)。"""
    sm = _make_sm()
    # 先进入 STOP
    sm.evaluate(_normal_inputs(overall_confidence=0.0))
    assert sm.state_name == "S3_STOP"

    # 1 帧正常 → 仍为 STOP
    sm.evaluate(_normal_inputs())
    assert sm.state_name == "S3_STOP"

    # 连续 4 帧 → 仍为 STOP
    for _ in range(3):
        sm.evaluate(_normal_inputs())
    assert sm.state_name == "S3_STOP"

    # 第 5 帧 → 恢复
    sm.evaluate(_normal_inputs())
    assert sm.state_name != "S3_STOP"


def test_upgrade_immediate():
    """安全升级立即生效, 不经过防抖。"""
    sm = _make_sm()
    # NORMAL → STOP 应该只需 1 帧
    r = sm.evaluate(_normal_inputs(worker_distance=0.5))
    assert r["safety_state"] == "S3_STOP"


# ═══════════════════════════════════════════════════════
#  复位
# ═══════════════════════════════════════════════════════

def test_reset():
    sm = _make_sm()
    sm.evaluate(_normal_inputs(overall_confidence=0.0))
    assert sm.state_name == "S3_STOP"
    sm.reset()
    assert sm.state_name == "S0_NORMAL"
