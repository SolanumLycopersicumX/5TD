"""
公共数据结构与工具函数。
所有模块共享的 dataclass 定义和可视化辅助函数。
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ============================================================================
# 数据类
# ============================================================================

@dataclass
class FreeSpaceState:
    """可行驶区域检测结果"""
    is_valid: bool = False
    center_offset: float = 0.0          # 车辆中心相对可行驶区域中心的偏移 (-1~1, 左负右正)
    left_free_score: float = 0.0        # 左侧可行驶评分 (0~1, 越大越畅通)
    center_free_score: float = 0.0      # 中间可行驶评分
    right_free_score: float = 0.0       # 右侧可行驶评分
    free_space_polygon: Optional[list] = None  # 可行驶区域多边形顶点
    confidence: float = 0.0             # 综合置信度 (0~1)
    debug_image: Optional[np.ndarray] = None   # 可视化结果图


@dataclass
class ObstacleState:
    """障碍检测结果"""
    has_obstacle: bool = False
    obstacle_boxes: list = field(default_factory=list)   # [(x,y,w,h), ...]
    largest_obstacle_area: float = 0.0
    danger_level: float = 0.0           # 危险等级 (0~1)
    blocked_left: bool = False
    blocked_center: bool = False
    blocked_right: bool = False
    closest_obstacle_zone: str = "NONE"  # LEFT / CENTER / RIGHT / NONE
    debug_image: Optional[np.ndarray] = None


@dataclass
class Decision:
    """避障决策结果"""
    command: str = "FORWARD"     # FORWARD / TURN_LEFT / TURN_RIGHT / STOP / SLOW_DOWN / SEARCH_LANE
    speed: float = 0.5           # 0~1
    steering: float = 0.0        # v2.2 差速驱动: 归一化角速度 [-1, 1]，左负右正
    reason: str = ""             # 决策原因（日志用）
    confidence: float = 1.0      # 决策置信度 (0~1)
    drive_state: str = "NORMAL"  # NORMAL / CAUTION / EVASIVE / EMERGENCY


@dataclass
class LaneBoundaryState:
    """纵向结构检测结果 (#31-32): 隔离沟 + 隔离带"""
    is_valid: bool = False
    left_barrier_px: Optional[int] = None    # 左侧隔离带 ROI 列坐标
    ditch_px: Optional[int] = None           # 中央隔离沟 ROI 列坐标
    right_barrier_px: Optional[int] = None   # 右侧隔离带 ROI 列坐标
    lane_width_m: float = 0.0               # 车道宽度 (米)
    vanishing_point: Optional[tuple] = None  # (x, y) 消失点图像坐标
    confidence: float = 0.0                  # 检测置信度 (0~1)
    debug_image: Optional[np.ndarray] = None


@dataclass
class DebrisState:
    """碎石/碎渣检测结果 (#18)"""
    has_debris: bool = False
    debris_boxes: list = field(default_factory=list)   # [(x,y,w,h,size_cm), ...]
    has_large_debris: bool = False                     # 存在 >15cm 需避让的大石块
    debug_image: Optional[np.ndarray] = None


@dataclass
class PathPlan:
    """路径规划结果 (#46-47)"""
    is_valid: bool = False
    passable: bool = True                    # 车道是否可通过
    steering: float = 0.0                   # v2.2: 归一化角速度 [-1, 1]
    speed: float = 0.5                      # 0~1
    clearance_cm: float = 0.0               # 最小通过间距 (厘米)
    blocked_reason: str = ""                # 不可通过原因: LANE_TOO_NARROW / OBSTACLE / ""
    debug_image: Optional[np.ndarray] = None


@dataclass
class PreprocessResult:
    """图像预处理结果"""
    roi_frame: Optional[np.ndarray] = None      # ROI 裁剪后的彩色图
    gray: Optional[np.ndarray] = None           # 灰度图
    enhanced: Optional[np.ndarray] = None       # CLAHE 增强后的图
    edges: Optional[np.ndarray] = None          # Canny 边缘图
    binary_mask: Optional[np.ndarray] = None    # 二值分割 mask
    dehazed_frame: Optional[np.ndarray] = None  # 去雾后的彩色图 (Phase 2 新增)
    denoised_frame: Optional[np.ndarray] = None # 时域降噪后的图 (Phase 2 新增)
    debug_images: dict = field(default_factory=dict)


@dataclass
class ExposureState_(object):
    """曝光状态检测结果 (避免与 enum 重名, 用 ExposureState_ 表示 dataclass)"""
    state: str = "NORMAL"           # NORMAL / OVEREXPOSED / UNDEREXPOSED / AE_ADJUSTING
    overexposed_ratio: float = 0.0
    underexposed_ratio: float = 0.0
    brightness_delta: float = 0.0


@dataclass
class DegradationStatus:
    """安全降级状态"""
    level: int = 0                  # 0-4
    level_name: str = "L0_NORMAL"
    speed_limit_kmh: float = 30.0
    clearance_multiplier: float = 1.0
    allow_detour: bool = True
    should_stop: bool = False
    needs_takeover: bool = False
    summary: str = ""


# ============================================================================
# 可视化工具
# ============================================================================

def draw_zone_lines(image, roi_w, roi_h):
    """
    在 ROI 图像上绘制左/中/右分区竖线。
    竖线位置由 config.ZONE_LEFT_RATIO / ZONE_RIGHT_RATIO 决定。
    """
    from config import ZONE_LEFT_RATIO, ZONE_RIGHT_RATIO

    left_x = int(roi_w * ZONE_LEFT_RATIO)
    right_x = int(roi_w * ZONE_RIGHT_RATIO)

    cv2.line(image, (left_x, 0), (left_x, roi_h), (255, 255, 0), 1)
    cv2.line(image, (right_x, 0), (right_x, roi_h), (255, 255, 0), 1)

    # 标注区域名称
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, "LEFT", (left_x // 2 - 20, 20), font, 0.5, (255, 255, 0), 1)
    cv2.putText(image, "CENTER", (left_x + 10, 20), font, 0.5, (255, 255, 0), 1)
    cv2.putText(image, "RIGHT", (right_x + 10, 20), font, 0.5, (255, 255, 0), 1)


def draw_zone_lines_dynamic(image, left_zone, center_zone, right_zone, roi_h):
    """
    使用动态车道边界在 ROI 图像上绘制区域分割线。
    替代 draw_zone_lines 当 LaneBoundaryState 有效时使用。
    """
    # 左区右边界 = center 左边界
    left_x = left_zone[2]
    # 右区左边界 = center 右边界
    right_x = right_zone[0]

    cv2.line(image, (left_x, 0), (left_x, roi_h), (255, 255, 0), 1)
    cv2.line(image, (right_x, 0), (right_x, roi_h), (255, 255, 0), 1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, "LEFT", (max(5, left_x // 2 - 20), 20),
                font, 0.5, (255, 255, 0), 1)
    cv2.putText(image, "CENTER", (left_x + 10, 20),
                font, 0.5, (255, 255, 0), 1)
    cv2.putText(image, "RIGHT", (right_x + 10, 20),
                font, 0.5, (255, 255, 0), 1)


def draw_decision_overlay(canvas, decision: Decision, free_state: FreeSpaceState,
                          obstacle_state: ObstacleState, fps: float, latency_ms: float):
    """
    在最终叠加图上绘制决策信息、状态栏、FPS、延迟。
    canvas 尺寸: (360, 640, 3) —— 信息面板用。
    """
    h, w = canvas.shape[:2]
    cv2.rectangle(canvas, (0, 0), (w, h), (30, 30, 30), -1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    y = 25

    # 性能指标
    fps_color = (0, 255, 0) if fps >= 15 else (0, 165, 255) if fps >= 10 else (0, 0, 255)
    lat_color = (0, 255, 0) if latency_ms < 100 else (0, 0, 255)
    cv2.putText(canvas, f"FPS: {fps:.1f}", (10, y), font, 0.55, fps_color, 2); y += 25
    cv2.putText(canvas, f"Latency: {latency_ms:.1f}ms", (10, y), font, 0.55, lat_color, 2); y += 30

    # 决策
    cmd_colors = {
        "FORWARD": (0, 255, 0), "TURN_LEFT": (255, 200, 0),
        "TURN_RIGHT": (255, 200, 0), "STOP": (0, 0, 255),
        "SLOW_DOWN": (0, 200, 255), "SEARCH_LANE": (200, 200, 0),
    }
    color = cmd_colors.get(decision.command, (255, 255, 255))
    cv2.putText(canvas, f"CMD: {decision.command}", (10, y), font, 0.7, color, 2); y += 30
    cv2.putText(canvas, f"Speed: {decision.speed:.2f}  Angular: {decision.steering:+.2f}",
                (10, y), font, 0.5, (200, 200, 200), 1); y += 22
    cv2.putText(canvas, f"Reason: {decision.reason[:60]}", (10, y), font, 0.4, (180, 180, 180), 1); y += 25

    # 障碍状态
    cv2.putText(canvas, "Obstacle:", (10, y), font, 0.5, (200, 200, 200), 1); y += 20
    blocked_parts = []
    if obstacle_state.blocked_left: blocked_parts.append("LEFT")
    if obstacle_state.blocked_center: blocked_parts.append("CENTER")
    if obstacle_state.blocked_right: blocked_parts.append("RIGHT")
    blocked_str = ",".join(blocked_parts) if blocked_parts else "NONE"
    block_color = (0, 0, 255) if blocked_parts else (0, 255, 0)
    cv2.putText(canvas, f"  Blocked: {blocked_str}", (10, y), font, 0.45, block_color, 1); y += 18
    cv2.putText(canvas, f"  Danger: {obstacle_state.danger_level:.2f}",
                (10, y), font, 0.45, (200, 200, 200), 1); y += 22

    # 可行驶区域评分
    cv2.putText(canvas, "FreeSpace Score:", (10, y), font, 0.5, (200, 200, 200), 1); y += 20
    fs = free_state
    cv2.putText(canvas, f"  L:{fs.left_free_score:.2f}  C:{fs.center_free_score:.2f}  R:{fs.right_free_score:.2f}",
                (10, y), font, 0.45, (0, 220, 255), 1); y += 18
    cv2.putText(canvas, f"  Offset: {fs.center_offset:+.3f}  Conf: {fs.confidence:.2f}",
                (10, y), font, 0.4, (180, 180, 180), 1)


def create_decision_canvas(height=360, width=640):
    """创建决策叠加图层。"""
    return np.zeros((height, width, 3), dtype=np.uint8)
