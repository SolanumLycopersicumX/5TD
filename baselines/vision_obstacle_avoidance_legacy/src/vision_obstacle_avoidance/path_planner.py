"""
车道内路径规划模块 (#46-47)。
在检测到的车道边界内进行简化的 DWA (Dynamic Window Approach) 路径规划。

算法:
  1. 可通行性预检: 车道宽度 >= 车宽 + 2×安全余量
  2. 生成离散曲率候选轨迹（圆弧）
  3. 对每条轨迹评分: 安全间距 + 平滑性 + 前进进度
  4. 选最优轨迹，输出 angular + speed
  5. 无可行轨迹 → passable=False → 由决策层 STOP

v2.1 修复:
  - 阿克曼几何非线性转向映射 (atan)
  - dt-based 帧间转向限幅
  - 占用区域精细化投影
  - 碰撞检测向量化

v2.2 改动:
  - 从阿克曼转向切换为差速驱动
  - 曲率 → 角速度映射: ω = v·κ (替代 atan(κ·wheelbase))
  - 帧间限幅改为角加速度限制 (MAX_ANGULAR_ACCEL_RADPS2)
"""

import math
import time
import numpy as np
import cv2

import config
from utils import PathPlan


class PathPlanner:
    """
    车道约束内的简化 DWA 路径规划器 (v2.2: 差速驱动)。

    输入: lane_boundary, debris_state, free_space_state, obstacle_state,
          calibrator, vehicle_width_m, safety_margin_m

    输出: PathPlan (steering 字段在差速模式下表示归一化角速度 [-1, 1])
    """

    def __init__(self):
        self._last_angular = 0.0       # v2.2: 上一帧的归一化角速度 (原 _last_steering)
        self._last_time = time.time()

    # ── 公开接口 ──────────────────────────────────────────────────────────

    def plan(self, lane_boundary=None, debris_state=None,
             free_space_state=None, obstacle_state=None,
             calibrator=None, vehicle_width_m=None,
             safety_margin_m=None) -> PathPlan:
        """规划车道内路径。"""
        plan = PathPlan()

        vw = vehicle_width_m or config.VEHICLE_WIDTH_M
        sm = safety_margin_m or config.SAFETY_MARGIN_M

        # 计算 dt
        now = time.time()
        dt = now - self._last_time
        self._last_time = now
        if dt <= 0 or dt > 1.0:
            dt = 1.0 / 30.0  # 默认 30 FPS

        # ---- 1. 可通行性预检 ----
        if lane_boundary is None or not lane_boundary.is_valid:
            plan.passable = True
            plan.is_valid = True
            plan.steering = 0.0
            plan.speed = config.DEFAULT_SPEED
            return plan

        lane_w = lane_boundary.lane_width_m
        if lane_w > 0:
            min_width = vw + 2 * sm
            if lane_w < min_width:
                plan.passable = False
                plan.blocked_reason = f"LANE_TOO_NARROW: {lane_w:.2f}m < {min_width:.2f}m"
                return plan

        # ---- 2. 生成候选曲率 ----
        curvatures = self._generate_curvatures()
        lookahead_m = config.PATH_LOOKAHEAD_M
        step_m = config.PATH_STEP_LENGTH_M
        n_steps = int(lookahead_m / step_m)

        best_score = -float("inf")
        best_curvature = 0.0

        # v2.1: 占用区域精细化
        occupied = self._build_occupied_zones(
            lane_boundary, debris_state, obstacle_state,
            free_space_state, calibrator)

        # v2.1: 向量化碰撞检测 — 预转换 occupied 为 numpy 数组
        occ_np = np.array(occupied, dtype=np.float64) if occupied else np.zeros((0, 4))

        for curv in curvatures:
            score = self._evaluate_trajectory(
                curv, n_steps, step_m, occ_np, lane_boundary)
            if score > best_score:
                best_score = score
                best_curvature = curv

        # ---- 3. 判断是否有可行路径 ----
        if best_score < -100:
            plan.passable = False
            plan.blocked_reason = "OBSTACLE: no clear path"
            return plan

        # ---- 4. 组装输出 ----
        plan.is_valid = True
        plan.passable = True
        # v2.2: 曲率 → 归一化角速度 (差速驱动)
        plan.steering = self._curvature_to_angular(best_curvature, plan.speed)
        plan.speed = self._adapt_speed(best_score)
        plan.clearance_cm = max(0, best_score) * 100

        # v2.2: 差速驱动 — 角加速度限幅 (rad/s² 约束)
        max_delta = config.MAX_ANGULAR_ACCEL_RADPS2 / config.MAX_ANGULAR_VELOCITY_RADPS * dt
        delta = plan.steering - self._last_angular
        delta = max(-max_delta, min(max_delta, delta))
        plan.steering = self._last_angular + delta
        self._last_angular = plan.steering

        return plan

    # ── 内部 ──────────────────────────────────────────────────────────────

    @staticmethod
    def _generate_curvatures() -> list:
        """生成离散曲率候选。返回曲率列表 (1/m)，0=直行，正=右转。"""
        n = config.PATH_NUM_CURVATURES
        max_k = config.PATH_MAX_CURVATURE
        if n % 2 == 0:
            n += 1
        half = n // 2
        return [max_k * i / half for i in range(-half, half + 1)]

    def _evaluate_trajectory(self, curvature: float, n_steps: int,
                              step_m: float, occ_np: np.ndarray,
                              lane_boundary) -> float:
        """
        评估一条轨迹。v2.1: 向量化碰撞检测。

        返回综合得分（越高越好）。
        occ_np: (N, 4) numpy 数组 [(x_min, x_max, y_min, y_max), ...]
        """
        if lane_boundary is not None and lane_boundary.left_barrier_px is not None \
                and lane_boundary.ditch_px is not None:
            start_x = lane_boundary.lane_width_m / 2.0
        else:
            start_x = config.HALF_LANE_WIDTH_M / 2.0

        x, y, theta = start_x, 0.0, 0.0
        min_clearance = float("inf")
        progress = 0.0
        inner_wheel_offset = config.PATH_INNER_WHEEL_OFFSET_M

        for _ in range(n_steps):
            if abs(curvature) < 1e-6:
                x_new, y_new = x, y + step_m
            else:
                radius = 1.0 / curvature
                dtheta = step_m / radius
                x_new = x + radius * (math.cos(theta) - math.cos(theta + dtheta))
                y_new = y + radius * (math.sin(theta + dtheta) - math.sin(theta))
                theta += dtheta

            progress += step_m

            if occ_np.size > 0:
                # v2.1: 向量化距离计算
                dists = self._dist_to_rects_vectorized(x_new, y_new, occ_np)
                batch_min = float(np.min(dists))
                if batch_min < min_clearance:
                    min_clearance = batch_min
                if batch_min < config.PATH_CLEARANCE_MIN_CM / 100.0:
                    return -1000

            # 内轮差
            if abs(curvature) > 1e-6 and occ_np.size > 0:
                inner_x = x_new - inner_wheel_offset if curvature > 0 else x_new + inner_wheel_offset
                dists = self._dist_to_rects_vectorized(inner_x, y_new, occ_np)
                if np.any(dists < config.PATH_CLEARANCE_MIN_CM / 100.0):
                    return -1000

            x, y = x_new, y_new

        # 出界检查
        lane_w = (lane_boundary.lane_width_m
                  if (lane_boundary is not None and lane_boundary.lane_width_m > 0)
                  else config.HALF_LANE_WIDTH_M)
        if x < 0 or x > lane_w:
            return -500

        clearance_score = min_clearance if min_clearance != float("inf") else 5.0
        smoothness = 1.0 - abs(curvature) / config.PATH_MAX_CURVATURE
        progress_score = progress / config.PATH_LOOKAHEAD_M

        return (config.PATH_WEIGHT_CLEARANCE * clearance_score +
                config.PATH_WEIGHT_SMOOTHNESS * smoothness +
                config.PATH_WEIGHT_PROGRESS * progress_score)

    @staticmethod
    def _dist_to_rects_vectorized(x: float, y: float, rects: np.ndarray) -> np.ndarray:
        """向量化点到矩形距离计算。rects: (N,4) [x1, x2, y1, y2]"""
        cx = np.maximum(rects[:, 0], np.minimum(x, rects[:, 1]))
        cy = np.maximum(rects[:, 2], np.minimum(y, rects[:, 3]))
        return np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    @staticmethod
    def _build_occupied_zones(lane_boundary, debris_state,
                               obstacle_state, free_space_state,
                               calibrator) -> list:
        """
        构建不可通行区域列表 (在车道地面坐标系中)。
        v2.1: 精细化投影 — 基于评分比例而非硬编码封死整个区域。
        返回 [(x_min_m, x_max_m, y_min_m, y_max_m), ...]
        """
        occupied = []
        lane_w = lane_boundary.lane_width_m if lane_boundary.lane_width_m > 0 \
            else config.HALF_LANE_WIDTH_M

        # 硬边界: 车道两侧
        occupied.append((-0.3, 0.0, 0.0, config.PATH_LOOKAHEAD_M))
        occupied.append((lane_w, lane_w + 0.3, 0.0, config.PATH_LOOKAHEAD_M))

        # 碎石斑块
        if debris_state is not None and debris_state.has_debris:
            for _, _, _, _, sz_cm in debris_state.debris_boxes:
                if sz_cm >= config.DEBRIS_LARGE_CM:
                    cx = lane_w / 2
                    r = max(0.1, sz_cm / 200.0)
                    occupied.append((cx - r, cx + r, 2.0, 4.0))

        # v2.1: 基于自由空间评分的精细化占用
        if free_space_state is not None:
            # 左区
            if free_space_state.left_free_score < 0.3:
                if free_space_state.left_free_score == 0:
                    occupied.append((0.0, lane_w * 0.2, 1.0, 5.0))  # 完全阻塞
                else:
                    occupied.append((0.0, lane_w * 0.1, 1.0, 5.0))  # 部分占用 50%
            # 中区
            if free_space_state.center_free_score < 0.3:
                if free_space_state.center_free_score == 0:
                    occupied.append((lane_w * 0.2, lane_w * 0.8, 1.0, 5.0))
                else:
                    # 部分阻塞: 缩小占用范围
                    center = lane_w * 0.5
                    half_w = lane_w * 0.1
                    occupied.append((center - half_w, center + half_w, 1.5, 4.0))
            # 右区
            if free_space_state.right_free_score < 0.3:
                if free_space_state.right_free_score == 0:
                    occupied.append((lane_w * 0.8, lane_w, 1.0, 5.0))
                else:
                    occupied.append((lane_w * 0.9, lane_w, 1.0, 5.0))

        return occupied

    @staticmethod
    def _curvature_to_angular(curvature: float, speed: float = None) -> float:
        """
        v2.2 差速驱动: 曲率 → 归一化角速度。
        差速运动学: ω = v · κ
        归一化: angular_norm = ω / ω_max, 钳制到 [-1, 1]。

        曲率 κ (1/m): 正=右转, 负=左转
        角速度 ω (rad/s): 正=右转(顺时针), 负=左转(逆时针)
        """
        if abs(curvature) < 1e-6:
            return 0.0
        # 实际线速度 (m/s)
        actual_v = (speed if speed and speed > 0.01 else config.DEFAULT_SPEED) * config.MAX_LINEAR_SPEED_MPS
        # ω = v · κ
        angular_radps = actual_v * curvature
        # 归一化
        angular_norm = angular_radps / config.MAX_ANGULAR_VELOCITY_RADPS
        return max(-1.0, min(1.0, angular_norm))

    @staticmethod
    def _adapt_speed(score: float) -> float:
        """根据路径得分调节速度。"""
        if score > 2.0:
            return config.DEFAULT_SPEED
        elif score > 0.5:
            return config.DEFAULT_SPEED * 0.7
        else:
            return config.SLOW_SPEED
