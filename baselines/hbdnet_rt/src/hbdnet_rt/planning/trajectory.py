"""
轨迹前向模拟。自行车模型: 给定起点、速度、转向角 → 生成离散轨迹点。
"""
import numpy as np
from typing import List


def simulate_trajectory(
    start_pose: List[float],
    velocity: float,
    steering: float,
    wheelbase: float,
    dt: float,
    steps: int,
) -> np.ndarray:
    """
    自行车模型前向模拟。
    start_pose: [x, y, yaw] — 起点 (m, m, rad)
    velocity: 线速度 (m/s)
    steering: 前轮转向角 (rad)
    wheelbase: 轴距 (m)
    dt: 模拟步长 (s)
    steps: 步数
    返回: [steps+1, 3] — 每行 [x, y, yaw]
    """
    x, y, yaw = float(start_pose[0]), float(start_pose[1]), float(start_pose[2])
    traj = [(x, y, yaw)]

    for _ in range(steps):
        if abs(steering) < 1e-6:
            # 直行
            x += velocity * np.cos(yaw) * dt
            y += velocity * np.sin(yaw) * dt
        else:
            # 转弯: 自行车模型
            turn_radius = wheelbase / np.tan(steering)
            angular_vel = velocity / turn_radius
            yaw += angular_vel * dt
            x += velocity * np.cos(yaw) * dt
            y += velocity * np.sin(yaw) * dt
        traj.append((x, y, yaw))

    return np.array(traj, dtype=np.float32)


def simulate_trajectory_vectorized(
    velocities: np.ndarray,
    steerings: np.ndarray,
    start_pose: List[float],
    wheelbase: float,
    dt: float,
    steps: int,
) -> np.ndarray:
    """
    批量轨迹生成 (向量化优化)。
    velocities: [N_vel], steerings: [N_steer]
    返回: [N_vel, N_steer, steps+1, 3]
    """
    nv, ns = len(velocities), len(steerings)
    x0, y0, yaw0 = float(start_pose[0]), float(start_pose[1]), float(start_pose[2])
    traj = np.zeros((nv, ns, steps + 1, 3), dtype=np.float32)
    traj[:, :, 0, :] = [x0, y0, yaw0]

    for i in range(steps):
        px, py, pyaw = traj[:, :, i, 0], traj[:, :, i, 1], traj[:, :, i, 2]
        v = velocities[:, None]                      # [N_vel, 1]
        steer = steerings[None, :]                   # [1, N_steer]

        # 转向半径 (直行时用大值避免除零)
        eps = 1e-9
        tan_s = np.tan(steer)
        turn_radius = np.where(np.abs(tan_s) > eps, wheelbase / tan_s, 1e9)

        angular_vel = np.where(np.abs(tan_s) > eps, v / turn_radius, 0.0)
        new_yaw = pyaw + angular_vel * dt
        new_x = px + v * np.cos(new_yaw) * dt
        new_y = py + v * np.sin(new_yaw) * dt

        traj[:, :, i + 1, 0] = new_x
        traj[:, :, i + 1, 1] = new_y
        traj[:, :, i + 1, 2] = new_yaw

    return traj
