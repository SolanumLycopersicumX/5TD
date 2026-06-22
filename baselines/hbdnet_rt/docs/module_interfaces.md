# 模块接口说明

## 感知模块 (perception)

```
PerceptionInference
  输入: torch.Tensor [B, 3, 384, 640]  归一化 RGB 图像
  输出: {
    "detections": {
      "boxes": Tensor[N, 4],      # [x1, y1, x2, y2] 原图坐标
      "scores": Tensor[N],        # 置信度 0~1
      "labels": Tensor[N]         # 0=construction_vehicle, 1=worker, 2=suspended, 3=debris
    },
    "ego_passable_mask": [B, 1, 96, 160],   # 本车侧可通行区域 (二值)
    "hard_boundary_mask": [B, 5, 96, 160],  # 4类 + 合并 (softmax 概率)
    "hard_boundary_edge": [B, 1, 96, 160],  # 边界边缘 (二值)
    "confidence": {
      "detection": float,    # 检测置信度 (mean score)
      "passable": float,     # 可行驶区域置信度
      "boundary": float,     # 硬边界置信度
      "overall": float       # 综合 = 0.4×det + 0.3×passable + 0.3×boundary
    }
  }

HBDNetRT (模型)
  同 PerceptionInference.forward() 但输出为原始 logits
  (未经过 sigmoid/softmax/threshold)
```

## 栅格模块 (mapping)

```
BEVProjector
  属性:
    grid_shape: (nx, ny)  # 横向/纵向栅格单元数
    grid_extent: {x_min, x_max, y_min, y_max, resolution}
  方法:
    project_mask_to_bev(mask) → Tensor [B, C, ny, nx]
    image_xy_to_grid_xy(px, py) → (x_m, y_m)
    set_homography(H: np.ndarray)  # 预留

OccupancyGrid
  输入: 感知输出字典
  输出: {"occupancy_grid": [1,1,ny,nx], "metadata": {...}}
  规则: hard_boundary→占用, ego_passable外→占用, detection bbox→占用(带膨胀)

RiskGrid
  输入: 感知输出字典 + occupancy_grid (可选)
  输出: {"risk_grid": [1,1,ny,nx], "max_risk": float, "metadata": {...}}
  规则: hard_boundary→1.0, ego_passable外→1.0, detection→按类别, 低置信度→加偏置
```

## DWA 模块 (planning)

```
DWAPlanner
  输入: {
    "current_pose": [x, y, yaw],       # 当前位姿 (m, m, rad)
    "current_velocity": float,          # 当前速度 (m/s)
    "risk_grid": ndarray [ny, nx],      # 风险栅格
    "grid_extent": dict,                # 坐标范围
    "goal_direction": float (可选)      # 目标方向
  }
  输出: {
    "target_speed": float,              # 最优速度 (m/s)
    "target_steering": float,           # 最优转向角 (rad)
    "selected_trajectory": list,        # [[x,y,yaw],...] 最优轨迹
    "planner_status": "OK" | "NO_PATH",
    "cost_breakdown": {total_score, clearance, smoothness, progress, collision, max_risk},
    "num_candidates": int,
    "num_feasible": int
  }
  规则: 穿越 hard-boundary (risk≥0.99) → 碰撞, 出 grid 范围 → 碰撞, 全部碰撞 → NO_PATH
```

## 安全状态机 (safety)

```
SafetyStateMachine
  输入: {
    "overall_confidence": float,    # 感知综合置信度
    "max_risk": float,              # risk_grid 最大风险值
    "boundary_distance": float,     # 距硬边界最近距离 (m)
    "worker_distance": float,       # 距最近工人距离 (m)
    "has_feasible_path": bool       # DWA 是否有可行轨迹
  }
  输出: {
    "safety_state": str,            # S0_NORMAL ~ S4_MANUAL_TAKEOVER
    "speed_limit_ratio": float,     # 速度上限比例
    "brake": bool,
    "reason": str                   # 状态转换原因
  }

  apply_to_dwa(dwa_output, safety_output) → 最终控制命令:
    STOP / TAKEOVER → speed=0, steering=0, brake=True
    CAUTIOUS / SLOWDOWN → speed×limit, steering 保留
    NORMAL → 保留 DWA 输出
```

## 控制命令 (control)

```
ControlCommand (dataclass)
  timestamp: float
  target_speed: float         # m/s, STOP/TAKEOVER 强制为 0
  target_steering: float      # rad, STOP/TAKEOVER 强制为 0
  brake: bool
  safety_state: str
  reason: str
  debug: dict

generate_command(dwa_output, safety_output) → ControlCommand
  DWA 输出 + 安全状态机修正 → 最终控制指令
