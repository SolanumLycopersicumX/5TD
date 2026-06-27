# 隧道 UGV 可行进路面到运动轨迹转换方案

**版本日期**：2026-06-24  
**当前阶段**：方案 A 优先落地；方案 B 作为后续增强路线  
**项目定位**：Safety-Aware Learning-Augmented Navigation System  
**核心目标**：将已识别出的可行进路面与排水渠 / hard-boundary 信息，转换为可执行的局部运动轨迹，并最终输出给底盘驱动。

---

## 0. 一句话结论

当前最稳妥的工程路线是：

```text
RGB 图像
→ ego_passable_mask / hard_boundary_mask / detections
→ BEV / IPM 俯瞰投影
→ Occupancy Grid + Risk Grid
→ DWA 局部轨迹规划
→ Safety State Machine 安全修正
→ target_speed + target_steering / angular_velocity + brake
→ 底盘驱动接口
```

后续可以参考 OpenPilot 的思想加入 **waypoint / trajectory prediction head**，但它只能作为 **candidate trajectory proposal**，不能绕过 risk grid、hard-boundary check 和 safety filter 直接控制电机。

---

## 1. 当前项目状态

目前项目已经基本完成：

- 可行进路面识别（ego-passable surface segmentation）
- 排水渠 / 深沟 / hard-boundary 识别
- RGB-only baseline 方向确认
- HBD-Net-RT 感知框架已有工程骨架
- BEV projector、occupancy grid、risk grid、DWA planner、safety state machine 已有基础模块
- 低层驱动侧已有 CAN / RS232-Modbus 相关控制代码可作为后续对接参考

因此，下一阶段的重点不是继续提高 mask 可视化效果，而是把：

```text
可行进区域 mask
```

真正变成：

```text
车辆可跟踪的局部轨迹 + 安全控制命令
```

---

## 2. 两阶段路线锁定

## 2.1 方案 A：当前最稳妥工程路线

**优先级：最高，立即执行。**

目标是先建立一条可运行、可解释、可验证的工程闭环：

```text
RGB perception
→ BEV risk grid
→ DWA / local planner
→ safety state machine
→ low-level vehicle command
```

该路线的优点：

- 与当前 HBD-Net-RT baseline 最兼容；
- 不需要模型直接学习控制；
- 每一步都可视化、可调参、可测试；
- 对排水渠 / 深沟这类 catastrophic risk 更安全；
- 后续可以自然接入 LiDAR、右侧距离传感器或 OpenPilot-inspired trajectory head。

---

## 2.2 方案 B：OpenPilot-inspired 改进路线

**优先级：后续增强，不作为当前主线。**

目标是在方案 A 跑通之后，加入学习型轨迹预测模块：

```text
RGB / BEV feature
→ waypoint head / trajectory prediction head
→ candidate trajectories
→ risk-grid collision check
→ safety filter
→ final control command
```

关键原则：

```text
learning trajectory ≠ final control command
learning trajectory = candidate proposal
final command = safety-checked trajectory
```

不建议直接照搬 OpenPilot，因为 OpenPilot 面向公路车辆、车道线 / 道路结构、量产车 CAN / ADAS 接口和 L2 驾驶辅助；而本项目是隧道工程 UGV，在无车道线、重复纹理、右侧深沟、高安全约束场景下运行。

---

# Part A：方案 A 工程实施方案

---

## 3. 总体架构

```text
Sensors
  └── Front RGB Camera
        ↓
Perception
  ├── ego_passable_mask
  ├── hard_boundary_mask
  ├── hard_boundary_edge
  ├── detections
  └── confidence
        ↓
Mapping
  ├── BEVProjector / IPM / Homography
  ├── OccupancyGrid
  └── RiskGrid
        ↓
Planning
  └── Risk-Adaptive DWA
        ↓
Safety
  └── SafetyStateMachine
        ↓
Control Adapter
  ├── differential drive adapter
  ├── Ackermann adapter
  ├── CAN adapter
  └── RS232 / Modbus VCU adapter
        ↓
Vehicle Actuation
```

---

## 4. 输入与输出定义

## 4.1 感知输入

```python
image: RGB frame
shape: [H, W, 3]
source: front camera / video / saved image
```

## 4.2 感知输出

```python
perception_output = {
    "ego_passable_mask": Tensor[B, 1, Hm, Wm],
    "hard_boundary_mask": Tensor[B, C, Hm, Wm],
    "hard_boundary_edge": Tensor[B, 1, Hm, Wm],
    "detections": {
        "boxes": Tensor[N, 4],
        "scores": Tensor[N],
        "labels": Tensor[N]
    },
    "confidence": {
        "detection": float,
        "passable": float,
        "boundary": float,
        "overall": float
    }
}
```

## 4.3 规划输出

```python
dwa_output = {
    "target_speed": float,              # m/s
    "target_steering": float,           # rad, or target_steering_angle converted to this key
    "selected_trajectory": list,        # [[x, y, yaw], ...]
    "planner_status": "OK" | "STOP" | "NO_FEASIBLE_PATH",
    "max_risk_on_path": float,
    "min_clearance_m": float
}
```

## 4.4 最终控制输出

```python
final_command = {
    "target_speed": float,
    "target_steering": float,
    "target_angular_velocity": float,   # optional, for differential / VCU interface
    "brake": bool,
    "safety_state": str,
    "reason": str
}
```

---

## 5. Step 1：固定 perception → mapping 接口

当前 perception 模块应继续稳定输出以下核心内容：

| 输出 | 含义 | 后续用途 |
|---|---|---|
| `ego_passable_mask` | 本车所在侧可通行区域 | 生成可通行 BEV 区域 |
| `hard_boundary_mask` | 排水渠、隔离带、隧道壁等硬边界 | 生成 forbidden zone |
| `hard_boundary_edge` | 边界边缘 | 增强边界安全裕度 |
| `detections` | 人、工程车、碎石、悬挂物等 | 叠加语义风险 |
| `confidence.overall` | 感知总体置信度 | 安全状态机降速 / 停车 |

第一阶段不要改变接口，先保证现有 pipeline 能稳定串起来。

---

## 6. Step 2：RGB mask 到 BEV / IPM 投影

## 6.1 第一版：先使用现有简化 BEVProjector 跑通

当前可以先使用简化投影：

```text
image mask
→ simplified BEV projection
→ local grid in vehicle coordinate
```

建议初始 BEV 范围：

```yaml
bev_grid:
  range_x_min: -2.5
  range_x_max: 2.5
  range_y_min: 0.0
  range_y_max: 8.0
  resolution: 0.10
```

含义：

| 坐标 | 含义 |
|---|---|
| `x` | 车辆左右方向，左负右正或按项目约定 |
| `y` | 车辆前进方向 |
| `resolution` | 每个 grid cell 对应的米制尺寸 |

## 6.2 第二版：替换为 Homography / IPM

当 v0 管线跑通后，需要进行相机标定：

```text
camera intrinsic calibration
+ camera height measurement
+ camera pitch measurement
+ ground reference points
→ homography matrix H
→ calibrated IPM / BEV
```

必须验证：

| 验证项 | 目标 |
|---|---|
| 前方 2 m 标记点 | BEV 坐标误差可接受 |
| 前方 5 m 标记点 | BEV 坐标误差可接受 |
| 左右道路宽度 | 与实测宽度一致 |
| 排水渠边缘 | 不被错误投影到可行进区域 |
| 车辆自身宽度 | 在 BEV 中占据尺度正确 |

---

## 7. Step 3：Occupancy Grid 生成

Occupancy grid 是二值地图：

```text
0 = 可通行
1 = 占用 / 禁止通过
```

推荐规则优先级：

| 优先级 | 规则 | occupancy |
|---:|---|---:|
| 1 | hard_boundary 区域 | 1 |
| 2 | ego_passable 外部区域 | 1 |
| 3 | detection bbox 投影区域，带安全膨胀 | 1 |
| 4 | ego_passable 内部区域 | 0 |

核心原则：

```text
排水渠 / 深沟 / hard-boundary 永远不可跨越。
```

---

## 8. Step 4：Risk Grid 生成

Risk grid 是连续风险地图：

```text
0.0 = 低风险
1.0 = 最高风险 / 禁止通行
```

推荐风险规则：

| 区域 | risk 建议值 |
|---|---:|
| 可行进路面内部 | 0.0–0.2 |
| 靠近排水渠边缘 | 0.6–0.9 |
| hard-boundary / 排水渠 / 深沟 | 1.0 |
| ego-passable 外部 | 1.0 |
| 工人 | 0.95–1.0 |
| 工程车辆 | 0.95 |
| 悬挂物 | 0.80 |
| 碎石 / 掉落物 | 0.65–0.85 |
| 未知障碍物 | 0.7–0.9 |
| 低置信度区域 | 全局 risk bias |

建议加入边界膨胀：

```text
hard_boundary dilation
+ vehicle half width
+ safety margin
→ keep-out zone
```

---

## 9. Step 5：DWA 局部轨迹规划

DWA 的作用是：

```text
在速度 v 和转角 δ 的采样空间中，模拟未来短时间轨迹，并选择风险最低、 clearance 最大、动作最平滑、前进效率最高的轨迹。
```

## 9.1 DWA 输入

```python
dwa_input = {
    "current_pose": [0.0, 0.0, 0.0],
    "current_velocity": current_v,
    "risk_grid": risk_grid,
    "grid_extent": grid_metadata,
    "goal_direction": [0.0, 1.0],
    "vehicle_profile": vehicle_profile
}
```

## 9.2 DWA 采样

示例配置：

```yaml
dwa:
  min_velocity_ms: 0.2
  max_velocity_ms: 1.5
  velocity_samples: 5
  min_steering_rad: -0.5
  max_steering_rad: 0.5
  steering_samples: 9
  predict_time_s: 2.0
  dt_s: 0.1
```

## 9.3 DWA 代价函数

推荐代价函数：

```text
score =
  w_clearance  × clearance_score
+ w_risk       × risk_score
+ w_smoothness × smoothness_score
+ w_progress   × progress_score
```

建议权重：

```yaml
cost_weights:
  clearance: 0.35
  risk_cost: 0.25
  smoothness: 0.25
  progress: 0.15
```

## 9.4 DWA 必须满足的硬约束

| 约束 | 处理方式 |
|---|---|
| 轨迹穿越 hard-boundary | 直接剔除 |
| 轨迹进入 ego-passable 外部 | 直接剔除 |
| 最小 clearance 不足 | 直接剔除或大幅惩罚 |
| 轨迹最大 risk 超过 stop threshold | STOP |
| 没有可行轨迹 | STOP / manual takeover |

---

## 10. Step 6：Safety State Machine 安全修正

DWA 输出不能直接发给驱动，必须经过安全状态机。

## 10.1 安全状态

| 状态 | 含义 | 控制限制 |
|---|---|---|
| `S0_NORMAL` | 正常行驶 | 速度比例 1.0 |
| `S1_CAUTIOUS` | 谨慎行驶 | 限速 50% |
| `S2_SLOWDOWN` | 明显风险 | 限速 25% |
| `S3_STOP` | 停车 | speed = 0, brake = True |
| `S4_MANUAL_TAKEOVER` | 人工接管 | speed = 0, brake = True, alarm |

## 10.2 状态机输入

```python
safety_input = {
    "overall_confidence": confidence_overall,
    "max_risk": max_risk,
    "boundary_distance": boundary_distance_m,
    "worker_distance": worker_distance_m,
    "has_feasible_path": planner_status == "OK"
}
```

## 10.3 推荐触发规则

| 条件 | 动作 |
|---|---|
| 感知置信度低 | 降速或停车 |
| risk 高 | 降速或停车 |
| 距 hard-boundary 太近 | 降速 |
| 工人距离小于阈值 | 停车 |
| 无可行轨迹 | 停车 |
| 连续多帧异常 | 人工接管 |

---

## 11. Step 7：控制命令适配到底盘

## 11.1 规划层统一输出

建议规划层统一输出：

```python
command = {
    "target_speed": v,          # m/s
    "target_steering": delta,   # rad
    "brake": brake,
    "safety_state": state,
    "reason": reason
}
```

## 11.2 如果底盘是差速驱动

差速驱动需要转换为左右轮速度：

```text
v_left  = v - ω × track_width / 2
v_right = v + ω × track_width / 2
```

其中：

```text
ω ≈ v × tan(δ) / wheelbase
```

最终可输出：

```python
set_speed(left_speed_cmd, right_speed_cmd)
```

## 11.3 如果底盘使用 RS232 / Modbus VCU

若底盘控制器支持线速度和角速度寄存器，则可以直接输出：

```text
linear velocity  → register 1040, unit 0.001 m/s
angular velocity → register 1041, unit 0.001 rad/s
```

即：

```python
linear_cmd  = int(v * 1000)
angular_cmd = int(omega * 1000)
```

## 11.4 如果底盘是 Ackermann

则保留：

```python
speed_cmd = target_speed
steering_cmd = target_steering
```

并交给底盘转向执行器。

---

## 12. 需要优先修复的接口问题

建议优先统一 DWA 与 safety/control 之间的 steering key。

可能存在：

```python
"target_steering_angle"
```

和：

```python
"target_steering"
```

两个字段不一致的问题。

建议统一为：

```python
"target_steering"
```

或者在 safety 层兼容：

```python
target_steering = dwa_output.get(
    "target_steering",
    dwa_output.get("target_steering_angle", 0.0)
)
```

否则可能出现：

```text
DWA 已经规划出转向角，但最终控制命令 steering 变成 0。
```

---

## 13. 可视化与调试面板

建议 dashboard 至少显示以下内容：

| 面板 | 内容 |
|---|---|
| Original | 原始 RGB 图像 |
| Ego-Passable | 可行进路面 mask |
| Hard-Boundary | 排水渠 / 隔离带 / 隧道壁 mask |
| BEV Passable | 投影后的可通行区域 |
| Occupancy Grid | 二值占用图 |
| Risk Grid | 连续风险图 |
| DWA Trajectories | 候选轨迹 + 最优轨迹 |
| Command Stats | speed、steering、brake、safety state、reason |

调试目标：

```text
必须能一眼看出：
为什么车选择这条轨迹，为什么降速，为什么停车。
```

---

## 14. 场景测试设计

## 14.1 必测场景

| 场景编号 | 场景 | 期望结果 |
|---|---|---|
| S01 | 空旷直道 | 正常直行 |
| S02 | 轻微靠近排水渠 | 向安全区域修正，必要时降速 |
| S03 | 前方有小障碍 | 绕行或减速 |
| S04 | 前方大障碍堵死 | STOP |
| S05 | hard-boundary 误靠近 | 不跨越边界，降速或停车 |
| S06 | 工人出现 | 减速 / 停车 |
| S07 | 低置信度 mask | 降速或停车 |
| S08 | BEV 投影异常 | STOP / manual takeover |
| S09 | 光照突变 | 保守运行 |
| S10 | 水渍 / 反光区域 | 提高 risk，低速通过或绕行 |

## 14.2 验收标准

| 项目 | 合格标准 |
|---|---|
| 轨迹不穿越排水渠 | 100% 必须满足 |
| hard-boundary 不可跨越 | 100% 必须满足 |
| 无可行轨迹时停车 | 必须满足 |
| 工人近距离停车 | 必须满足 |
| 输出控制命令稳定 | steering 不明显抖动 |
| BEV 误差可控 | 标定后满足实车安全边界 |
| pipeline 延迟 | 满足实时闭环要求 |

---

## 15. 方案 A 阶段计划

## A0：端到端最小闭环

目标：

```text
单张 RGB / 视频帧
→ perception output
→ BEV risk grid
→ DWA selected trajectory
→ final command
```

验收：

- 能显示 BEV risk grid；
- 能显示 selected trajectory；
- 能输出 `target_speed` 和 `target_steering`；
- 无路可走时输出 STOP；
- hard-boundary 不被轨迹穿越。

---

## A1：BEV 投影可信化

目标：

```text
从“看起来像 BEV”
变成
“米制坐标可信的 BEV”
```

任务：

- 测量相机高度；
- 测量俯仰角；
- 采集地面标定点；
- 求 Homography；
- 验证 BEV 中 1 m 是否对应实地 1 m；
- 验证排水渠边缘位置误差。

---

## A2：轨迹稳定化

可能问题：

- mask 边缘抖动；
- risk grid 跳变；
- steering 左右抖动；
- 轨迹贴近排水渠；
- 低速时转角过大。

解决：

```text
risk grid temporal smoothing
+ steering low-pass filter
+ trajectory smoothing
+ minimum trench margin constraint
+ steering rate limit
```

示例：

```python
steering_cmd = alpha * steering_new + (1 - alpha) * steering_last
```

并限制：

```text
|δ_t - δ_{t-1}| < max_delta_per_frame
```

---

## A3：真实驱动接口

目标：

```text
target_speed / target_steering
→ chassis-specific control adapter
→ motor / VCU command
```

需要确认：

| 问题 | 说明 |
|---|---|
| 底盘类型 | differential / Ackermann / four-wheel independent |
| 控制接口 | CAN / RS232 / Modbus / PWM |
| 速度单位 | m/s, rpm, raw command |
| 转向单位 | rad, rad/s, raw command |
| 急停接口 | 软件急停 + 硬件急停 |
| 状态反馈 | voltage, current, encoder, fault code |

---

# Part B：OpenPilot-inspired 后续增强路线

---

## 16. 为什么暂时不做方案 B

当前不建议直接做 OpenPilot-inspired 的端到端轨迹预测，因为：

- 当前更需要可靠闭环 baseline；
- 隧道无车道线，和 OpenPilot 公路场景不同；
- 右侧深沟风险不能交给纯学习模型；
- 现场工程安全要求高于 demo；
- 没有足够真实驾驶 / 遥操作轨迹数据训练 waypoint head。

---

## 17. 方案 B 的正确接入方式

方案 B 应作为 `TrajectoryProposalModule`：

```text
BEV feature / RGB feature
→ waypoint head
→ candidate trajectories
→ risk check
→ safety filter
→ final command
```

接口建议：

```python
trajectory_proposal = {
    "candidate_trajectories": [
        [[x1, y1, yaw1], [x2, y2, yaw2], ...],
        [[x1, y1, yaw1], [x2, y2, yaw2], ...]
    ],
    "confidence": float,
    "source": "waypoint_head" | "rl" | "diffusion" | "dwa"
}
```

所有 candidate 都必须经过：

```text
risk_grid collision check
+ hard-boundary check
+ trench margin check
+ vehicle kinematic check
+ safety state machine
```

---

## 18. 未来训练数据来源

方案 B 需要数据，可从以下来源收集：

| 数据来源 | 用途 |
|---|---|
| DWA 生成轨迹 | 初始 imitation learning label |
| 人工遥操作轨迹 | 学习人类安全驾驶偏好 |
| 仿真隧道环境 | 生成大量边界 / 障碍场景 |
| 实车低速测试 | fine-tuning / validation |
| 失败场景回放 | hard negative mining |

---

## 19. 当前不要做的内容

在方案 A 跑通前，不建议做：

- RGB 直接输出电机 PWM；
- Transformer 直接控制底盘；
- RL 直接控制速度和转角；
- Diffusion 直接绕过 safety filter；
- OpenPilot 模型完整复现；
- VLM / LLM 参与实时低层控制。

这些都应放在方案 B 或 research track 中。

---

# 20. 最终执行清单

## 20.1 立即执行

```text
1. 统一 DWA → Safety → Control 的输出字段
2. 确认 perception output 稳定输出 ego_passable / hard_boundary
3. 用当前 BEVProjector 跑通 mask → BEV grid
4. 生成 occupancy grid 和 risk grid
5. 接 DWA，显示 selected trajectory
6. 接 SafetyStateMachine，输出 final command
7. 增加 dashboard 可视化
8. 增加场景测试
```

## 20.2 第二阶段执行

```text
1. 做相机标定
2. 替换 Homography / IPM
3. 验证 BEV 米制误差
4. 加 steering smoothing
5. 加 trench margin check
6. 接真实底盘控制接口
```

## 20.3 第三阶段执行

```text
1. 采集遥操作轨迹
2. 训练 waypoint / trajectory prediction head
3. 作为 candidate trajectory proposal 接入
4. 保留 DWA / safety filter 作为最终裁决层
```

---

# 21. 推荐项目表述

可以在项目文档中这样描述本阶段技术路线：

> 本阶段采用 RGB-to-BEV 的工程闭环路线。系统首先通过 RGB 分割网络识别本车侧可行进路面与排水渠 / hard-boundary 区域，再通过 BEV / IPM 将图像空间的语义 mask 投影到车辆局部地面坐标系中，生成 occupancy grid 与 semantic risk grid。随后使用 Risk-Adaptive DWA 在风险地图上采样并评估短时运动轨迹，选择满足车辆运动学、安全距离和边界约束的最优轨迹。最终控制命令必须经过 safety state machine 修正，以实现人员停车、排水渠防越界、低置信度降速、无可行路径停车和人工接管等安全逻辑。OpenPilot-inspired 的 waypoint prediction 可作为后续增强模块，但只作为候选轨迹生成器，不直接输出底层驱动命令。

---

## 22. 最终总结

当前阶段的核心任务是：

```text
可行进路面识别
→ 可行进轨迹生成
→ 安全控制命令输出
```

最稳妥路线是：

```text
RGB mask
→ BEV risk grid
→ DWA trajectory
→ safety command
→ chassis control
```

OpenPilot-inspired 路线保留为后续增强：

```text
learning-based waypoint proposal
→ safety checked trajectory
→ final command
```

不要让学习模型绕过安全层。

