# 隧道工程无人车导航项目评估报告（LiDAR + RGB + Transformer 更新版）

**项目名称（建议）**：面向隧道工程场景的 **激光雷达-视觉融合安全感知学习增强无人车导航系统**
**英文标题（建议）**：Safety-Aware LiDAR-RGB Fusion and Transformer-Assisted Navigation for Tunnel UGVs
**版本日期**：2026-06-22
**更新依据**：项目负责人建议采用 **雷达 + RGB 相机**，并提到可以考虑 **Transformer**；本版报告将“激光雷达 + RGB”作为主传感器路线，并说明 Transformer 在该项目中的合理应用方式。
**重要假设**：本文中的“雷达”默认指 **激光雷达（LiDAR）**，而不是毫米波雷达（mmWave Radar）。如果项目负责人实际指的是毫米波雷达，则需要额外修改传感器和算法路线。

---

## 0. 一句话结论

本项目不建议做成“深度相机 + 端到端 RL 直接控制小车”，而应改成：

> **LiDAR 提供几何安全底座，RGB 相机提供语义识别，Transformer 用于多模态融合、时序理解和轨迹/策略生成，最终输出必须经过 costmap、safety filter 和低层控制器。**

更推荐的技术路线是：

```text
LiDAR Geometry + RGB Semantics
        ↓
Transformer-based Fusion / Temporal Understanding
        ↓
Semantic Risk Costmap / Traversability Map
        ↓
Classical Planner / MPC Baseline
        ↓
RL / Decision Transformer / Diffusion Trajectory Proposal as Assistant
        ↓
Safety Filter / Emergency Stop / Human Override
        ↓
Low-level Vehicle Control
```

这里的关键点是：

- **LiDAR（激光雷达）**：负责障碍物距离、深沟边界、铁轨边界、可通行区域、局部 costmap。
- **RGB Camera（彩色相机）**：负责人员、工程车辆、工具、线缆、箱体、未知物体等语义识别。
- **Transformer**：不要只当作“炫技模型”，应主要用于 **LiDAR-RGB 融合、时序场景理解、轨迹候选生成、策略选择**。
- **RL / Diffusion / VLA**：只能作为辅助模块，不能绕过安全层直接输出电机 PWM。
- **Safety Filter（安全过滤器）**：负责防撞、防坠入深沟、人员停车、限速、急停和人工接管。

---

## 1. 项目场景重新定义

### 1.1 现场任务

你的无人车需要在隧道内的一侧前进。现场大致特征是：

| 方向 | 场景元素 | 风险 |
|---|---|---|
| 左侧 | 杂物、电机、工程工具、线缆、材料 | 碰撞、缠绕、遮挡 |
| 右侧 | 铁轨、很宽的深沟 / 排水槽 | 高风险，可能导致车辆侧翻或坠落 |
| 前方 | 隧道道路、人员、未知障碍、工程车 | 避障、停车、绕行 |
| 环境 | 隧道重复结构、光照变化、粉尘、水渍、金属反光 | 视觉定位困难，传感器噪声高 |

因此，这个项目虽然表面上像“巡线小车”，但本质不是传统 line following，而是：

> **无固定线索、重复环境、强安全约束下的局部自主导航（local navigation under safety constraints）。**

### 1.2 为什么不适合传统巡线 / 普通导航？

| 方法 | 为什么不适合 |
|---|---|
| 传统巡线 | 隧道内没有稳定可跟踪的线，且道路边界不规则 |
| 纯视觉 SLAM | 隧道重复纹理、弱光、粉尘、金属反光会导致特征不稳定 |
| 纯 GPS | 隧道内通常无 GPS 或信号极弱 |
| 纯深度相机 | 有效距离、FOV、光照/反光/粉尘鲁棒性不足 |
| 纯端到端 RL | 安全不可解释，数据需求大，现场风险高 |
| 纯大模型控制 | 延迟、幻觉、不可验证，不适合直接闭环控制底盘 |

---

## 2. 为什么主方案应采用 LiDAR + RGB，而不是深度相机？

项目负责人建议使用 **雷达 + RGB 相机**，这个判断是合理的。对于隧道 UGV，LiDAR + RGB 的分工比 RGB-D 更清晰。

### 2.1 深度相机的主要问题

| 问题 | 对项目的影响 |
|---|---|
| 有效距离较短 | 大型 UGV 需要提前感知障碍，否则刹车距离不足 |
| FOV 有限 | 右侧深沟、左侧杂物和前方障碍可能无法同时覆盖 |
| 受弱光、强反光、水渍、粉尘影响 | 隧道工程环境中深度图容易出现空洞、噪声或丢失 |
| 对安装高度和俯仰角敏感 | 车体震动会影响地面和边界深度估计 |
| 安全冗余不足 | 深沟属于 catastrophic risk，不能依赖单一 RGB-D 传感器 |

深度相机不是完全不能用，但更适合作为调试工具或近距离辅助传感器，而不是主安全传感器。

### 2.2 LiDAR + RGB 的优势分工

| 模块 | LiDAR 负责 | RGB 相机负责 |
|---|---|---|
| 几何距离 | 障碍物距离、轮廓、空间占据 | 不擅长直接给稳定绝对距离 |
| 深沟 / 边界 | 检测右侧边缘、高差、不可通行区域 | 辅助识别铁轨、沟边纹理、标识物 |
| 障碍物 | 不需要知道是什么，只需知道空间被占据 | 判断人、工程车、工具、线缆、箱体等类别 |
| 重复隧道环境 | 通过局部点云几何维持短时稳定感知 | 纯视觉容易被重复纹理误导 |
| 安全控制 | 可直接进入 costmap / voxel map / keep-out zone | 进入 semantic costmap，调整不同对象风险权重 |

一句话总结：

> **LiDAR 解决“哪里不能走”，RGB 解决“那是什么、风险等级是什么”。**

---

## 3. Transformer 在该项目中到底能做什么？

项目负责人提到 Transformer 时，你可以理解为：

> Transformer 不是单独替代导航系统，而是可以作为 **多模态融合模型（multi-modal fusion model）**、**时序场景理解模型（temporal scene understanding model）**、**轨迹/策略生成模型（trajectory / policy model）** 使用。

在你的项目中，Transformer 最适合放在以下 5 个位置。

---

## 4. Transformer 应用位置 1：LiDAR + RGB 多模态融合

### 4.1 核心作用

LiDAR 给的是点云 / 扫描 / BEV 占据图；RGB 给的是图像语义。Transformer 可以通过 **attention / cross-attention** 把两种信息融合在一起。

推荐结构：

```text
LiDAR Point Cloud
  → BEV Grid / Voxel Feature / Point Tokens

RGB Image
  → Image Feature Tokens / Detection Tokens / Segmentation Tokens

LiDAR Tokens + RGB Tokens
  → Cross-Attention Fusion Transformer

Output
  → Semantic Occupancy Map
  → Traversability Map
  → Obstacle Risk Map
  → Trench Keep-out Mask
```

### 4.2 为什么 Transformer 适合做融合？

普通 early fusion / late fusion 通常比较机械：

- early fusion：一开始就把数据拼起来，可能导致噪声互相污染；
- late fusion：各自检测后再合并，可能错过跨模态信息；
- hard projection：把 LiDAR 点强行投影到 image，或把 image 投影到 LiDAR，容易受标定误差影响。

Transformer 的好处是：

- 可以让 LiDAR token 主动关注相关 RGB token；
- 可以让 RGB 语义补充 LiDAR 几何盲区；
- 可以做 soft association，而不是硬匹配；
- 对“未知障碍物但几何明显存在”的情况更保守；
- 对“RGB 看出是人，但 LiDAR 点云稀疏”的情况可以提高风险权重。

### 4.3 推荐融合方式

#### 方案 A：TransFuser-style 融合

TransFuser 的思想是使用 Transformer / self-attention 融合图像特征和 LiDAR BEV 特征。你的项目可以借鉴这个思路，但不要照搬自动驾驶端到端控制，而是输出 **semantic risk costmap**。

```text
RGB Feature Map + LiDAR BEV Feature Map
        ↓
Multi-scale Transformer Fusion
        ↓
Fused BEV Feature
        ↓
Risk Costmap / Local Waypoints / Traversability
```

适合你的原因：

- 隧道场景天然适合 BEV 表达；
- LiDAR 可提供稳定几何；
- RGB 可提供人员、工具、工程车语义；
- Transformer 负责把“几何 + 语义”融合成统一局部地图。

#### 方案 B：BEVFusion-style 融合

BEVFusion 的思想是把不同传感器的信息统一到 **bird’s-eye view（BEV，鸟瞰图）** 表示中。这对地面无人车非常有用，因为车辆局部规划本来就需要俯视图。

```text
Camera View Features
        ↓ view transform
BEV Image Features

LiDAR Point Cloud
        ↓ voxel / pillar encoder
BEV LiDAR Features

BEV Image Features + BEV LiDAR Features
        ↓ Fusion Network / Transformer
Unified BEV Representation
```

适合你的原因：

- 车辆控制需要的是前方和侧方的可通行区域；
- 右侧深沟可以被编码成 keep-out zone；
- 左侧杂物和前方障碍可以映射到同一个 BEV costmap；
- 后续 planner / RL / diffusion 都可以吃 BEV 表示。

#### 方案 C：TransFusion-style 目标级融合

TransFusion 的思想更偏向 3D object detection：先用 LiDAR 产生 object query / bounding box，再用 RGB 图像特征辅助修正。你的项目可以用它做人员、工程车辆、大型障碍物检测。

```text
LiDAR Object Queries
        ↓
Transformer Decoder attends to RGB Features
        ↓
3D Object Boxes + Semantic Labels + Risk Score
```

适合你的原因：

- 隧道里障碍物类别不固定；
- 有些障碍物 RGB 很明显但 LiDAR 点不密；
- 有些障碍物 LiDAR 明显但 RGB 语义不确定；
- 目标级融合可以提高风险判断稳定性。

---

## 5. Transformer 应用位置 2：RGB 语义识别与开放词汇检测

### 5.1 为什么需要 RGB 语义？

LiDAR 只能告诉你“这里有东西”，但不能稳定告诉你“这是什么”。在工程隧道中，语义非常重要：

| 物体 | 应对策略 |
|---|---|
| 人员 | 立即减速 / 停车，保持安全距离 |
| 工程车 | 高风险障碍，提前绕行或停车 |
| 工具 / 电机 / 箱体 | 静态障碍，局部绕行 |
| 线缆 / 绳索 | 高缠绕风险，最好避开 |
| 水坑 / 反光区域 | 低速谨慎通过或标记为未知风险 |
| 未知物体 | 采用保守 cost，而不是直接忽略 |

### 5.2 推荐模型方向

可以把 RGB 相机接入以下模型：

| 模型方向 | 用途 | 是否适合实时部署 |
|---|---|---|
| YOLO / RT-DETR | 人、车辆、工具检测 | 适合，速度快 |
| DINO / Grounding DINO | 开放词汇检测，例如“工程车”“电机”“线缆” | 适合离线标注 / 半实时辅助 |
| SAM / SAM2 | 分割物体边界、辅助建立 mask | 更适合离线标注或低频辅助 |
| SegFormer / Mask2Former | 语义分割道路、墙、铁轨、沟边 | 可训练轻量版部署 |
| Vision Transformer (ViT) | 图像特征提取 | 可作为 backbone |

### 5.3 推荐实践

不要一开始就把大模型直接放到车上实时跑。更稳的做法是：

```text
Step 1: 现场采集 RGB 视频 + LiDAR rosbag
Step 2: 用 Grounding DINO / SAM2 做半自动标注
Step 3: 人工修正关键帧标签
Step 4: 训练轻量化检测 / 分割模型
Step 5: 部署到车载端实时运行
Step 6: 输出 semantic layer 给 costmap
```

也就是说：

> **Grounding DINO / SAM2 更适合做数据标注和开放类别发现，车上实时运行可以用更轻量的 detector / segmentation model。**

---

## 6. Transformer 应用位置 3：时序场景理解与隧道记忆

### 6.1 为什么需要时序？

隧道场景不断循环、视觉纹理重复，单帧判断很容易不稳定。例如：

- 一帧里看不到深沟边界，不代表右侧安全；
- 一帧里人员被遮挡，不代表人员消失；
- 一帧里 LiDAR 点云有噪声，不代表障碍物不存在；
- 光照突然变化可能导致 RGB 检测短暂失败。

因此，需要让系统记住最近几秒的状态。

### 6.2 时序 Transformer 的输入

```text
Past N Frames:
  - BEV occupancy maps
  - semantic costmaps
  - distance-to-trench measurements
  - robot velocity and angular velocity
  - IMU pitch / roll / yaw rate
  - previous planned trajectories
  - previous safety filter interventions
```

### 6.3 时序 Transformer 的输出

```text
Output:
  - smoothed obstacle risk map
  - dynamic object motion estimate
  - short-term local scene memory
  - planner confidence score
  - hazard prediction for next 1–3 seconds
```

### 6.4 应用价值

| 问题 | 时序 Transformer 的帮助 |
|---|---|
| 人员短暂遮挡 | 通过历史帧保持风险，不立即解除停车 |
| LiDAR 点云抖动 | 平滑 costmap，减少假障碍跳变 |
| 深沟边界短暂不可见 | 使用历史边界估计保持 keep-out zone |
| 隧道重复纹理 | 不依赖单帧视觉特征做定位 |
| 动态障碍 | 预测人员或工程车运动趋势 |

---

## 7. Transformer 应用位置 4：RL / Decision Transformer / Trajectory Transformer 导航策略

### 7.1 你的项目可以怎么用 RL？

不建议使用纯 RL 直接从 raw image / point cloud 输出电机控制。更合理的是：

```text
Observation:
  - LiDAR BEV occupancy map
  - RGB semantic risk map
  - distance to trench
  - robot velocity
  - local goal direction
  - previous action history

Policy Output:
  - short-horizon waypoints
  - candidate trajectory
  - candidate velocity command (v, ω)

Safety Layer:
  - collision check
  - trench distance constraint
  - human stop rule
  - speed / acceleration limit
  - emergency stop
```

也就是说，RL / Transformer policy 只能输出 **candidate**，不能直接控制底盘。

### 7.2 Decision Transformer 怎么应用？

Decision Transformer 将强化学习看成序列建模问题。对于你的项目，可以这样设计：

```text
Input Sequence:
  return-to-go / desired safety-progress score
  + previous BEV states
  + previous semantic maps
  + previous actions
  + previous rewards / costs

Transformer:
  causal attention over trajectory history

Output:
  next waypoint / next local action / short trajectory token
```

更直观地说：

> 你可以先让人遥控无人车安全通过隧道，采集“专家轨迹”；然后用 Transformer 学习在什么场景下应该直行、减速、向左绕障、向右修正或停车。

### 7.3 Trajectory Transformer 怎么应用？

Trajectory Transformer 更像是把整段轨迹作为 token 序列建模，然后搜索一条高回报、低风险的动作序列。

```text
State Tokens:
  BEV map, semantic map, trench distance, velocity

Action Tokens:
  candidate v, ω or waypoint sequence

Reward / Cost Tokens:
  progress reward, collision cost, trench cost, human risk cost

Planning:
  beam search / sampling multiple trajectory candidates
  select trajectory with minimum risk and sufficient progress
```

适合用在：

- 左侧有杂物、右侧有深沟时的窄通道通过；
- 前方出现未知障碍时的绕行选择；
- 人员出现后的停车和重新启动；
- 局部 planner 失败时提供候选轨迹。

### 7.4 Diffusion Policy / Diffusion Trajectory Proposal 怎么结合？

Diffusion 可以生成多个候选轨迹，Transformer 可以作为条件编码器或时间序列建模器。

推荐路线：

```text
LiDAR BEV + RGB semantic risk map
        ↓
Transformer Encoder extracts scene context
        ↓
Diffusion model generates N candidate trajectories
        ↓
Safety evaluator ranks candidates
        ↓
MPC / local controller follows safest trajectory
```

输出不应是 PWM，而应是：

```text
Trajectory:
  [(x1, y1, θ1, v1), (x2, y2, θ2, v2), ..., (xT, yT, θT, vT)]
```

---

## 8. Transformer 应用位置 5：Safety-Aware MoE / 多策略选择

你的项目可以设计成 **Safety-Aware MoE（Mixture of Experts，安全感知专家混合）**。这和你的 Safety-Aware MoE Locomotion 方向也有联系。

### 8.1 专家策略设计

| Expert | 触发场景 | 输出 |
|---|---|---|
| Straight Expert | 前方空旷、左右安全 | 稳定向前 |
| Trench-Keeping Expert | 右侧深沟接近 | 向左修正、限速 |
| Obstacle-Avoidance Expert | 前方有静态障碍 | 绕行轨迹 |
| Human-Safety Expert | 检测到人员 | 减速 / 停车 |
| Narrow-Passage Expert | 左侧杂物 + 右侧深沟 | 低速、小曲率通过 |
| Recovery Expert | planner 卡住或局部死锁 | 停车、后退或请求接管 |

### 8.2 Transformer Gating Network

Transformer 可以作为 gating network，根据当前多模态场景选择专家：

```text
Input:
  - BEV occupancy tokens
  - RGB semantic tokens
  - trench distance token
  - velocity token
  - recent history tokens

Transformer Encoder:
  scene representation z_t

Gating Head:
  p(straight), p(trench), p(obstacle), p(human), p(narrow), p(recovery)

Expert Fusion:
  action_candidate = Σ p_i · expert_i(state)

Safety Filter:
  reject unsafe action
```

### 8.3 为什么 MoE 适合这个项目？

因为隧道导航不是单一场景，而是多个安全模式的切换：

- 正常前进；
- 靠近深沟；
- 前方杂物；
- 人员出现；
- 工程车占路；
- 窄通道；
- 传感器不确定；
- 需要人工接管。

MoE 可以让系统更可解释：当车辆做出动作时，可以解释为“当前由哪个专家主导”。

---

## 9. 推荐系统架构：LiDAR + RGB + Transformer + Safety Filter

### 9.1 完整架构

```text
Sensors
  ├── Front 3D LiDAR / 2D LiDAR
  ├── Front RGB Camera
  ├── Right-side LiDAR / ToF for trench distance
  ├── IMU
  └── Wheel Encoder

        ↓

Calibration & Synchronization
  ├── LiDAR-Camera Extrinsic Calibration
  ├── Time Synchronization
  ├── IMU-LiDAR Calibration
  └── ROS TF Tree

        ↓

Low-level Geometry Perception
  ├── Point Cloud Filtering
  ├── Ground / Non-ground Segmentation
  ├── Obstacle Clustering
  ├── Rail / Trench / Edge Detection
  └── Local Occupancy / Voxel Map

        ↓

RGB Semantic Perception
  ├── Human Detection
  ├── Engineering Vehicle Detection
  ├── Tool / Cable / Material Detection
  ├── Open-vocabulary Object Detection
  └── Semantic Segmentation

        ↓

Transformer Fusion Layer
  ├── LiDAR BEV Tokens
  ├── RGB Image / Object Tokens
  ├── Cross-attention Fusion
  ├── Temporal Memory Attention
  └── Unified BEV Scene Representation

        ↓

Semantic Risk Costmap
  ├── Occupancy Cost
  ├── Human Stop Zone
  ├── Vehicle / Tool High Cost
  ├── Unknown Object Conservative Cost
  ├── Trench Keep-out Zone
  └── Dynamic Obstacle Layer

        ↓

Planning Layer
  ├── Nav2 / Local Planner / MPC Baseline
  ├── Decision Transformer / Trajectory Transformer Candidate Policy
  ├── Diffusion Trajectory Proposal
  └── Safety-Aware MoE Expert Selection

        ↓

Safety Layer
  ├── Collision Check
  ├── Trench Distance Constraint
  ├── Human Stop Rule
  ├── Speed / Acceleration Limit
  ├── Watchdog
  ├── Sensor Health Check
  └── Emergency Stop / Manual Override

        ↓

Vehicle Control
  ├── v / ω Controller or Ackermann Controller
  ├── Motor Driver
  ├── Brake Interface
  └── Remote Takeover
```

### 9.2 推荐数据流

```text
/points_raw              → point cloud processing
/image_raw               → RGB semantic perception
/imu                     → state estimation
/wheel_odom              → odometry
/trench_distance         → side safety monitor
/local_costmap           → classical planner
/semantic_costmap        → risk-aware planner
/transformer_scene_state → learned fusion representation
/candidate_trajectories  → RL / diffusion / transformer planner
/safe_cmd_vel            → final command after safety filter
```

---

## 10. Transformer 输入输出设计

### 10.1 LiDAR token 设计

LiDAR 点云不建议直接把所有点作为 Transformer token，因为点太多、计算量太大。更推荐先转换成 BEV / voxel / pillar 表示。

```text
LiDAR Point Cloud
  → Crop Region of Interest
  → Downsample / Voxelize
  → BEV Grid
  → Patch / Cell Tokens
```

每个 BEV cell 可以包含：

| Channel | 含义 |
|---|---|
| occupancy | 是否被占据 |
| height_max | 最大高度 |
| height_min | 最小高度 |
| height_diff | 高度差，用于判断边缘 / 沟 / 障碍 |
| intensity | LiDAR 反射强度 |
| point_density | 点云密度 |
| distance_to_trench | 到右侧深沟的估计距离 |
| keepout_mask | 是否属于不可进入区域 |

### 10.2 RGB token 设计

RGB 可以有两种 token：

| Token 类型 | 来源 | 用途 |
|---|---|---|
| Image Patch Tokens | ViT / CNN backbone | 提供全局视觉上下文 |
| Object Tokens | detector 输出 | 人、车辆、工具、线缆等对象级信息 |
| Mask Tokens | segmentation 输出 | 可通行区域、墙体、铁轨、沟边 |
| Text Prompt Tokens | 可选 | “person / vehicle / tool / cable / trench”等开放词汇 |

### 10.3 Motion token 设计

由于车辆是移动平台，还需要运动状态：

| Token | 含义 |
|---|---|
| velocity token | 当前线速度 |
| yaw-rate token | 当前角速度 |
| steering token | 当前转向状态 |
| IMU token | roll / pitch / yaw rate |
| previous action token | 最近几步动作 |
| intervention token | 是否触发过 safety filter |

### 10.4 输出设计

不推荐输出底层电机控制，推荐输出中间层结果。

| 输出 | 推荐程度 | 说明 |
|---|---:|---|
| semantic risk costmap | 强推荐 | 最安全、最容易接传统 planner |
| traversability map | 强推荐 | 判断哪里可走 |
| local waypoints | 推荐 | 方便 MPC / Pure Pursuit 跟踪 |
| candidate trajectories | 推荐 | 适合 diffusion / trajectory transformer |
| expert weights | 推荐 | 适合 Safety-Aware MoE |
| raw motor PWM | 不推荐 | 不安全、不可解释、难验证 |

---

## 11. Costmap 与风险建模

### 11.1 基础 costmap

传统 costmap 可以包含：

```text
C_total(x, y) = C_occ + C_trench + C_human + C_vehicle + C_unknown + C_dynamic + C_smooth
```

其中：

| Cost | 含义 |
|---|---|
| C_occ | LiDAR 检测到的占据障碍 |
| C_trench | 靠近右侧深沟的惩罚 |
| C_human | 人员附近高惩罚 / 停车区 |
| C_vehicle | 工程车辆附近高惩罚 |
| C_unknown | 未知物体保守惩罚 |
| C_dynamic | 动态障碍预测惩罚 |
| C_smooth | 轨迹不平滑惩罚 |

### 11.2 深沟 keep-out zone

右侧深沟是最高风险点，建议定义硬约束：

```text
if distance_to_trench < d_stop:
    emergency_stop()
elif distance_to_trench < d_slow:
    limit_speed()
    bias_trajectory_left()
else:
    normal_navigation()
```

建议一开始采用非常保守的阈值，例如：

| 参数 | 含义 | 建议 |
|---|---|---|
| d_stop | 立即停车距离 | 现场测量后确定，初期取保守值 |
| d_slow | 减速距离 | 大于 d_stop |
| d_keepout | costmap 禁入区 | 覆盖深沟边缘与安全裕度 |

---

## 12. RL / Transformer 奖励函数与约束设计

如果后续确实要做 RL 或 Decision Transformer，可以设计 reward / cost：

```text
reward = w_progress * forward_progress
       - w_collision * collision_penalty
       - w_trench * trench_risk_penalty
       - w_human * human_risk_penalty
       - w_unknown * unknown_obstacle_penalty
       - w_jerk * action_smoothness_penalty
       - w_intervention * safety_filter_intervention_penalty
```

### 12.1 强制约束优先级

不要把所有安全问题都交给 reward。对于深沟和人员，必须用硬约束：

| 风险 | 处理方式 |
|---|---|
| 撞人 | hard stop，不是 reward penalty |
| 靠近深沟 | hard keep-out，不是 reward penalty |
| 传感器失效 | 立即降级 / 停车 |
| planner 置信度低 | 降速或请求接管 |
| 学习策略输出异常 | safety filter 拦截 |

---

## 13. 推荐开发路线

### Phase 0：传感器与安全边界确认

目标：先明确硬件能力和安全约束。

任务：

- 确认“雷达”具体类型：3D LiDAR / 2D LiDAR / 固态 LiDAR / mmWave；
- 确认 LiDAR 的 FOV、距离、帧率、点云接口；
- 确认 RGB 相机类型、帧率、低照度能力、是否可加补光；
- 测量车体尺寸、转弯半径、最大速度、刹车距离；
- 定义右侧深沟的 keep-out zone；
- 确认是否允许加右侧专用 LiDAR / ToF；
- 建立 emergency stop 和 remote takeover。

输出：

- Sensor Specification Table；
- Vehicle Kinematic Constraints；
- Safety Requirement Document；
- ODD（Operational Design Domain）。

---

### Phase 1：LiDAR + RGB 基础数据采集

目标：先采集真实隧道数据，不急着训练大模型。

任务：

- 遥控车在隧道中低速行驶；
- 同步记录 LiDAR、RGB、IMU、encoder、遥控指令；
- 采集空隧道、杂物、人员、工程车、线缆、水渍、反光、弱光场景；
- 用 rosbag 保存；
- 用 RViz / Foxglove 回放确认传感器覆盖范围；
- 标注关键帧的障碍物、深沟边界、人员位置、可通行区域。

输出：

- Calibrated rosbag dataset；
- Initial annotated dataset；
- Failure case library。

---

### Phase 2：传统安全 baseline

目标：不用 Transformer，也要先让车低速、安全地走起来。

任务：

- LiDAR 点云滤波；
- ground / non-ground 分割；
- obstacle clustering；
- local occupancy grid / costmap；
- 右侧 deep trench keep-out zone；
- RGB human detection → stop zone；
- Nav2 / local planner / MPC 低速导航；
- 人工遥控接管。

输出：

- 可视化 local costmap；
- 低速 autonomous demo；
- 人员停车 demo；
- 深沟距离保护 demo。

---

### Phase 3：Transformer 感知融合

目标：把 Transformer 用在最合理的位置：LiDAR-RGB fusion。

任务：

- 将 LiDAR 转为 BEV tokens；
- 将 RGB 转为 image/object/mask tokens；
- 训练或微调 cross-attention fusion 模型；
- 输出 semantic risk costmap；
- 对比以下 baseline：
  - LiDAR-only costmap；
  - RGB-only semantic detection；
  - late fusion；
  - Transformer fusion。

输出：

- Fusion model；
- Semantic risk costmap；
- Ablation study；
- 实时推理 FPS / latency 报告。

---

### Phase 4：RL / Decision Transformer / Diffusion 轨迹增强

目标：在已有安全 baseline 上增强复杂绕障能力。

任务：

- 用人工遥控轨迹做 imitation learning；
- 用仿真环境做 domain randomization；
- 用 Decision Transformer 学习历史状态到动作序列；
- 用 Trajectory Transformer / Diffusion 生成候选轨迹；
- 用 safety evaluator 筛掉不安全轨迹；
- 只执行通过安全过滤的轨迹。

输出：

- Candidate trajectory generator；
- RL / DT / Diffusion 对比实验；
- Safety filter intervention statistics；
- sim-to-real 测试记录。

---

### Phase 5：Safety-Aware MoE 与系统集成

目标：把不同场景切换做得更可解释。

任务：

- 定义 straight / trench / obstacle / human / narrow / recovery experts；
- 用 Transformer scene encoder 做 gating；
- 输出 expert weights；
- 可视化当前专家权重；
- 加入 fail-safe 规则：只要不确定，就降速或停车。

输出：

- Safety-Aware MoE policy；
- Expert activation logs；
- Failure mode report；
- 项目演示视频。

---

## 14. 验收指标建议

### 14.1 导航指标

| 指标 | 含义 |
|---|---|
| autonomous distance | 无接管自主行驶距离 |
| success rate | 完成测试段比例 |
| intervention count | 人工接管次数 |
| collision count | 碰撞次数，必须为 0 |
| near-miss count | 近距离危险事件次数 |
| average speed | 平均速度 |
| route smoothness | 轨迹平滑程度 |

### 14.2 安全指标

| 指标 | 含义 |
|---|---|
| minimum trench distance | 到右侧深沟的最小距离 |
| human stop distance | 检测到人员后的停车距离 |
| false negative rate for humans | 人员漏检率 |
| emergency stop latency | 急停延迟 |
| safety filter intervention rate | 安全层拦截学习策略的比例 |
| sensor dropout response | 传感器异常后的降级行为 |

### 14.3 模型指标

| 指标 | 含义 |
|---|---|
| mAP / recall | 人、车、工具检测能力 |
| segmentation IoU | 可通行区域 / 沟边 / 障碍分割能力 |
| BEV occupancy accuracy | BEV 占据预测准确率 |
| latency | 推理延迟 |
| FPS | 运行帧率 |
| robustness | 弱光、粉尘、反光下性能 |
| ablation gain | Transformer fusion 相比非 Transformer 的提升 |

---

## 15. 你可以向负责人这样解释 Transformer 的作用

### 中文版本

> 我理解 Transformer 不应该直接替代整个导航系统，而应该作为 LiDAR 和 RGB 的融合模块、时序场景理解模块以及轨迹候选生成模块。LiDAR 负责稳定的几何安全信息，例如障碍物距离、右侧深沟边界和局部 costmap；RGB 负责人员、工程车辆、工具和未知物体的语义识别。Transformer 可以通过 cross-attention 把 LiDAR 的几何信息和 RGB 的语义信息融合成 semantic risk costmap，再交给传统 planner、MPC 或 RL / diffusion 轨迹生成器。最终控制指令必须经过 safety filter，确保不会撞人、撞障碍或靠近深沟。

### English version

> I do not plan to use Transformer as a direct replacement for the entire navigation stack. Instead, Transformer can be used as a LiDAR-RGB fusion module, a temporal scene understanding module, and a trajectory proposal module. LiDAR provides reliable geometric safety information, such as obstacle distance, trench boundary and local costmap, while the RGB camera provides semantic understanding of humans, engineering vehicles, tools and unknown objects. A Transformer-based cross-attention module can fuse LiDAR geometry and RGB semantics into a semantic risk costmap, which is then used by the classical planner, MPC, RL policy or diffusion trajectory generator. The final command should always pass through a safety filter before being executed by the UGV.

---

## 16. 你需要向项目负责人确认的问题

### 16.1 关于雷达和相机

1. 这里的“雷达”具体是 **3D LiDAR、2D LiDAR、固态 LiDAR，还是毫米波雷达**？
2. LiDAR 的水平 FOV、垂直 FOV、最大距离、最小距离、帧率是多少？
3. 是否允许加一个右侧专用 LiDAR / ToF 来监控深沟？
4. RGB 相机是否是低照度工业相机？是否允许加补光？
5. 相机与 LiDAR 是否能刚性固定，方便外参标定？
6. 车上计算平台是什么？Jetson、x86 + GPU，还是普通工控机？

### 16.2 关于 Transformer

1. 负责人希望 Transformer 用在 **感知融合**，还是 **导航策略 / RL policy**？
2. 是否有足够数据训练 Transformer，还是主要使用预训练模型？
3. 是否要求实时运行？目标 FPS / latency 是多少？
4. 是否需要可解释输出，例如 attention map、expert weights、risk map？
5. 项目更看重工程安全落地，还是算法创新 demo？

### 16.3 关于安全验收

1. 右侧深沟最小安全距离是多少？
2. 人员出现时必须停车，还是允许低速绕行？
3. 允许的最大测试速度是多少？
4. 是否必须支持远程接管和物理急停？
5. 发生传感器故障时，系统是停车、低速前进，还是人工接管？

---

## 17. 推荐技术栈

| 层级 | 推荐工具 / 框架 |
|---|---|
| 系统中间件 | ROS 2 |
| 可视化 | RViz2 / Foxglove |
| 传统导航 | Nav2 / local costmap / DWB / Regulated Pure Pursuit / MPC |
| 点云处理 | PCL / Open3D / ROS point cloud tools |
| 感知模型 | YOLO / RT-DETR / Grounding DINO / SAM2 / SegFormer |
| LiDAR-RGB 融合 | TransFuser-style / BEVFusion-style / custom cross-attention fusion |
| 学习策略 | PyTorch / Isaac Sim / Isaac Lab / Gymnasium |
| 轨迹生成 | Decision Transformer / Trajectory Transformer / Diffusion Policy |
| 部署加速 | ONNX / TensorRT |
| 数据记录 | rosbag2 |
| 标注 | CVAT / Label Studio / SAM-assisted annotation |
| 安全 | costmap keep-out zone / watchdog / safety filter / emergency stop |

---

## 18. 不建议采用的路线

### 18.1 不建议：RGB-D + End-to-End RL + Motor PWM

```text
RGB-D Image
  → End-to-End RL Network
  → Motor PWM
```

问题：

- 深度相机在隧道工程环境中不够稳；
- 端到端 RL 难以解释为什么转向；
- 数据需求过大；
- 安全无法验证；
- 右侧深沟风险太高；
- 人员安全不应只靠 reward 学出来。

### 18.2 不建议：大模型直接闭环控制

```text
RGB Image
  → VLM / VLA
  → Direct Driving Command
```

问题：

- 大模型推理延迟较高；
- 可能产生不稳定或不可验证的输出；
- 难以满足实时安全控制；
- 不适合直接处理紧急制动；
- 更适合做高层解释、任务理解和离线分析。

---

## 19. 参考资料

以下资料用于支撑技术路线选择，实际工程实现仍应以公司硬件、现场数据和安全要求为准。

| 方向 | 资料 |
|---|---|
| Nav2 Obstacle Layer | https://docs.nav2.org/configuration/packages/costmap-plugins/obstacle.html |
| Nav2 Voxel Layer | https://docs.nav2.org/configuration/packages/costmap-plugins/voxel.html |
| TransFuser：Transformer-based LiDAR-RGB Fusion | https://arxiv.org/abs/2205.15997 |
| BEVFusion：统一 BEV 多传感器融合 | https://arxiv.org/abs/2205.13542 |
| TransFusion：Transformer LiDAR-Camera 3D Detection | https://arxiv.org/abs/2203.11496 |
| Grounding DINO：开放词汇目标检测 | https://arxiv.org/abs/2303.05499 |
| SAM2：图像/视频分割基础模型 | https://arxiv.org/abs/2408.00714 |
| Decision Transformer：RL as Sequence Modeling | https://arxiv.org/abs/2106.01345 |
| Trajectory Transformer：轨迹序列建模 | https://arxiv.org/abs/2106.02039 |
| Diffusion Policy：机器人动作扩散策略 | https://arxiv.org/abs/2303.04137 |
| OpenVLA：开源视觉-语言-动作模型 | https://arxiv.org/abs/2406.09246 |
| FAST-LIO2：LiDAR-Inertial Odometry | https://arxiv.org/abs/2107.06829 |
| Learning Control Barrier Functions survey | https://arxiv.org/abs/2404.16879 |

---

## 20. 最终推荐路线

最终建议把项目表述为：

> **A LiDAR-RGB fusion based safety-aware navigation system for tunnel UGVs, enhanced by Transformer-based multi-modal perception and learning-based trajectory proposal.**

中文表述为：

> **一种面向隧道工程无人车的激光雷达-视觉融合安全感知导航系统，并通过 Transformer 多模态融合与学习型轨迹生成增强复杂障碍场景下的鲁棒性。**

### 20.1 核心技术路线

```text
3D LiDAR / 2D LiDAR
  → Geometry / Occupancy / Trench Boundary

RGB Camera
  → Human / Vehicle / Tool / Unknown Object Semantics

Transformer Fusion
  → Unified BEV Scene Representation

Semantic Risk Costmap
  → Safe Local Planning

RL / Decision Transformer / Diffusion
  → Candidate Trajectories

Safety Filter
  → Reject Unsafe Commands

UGV Controller
  → Execute Safe Motion
```

### 20.2 最重要的工程原则

- 先做 **LiDAR costmap + RGB human detection + safety stop** baseline；
- 再做 **LiDAR-RGB semantic costmap**；
- 再做 **Transformer fusion**；
- 最后才做 **RL / Decision Transformer / Diffusion trajectory proposal**；
- 任何学习模块都只能输出候选，不允许绕过 safety filter；
- 对人员和深沟必须使用硬规则，而不是只用 reward；
- 现场测试必须低速、可接管、有急停。

一句话总结：

> **Transformer 在这个项目中最有价值的位置不是直接开车，而是把 LiDAR 的几何安全信息和 RGB 的语义信息融合成可解释、可验证、可规划的风险地图，再辅助 RL / Diffusion 生成更好的绕障候选轨迹。**
