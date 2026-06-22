# 隧道工程无人车 RL / RL辅助导航项目评估报告

**项目场景（Project Context）**：隧道内工程器械巡检 / 运输辅助无人车
**核心任务（Task）**：在没有巡线标记、场景高度重复、右侧存在铁轨与深沟、左侧存在杂物与工程器械的隧道侧边道路中，实现安全前进、边界识别、动态/未知障碍物避让。
**建议定位（Recommended Framing）**：**Safety-Aware Learning-Augmented Navigation System**，即“传统安全导航骨架 + 学习模块辅助感知/局部规划/RL策略”的混合系统，而不是纯 RL / 纯大模型端到端控制。

---

## 0. 一句话结论

这个项目**有较高工程价值，也具备可落地性**，但不建议直接做成“纯 RL 自动驾驶”或“纯 VLA 大模型控制小车”。更合理的路线是：

> **用 LiDAR / Depth / Camera / IMU / Wheel Encoder 构建局部环境表示（Local Map / Costmap），用传统局部规划器保证基础安全，再让 RL / Diffusion / VLM 作为增强模块：识别复杂障碍、生成绕行轨迹候选、处理未知物体与高层语义决策。**

如果公司当前硬件只有普通相机和底盘，而没有深度、LiDAR、右侧边缘检测、急停和远程接管，那么这个项目的风险会明显升高。尤其是右侧有深沟，这不是普通避障问题，而是**安全约束导航（Safety-Constrained Navigation）**问题。

---

## 1. 图片内容理解

### 图 1：白板草图

第一张图是白板上的手绘方案图：

- 左侧像是一个侧视或俯视的无人车 / 隧道边界示意。
- 右侧是一个矩形区域，下面写着 **CAM**，应该是在讨论相机视野、障碍物分布或图像中可见区域。
- 图中画了斜线、箭头、边界和类似障碍物的标记，整体表达的是：
  **需要通过摄像头或传感器理解隧道边界、可通行区域以及障碍物位置。**

### 图 2：实际无人车平台

第二张图是一台大型四轮无人车平台：

- 车体为黑色金属底盘，轮胎很大，适合粗糙地面或工程场地。
- 可以看到前后悬挂、外露电控仓、线缆和疑似电机/驱动模块。
- 车顶目前比较平整，有安装相机、LiDAR、天线、计算盒的空间。
- 车体惯量和尺寸应该不小，所以不能只考虑“能不能绕过去”，还必须考虑：
  - 刹车距离（Stopping Distance）
  - 转弯半径（Turning Radius）
  - 速度限制（Speed Limit）
  - 右侧深沟的安全距离（Safety Margin）
  - 人员出现时的紧急停止（Emergency Stop）

---

## 2. 场景本质：这不是“巡线小车”，而是“受约束的非结构化局部导航”

你说它“类似巡线小车”，但隧道内没有线，所以它的实际任务更接近：

> **在半结构化工程环境（Semi-structured Construction Tunnel）中的局部自主导航（Local Autonomous Navigation）**

它和普通室内导航 / 仓库 AGV / 自动驾驶都有相似点，但也有自己的特殊困难。

### 2.1 环境特点

| 特点 | 对算法的影响 |
|---|---|
| 隧道环境高度重复 | 传统视觉 SLAM 容易出现误匹配；仅靠图像特征定位不稳定 |
| 没有车道线 / 巡线标记 | 不能用 line following；必须识别可通行区域与边界 |
| 左侧是杂物、工具、电机、工程设备 | 障碍物形状不规则，类别开放，不能只训练固定类别 |
| 右侧是铁轨和很宽的深沟 | 属于高风险区域，需要硬安全约束，不能只靠 reward 学出来 |
| 可能出现人员和工程车辆 | 需要动态障碍物检测与保守避让策略 |
| 光照、粉尘、反光、水渍可能变化 | 纯视觉鲁棒性不足，应多传感器融合 |
| 场景循环、重复 | 全局地图导航未必可靠，局部感知和局部规划更重要 |

---

## 3. 可行性评估

### 3.1 总体可行性：中高，但取决于系统定义

| 项目路线 | 可行性 | 风险 | 结论 |
|---|---:|---:|---|
| 纯相机 + 端到端 RL | 低到中 | 高 | 不建议作为主路线 |
| 纯视觉大模型 / VLA 控制车辆 | 低到中 | 高 | 可做研究 demo，不适合直接现场闭环 |
| LiDAR / Depth + Costmap + Nav2 / MPC | 高 | 中 | 推荐作为安全 baseline |
| 语义感知 + 几何 costmap | 高 | 中 | 推荐作为主路线 |
| RL 作为局部策略辅助 | 中到高 | 中 | 推荐在仿真充分训练后引入 |
| Diffusion 生成绕行轨迹候选 | 中 | 中 | 可作为 trajectory proposal，不能直接绕过安全层 |
| VLM / LLM 做高层语义判断 | 中 | 中 | 适合作为 supervisor，不适合低层实时控制 |

### 3.2 项目最大技术风险

1. **右侧深沟风险不是普通 obstacle avoidance**
   深沟属于不可恢复危险区域（Catastrophic Region）。如果车辆掉入排水槽，后果比撞到纸箱严重得多，因此不能只通过 RL reward 约束。

2. **未知障碍物很难靠分类解决**
   工具、线缆、电机、工程车、施工材料都可能没有见过。系统应该优先识别：
   - 哪里能走（Traversable Area）
   - 哪里不能走（Non-traversable Area）
   - 哪些区域风险高（High-risk Zone）
   而不是只问“这是什么物体”。

3. **隧道重复结构会削弱全局定位**
   如果依赖特征点或视觉地图，很可能因为墙体、轨道、管线重复而定位漂移。更建议使用局部 LiDAR-inertial odometry / local mapping，并结合里程计、IMU 和必要的人工标记。

4. **工程现场安全要求高于论文 demo**
   大模型、RL 和 Diffusion 都可以做增强，但最终控制命令必须通过安全过滤器（Safety Filter）。

---

## 4. 推荐总体架构

建议采用 **“感知层 + 局部地图层 + 规划层 + 安全层 + 控制层”** 的架构。

```text
Sensors
  ├── RGB / Low-light Camera
  ├── Depth Camera or 3D LiDAR
  ├── 2D LiDAR / Side ToF / Ultrasonic
  ├── IMU
  └── Wheel Encoder

        ↓

Perception Layer
  ├── Free-space / Traversability Segmentation
  ├── Trench / Rail / Boundary Detection
  ├── Obstacle Detection
  ├── Human / Vehicle Detection
  └── Open-vocabulary Recognition

        ↓

Local Map / Risk Map
  ├── Occupancy Grid
  ├── Semantic Costmap
  ├── Keep-out Zone
  ├── Dynamic Obstacle Layer
  └── Unknown Area Risk Layer

        ↓

Planner
  ├── Classical Local Planner: DWB / TEB / MPPI / MPC
  ├── RL Local Policy: velocity / waypoint proposal
  ├── Diffusion Trajectory Proposal
  └── VLM High-level Supervisor

        ↓

Safety Filter
  ├── Collision Check
  ├── Trench Distance Constraint
  ├── Human Safety Constraint
  ├── Speed Limit
  ├── Emergency Stop
  └── Manual Override

        ↓

Low-level Control
  ├── Steering / Differential Drive / Ackermann Control
  ├── Motor Driver
  └── Watchdog
```

---

## 5. 传感器方案建议

### 5.1 不建议只用普通 RGB Camera

普通相机可以提供丰富语义信息，但在隧道工程场景中会遇到：

- 光照不足
- 强反光
- 粉尘和水雾
- 重复纹理
- 无法直接获得障碍物距离
- 难以可靠估计深沟边缘高度差

因此，**相机适合做语义感知，不适合单独作为安全导航主传感器。**

### 5.2 推荐最低传感器组合

| 传感器 | 作用 | 推荐程度 |
|---|---|---:|
| 3D LiDAR 或深度相机 | 构建局部几何、检测障碍物和沟边 | 极高 |
| 前向 RGB / Low-light Camera | 识别人、车辆、工具、语义障碍 | 高 |
| 右侧 ToF / 超声 / 短距 LiDAR | 检测深沟边缘与右侧安全距离 | 极高 |
| IMU | 姿态、震动、坡度估计 | 高 |
| Wheel Encoder | 里程计、速度闭环 | 高 |
| 急停按钮 / 遥控接管 | 现场安全 | 必须 |
| Bumper / Contact Sensor | 低速碰撞保护 | 推荐 |

### 5.3 传感器安装建议

1. **前向感知传感器**
   安装在车头上方，尽量避开车体遮挡，视野覆盖 0–10 m 前方。

2. **右侧安全传感器**
   必须专门检测右侧铁轨 / 深沟 / 边缘，不要完全依赖前向相机。

3. **相机与 LiDAR 刚性固定**
   隧道车振动较大，传感器支架需要刚性好，否则标定会漂移。

4. **电控仓保护**
   目前图中电控仓有一定外露，建议做防尘、防水、线缆应力释放和保险保护。

---

## 6. 感知系统：先识别“能不能走”，再识别“是什么”

### 6.1 几何感知（Geometry Perception）

这是主安全层，负责回答：

- 地面在哪里？
- 深沟边缘在哪里？
- 右侧铁轨 / 沟是否太近？
- 前方是否有实体障碍？
- 左侧杂物是否侵入可通行区域？

可用方法：

- Point Cloud Ground Segmentation
- Plane Fitting / RANSAC
- Height Map
- Occupancy Grid
- BEV Grid Map
- Local Costmap

### 6.2 语义感知（Semantic Perception）

负责回答：

- 前方是不是人？
- 是不是工程车辆？
- 是不是电机、工具箱、线缆、管道？
- 障碍物是否可能移动？
- 是否需要减速、停车、绕行？

可用模型：

- YOLO / RT-DETR：固定类别实时检测
- Grounding DINO：开放词汇检测（Open-vocabulary Detection）
- SAM / SAM2：分割物体轮廓（Segmentation）
- Depth / LiDAR + Semantic Fusion：把语义投影到 costmap

### 6.3 对未知障碍物的原则

对于没见过的障碍物，不要强行分类，而是用保守策略：

> **只要占据了可通行空间，并且高度/形状/距离不满足安全通过条件，就先当作障碍物。**

这比“识别出它是什么”更可靠。

---

## 7. 地图与定位：不要过度依赖全局地图

### 7.1 为什么传统全局 SLAM 在隧道内会困难

隧道内常见问题：

- 墙体、轨道、管线重复
- 可区分特征少
- 长走廊导致观测退化
- 回环检测容易误判
- 光照变化导致视觉特征不稳定

所以，不能把项目定义成“做一张大地图，然后按地图走”。

### 7.2 更推荐的定位方式

| 定位方式 | 用途 | 建议 |
|---|---|---|
| Wheel Odometry | 短时速度和位移 | 必须 |
| IMU | 姿态与短时运动估计 | 必须 |
| LiDAR-Inertial Odometry | 中短距离稳定定位 | 推荐 |
| Visual-Inertial Odometry | 辅助 | 可选 |
| Local Mapping | 只维护附近环境 | 推荐 |
| UWB / AprilTag / 人工标记 | 长距离纠偏 | 可选 |
| Full Global SLAM | 仅作为辅助 | 不建议作为唯一依赖 |

核心思想：

> **车辆只需要知道“现在附近哪里能走、右侧沟在哪里、前方是否有人或障碍、往前的安全通道在哪里”。**

---

## 8. 规划系统：传统规划为主，RL / Diffusion 为辅

### 8.1 基础规划层：Costmap + Local Planner

这是最应该先做出来的 baseline。

基本逻辑：

1. 传感器生成局部占据地图（Occupancy Grid）
2. 深沟、铁轨、墙、障碍物写入 costmap
3. 给右侧深沟设置 keep-out zone
4. 给人员设置动态障碍和停车区域
5. 局部规划器生成安全速度命令

可选规划器：

- DWB（Dynamic Window Approach）
- TEB（Timed Elastic Band）
- MPPI（Model Predictive Path Integral）
- MPC（Model Predictive Control）
- Regulated Pure Pursuit（适合路径跟踪，但复杂避障时要结合 costmap）

### 8.2 RL 适合做什么

RL 不应该直接替代所有导航模块，但适合做：

- 局部避障策略（Local Avoidance Policy）
- 狭窄通道通过策略
- 在杂物区选择更合理的绕行方向
- 根据风险图输出速度 / 转向建议
- 在传统 planner 失败时提供候选动作

RL policy 的输入可以是：

- BEV occupancy map
- Semantic costmap
- 右侧沟距离
- 目标方向
- 当前速度
- 最近几帧历史状态
- 动态障碍物位置和速度

RL policy 的输出建议是：

- 期望线速度 `v`
- 期望角速度 `ω`
- 或未来 2–5 秒 waypoint / trajectory proposal

不建议直接输出底层电机 PWM，因为这会增加训练难度和安全风险。

### 8.3 RL reward 设计示例

```text
Reward =
  + w_progress * forward_progress
  - w_collision * collision
  - w_trench * risk_near_trench
  - w_rail * risk_near_rail
  - w_human * unsafe_distance_to_human
  - w_obstacle * near_obstacle_penalty
  - w_unknown * unknown_area_penalty
  - w_jerk * control_jerk
  - w_reverse * unnecessary_reverse
```

终止条件（Episode Termination）：

- 撞到障碍物
- 距离深沟小于安全阈值
- 与人员距离低于安全阈值
- 车辆姿态异常
- 卡住超过一定时间
- 偏离可通行区域

### 8.4 Diffusion 适合做什么

Diffusion Policy / Diffusion Planner 比较适合做：

- 根据当前 BEV / RGB-D / costmap 生成多个绕行轨迹候选
- 处理多模态行为，例如左绕 / 右绕 / 停车等待
- 在复杂障碍物形态下生成更平滑轨迹

但必须注意：

> **Diffusion 生成的是“候选轨迹”，不是最终命令。最终轨迹必须经过碰撞检测、深沟安全距离检查、速度限制和急停规则。**

推荐流程：

```text
BEV + Semantic Map + Goal Direction
        ↓
Diffusion Trajectory Generator
        ↓
Generate N Candidate Trajectories
        ↓
Safety Filter + Cost Evaluation
        ↓
Select Best Safe Trajectory
        ↓
MPC / Low-level Controller
```

### 8.5 VLM / VLA / 大模型适合做什么

大模型适合做高层语义理解，例如：

- “前方有人，停车等待”
- “左侧有散落工具，降低速度并偏右一点”
- “右侧是深沟，禁止靠近”
- “前方工程车占道，等待或请求人工接管”
- “当前道路被完全堵塞，停止并报警”

但不建议让 VLM / VLA 直接以 10–50 Hz 频率输出底盘控制命令。原因：

- 延迟高
- 输出不稳定
- 安全约束难以形式化保证
- 对深沟这种高风险区域不能只靠语言模型判断

更好的方式是：

> **VLM 做 scene interpreter / supervisor，底层仍由 costmap + planner + safety filter 执行。**

---

## 9. 推荐的 Safety-Aware MoE 结构

如果你想把这个项目和 **MoE / Safety-Aware Locomotion** 结合，可以把“不同专家策略”设计成不同场景下的局部导航专家。

### 9.1 专家策略（Experts）

| Expert | 负责场景 | 输出 |
|---|---|---|
| Corridor Following Expert | 正常沿隧道前进 | 稳定速度与方向 |
| Trench Avoidance Expert | 右侧深沟接近 | 强制偏离深沟、减速 |
| Obstacle Bypass Expert | 静态杂物绕行 | 左/右绕行轨迹 |
| Human Safety Expert | 人员出现 | 减速、停车、等待 |
| Vehicle Interaction Expert | 工程车/大型障碍物 | 停车、绕行或请求接管 |
| Recovery Expert | 卡住、无路、定位异常 | 停止、倒退、重新规划 |
| Low-visibility Expert | 粉尘/暗光/传感器退化 | 降速、提高保守性 |

### 9.2 Gating Network 输入

Gating network 可以根据以下信息选择 expert：

- 距离右侧深沟的最小距离
- 前方障碍密度
- 人员检测置信度
- 可通行区域宽度
- 定位置信度
- 传感器健康状态
- 车辆速度与制动距离
- 当前 planner 是否失败

### 9.3 Safety Layer 必须覆盖所有 Expert

即使 gating 选错 expert，也不能允许车辆执行危险动作。

```text
Selected Expert Action
        ↓
Safety Filter
        ↓
Safe Action or Emergency Stop
```

---

## 10. 推荐开发路线

### Phase 0：需求和危险源确认

目标：把项目从“想做 RL 导航”变成可测试工程任务。

需要确认：

- 隧道宽度
- 可通行道路宽度
- 右侧深沟宽度、深度、边缘形态
- 允许最高速度
- 车辆尺寸、重量、最小转弯半径
- 最大刹车距离
- 是否允许加 LiDAR / 深度相机 / ToF
- 是否有人机混行
- 是否需要夜间 / 粉尘 / 水雾条件运行
- 是否能远程接管
- 现场是否允许贴 AprilTag / 反光标记 / UWB beacon

输出物：

- ODD（Operational Design Domain）
- Hazard List
- Safety Requirement
- Test Scenario List

---

### Phase 1：数据采集与基础遥控系统

目标：先让车能稳定记录数据。

任务：

- 搭建 ROS 2 数据记录系统
- 同步记录 RGB、Depth / LiDAR、IMU、Encoder、遥控输入
- 完成 camera-LiDAR / camera-depth 标定
- 遥控车在隧道内跑多轮数据
- 收集正常、障碍、人员、工程车、光照变化等场景

输出物：

- rosbag 数据集
- 传感器标定文件
- 初始场景标签
- 风险场景样本库

---

### Phase 2：传统安全导航 baseline

目标：不使用 RL，先实现保守可用的局部导航。

任务：

- 构建 Occupancy Grid / Costmap
- 检测右侧深沟和铁轨边界
- 设置 keep-out zone
- 前方障碍物避让
- 人员检测触发停车
- 实现紧急停止与远程接管
- 在低速下完成隧道侧边前进 demo

输出物：

- Baseline navigation stack
- Safety costmap visualization
- Field test video
- 失败案例报告

---

### Phase 3：语义感知增强

目标：让系统不仅知道“有东西”，还知道“可能是什么”。

任务：

- 训练 / 微调人、工程车、工具、线缆、堆料等检测模型
- 使用 SAM / Grounding DINO 辅助标注和开放词汇检测
- 将检测结果投影到 BEV / Costmap
- 对人员和车辆设置更高风险代价
- 对未知物体采用保守占据策略

输出物：

- Semantic costmap
- Obstacle type report
- Unknown obstacle policy

---

### Phase 4：仿真环境与 RL 训练

目标：在可控环境中训练局部策略。

仿真建议：

- Gazebo / Isaac Sim / Isaac Lab
- 建立隧道、铁轨、深沟、杂物、人员、工程车模型
- 随机化障碍物位置和形状
- 随机化光照、纹理、粉尘、传感器噪声
- 随机化轮胎摩擦、质量、控制延迟

训练策略：

- PPO / SAC：输出速度或 waypoint
- Imitation Learning：从遥控数据学习
- Offline RL：利用已采集数据
- Curriculum Learning：从空隧道逐步增加复杂度
- Domain Randomization：提高 sim-to-real 鲁棒性

输出物：

- RL local policy
- 仿真 success rate
- sim-to-real gap 分析
- 与 baseline 对比结果

---

### Phase 5：Diffusion / VLM 增强

目标：做出更先进的研究型亮点。

任务：

- Diffusion 生成多条绕行候选轨迹
- Safety Filter 过滤危险轨迹
- VLM 生成场景描述和高层决策建议
- MoE gating 根据风险状态选择专家策略
- 与 baseline / RL-only / diffusion-only 对比

输出物：

- Learning-augmented planner demo
- Ablation study
- 场景解释日志
- 最终报告和演示视频

---

## 11. 关键评价指标

### 11.1 安全指标

| 指标 | 说明 |
|---|---|
| Collision Rate | 撞障碍次数 / 测试距离 |
| Near-miss Rate | 接近障碍但未撞上的危险次数 |
| Minimum Distance to Trench | 与右侧深沟最小距离 |
| Human Stop Distance | 发现人员后的停车距离 |
| Emergency Stop Success Rate | 急停成功率 |
| Manual Intervention Rate | 人工接管次数 / km |
| Unsafe Command Rate | 被 safety filter 拦截的危险动作比例 |

### 11.2 导航性能指标

| 指标 | 说明 |
|---|---|
| Success Rate | 完成指定距离的比例 |
| Average Speed | 平均速度 |
| Progress Efficiency | 实际路径长度 / 理想前进距离 |
| Stuck Rate | 卡住或无法规划的次数 |
| Recovery Success Rate | 重新规划成功率 |
| Smoothness | 加速度、角速度变化是否平滑 |

### 11.3 感知指标

| 指标 | 说明 |
|---|---|
| Traversable Area IoU | 可通行区域分割准确率 |
| Trench Edge Error | 深沟边缘估计误差 |
| Human Detection Recall | 人员检测召回率 |
| Unknown Obstacle Detection Rate | 未知障碍物识别为不可通行的比例 |
| False Stop Rate | 误停车率 |

---

## 12. 现场测试场景设计

建议测试集不要只包括“顺利运行”，而要覆盖危险边界。

### 12.1 基础场景

1. 空隧道直线前进
2. 左侧有少量杂物
3. 左侧杂物侵入通行区域
4. 右侧深沟边缘不明显
5. 路面有水渍或反光
6. 光照变暗
7. 车速不同：低速 / 中速

### 12.2 障碍物场景

1. 小工具散落
2. 电机 / 箱体阻挡
3. 电缆横跨地面
4. 大型工程车局部占道
5. 不规则未知物体
6. 前方完全堵塞

### 12.3 人员场景

1. 人站在前方
2. 人从左侧突然进入
3. 人蹲下或半遮挡
4. 人与工具混在一起
5. 人在车辆旁边近距离作业

### 12.4 传感器退化场景

1. 相机曝光过暗
2. 强反光
3. LiDAR 局部遮挡
4. 深度相机失效
5. 轮胎打滑
6. IMU / encoder 短时异常

---

## 13. 主要风险与缓解措施

| 风险 | 严重性 | 原因 | 缓解措施 |
|---|---:|---|---|
| 掉入右侧深沟 | 极高 | 边界识别错误、转向过大、打滑 | 右侧专用距离传感器、keep-out zone、低速限制、CBF/MPC safety filter |
| 撞到人员 | 极高 | 视觉漏检、遮挡、延迟 | 人员检测 + LiDAR 动态障碍 + 安全停车距离 + 急停 |
| RL 输出危险动作 | 高 | reward 不完整、sim-to-real gap | RL 只输出候选动作，必须经过 safety filter |
| 隧道定位漂移 | 中高 | 重复纹理、长走廊 | LiDAR-inertial odometry + local map + 人工标记纠偏 |
| 未知障碍物漏检 | 高 | 训练集中没有该物体 | 几何占据优先，未知物体保守处理 |
| 传感器受粉尘/水雾影响 | 中高 | 工程环境恶劣 | 多传感器冗余、传感器健康监控 |
| 车体惯性导致刹不住 | 高 | 车辆重、速度高 | 速度上限、制动距离模型、提前减速 |
| 算法延迟过大 | 中高 | 大模型 / diffusion 推理慢 | 低层控制不用大模型；边缘部署 TensorRT；异步 supervisor |
| 电控仓进尘进水 | 中 | 工地环境 | 防护外壳、线缆固定、保险和急停 |

---

## 14. 最小可行系统（MVP）定义

### 14.1 MVP 不应包含的内容

第一阶段不要追求：

- 完整端到端 VLA
- 复杂多模态大模型闭环控制
- 全自动高速度运行
- 完全未知环境全局导航
- 无安全员现场测试

### 14.2 MVP 应该完成的内容

建议 MVP 定义为：

> 在低速、有人监管、固定测试隧道侧边道路中，无人车能够利用 LiDAR / Depth / Camera 识别可通行区域、右侧危险边界、前方障碍物和人员，并在不掉入深沟、不撞障碍、不靠近人员的前提下，连续前进一定距离。

### 14.3 建议验收标准

| 类别 | 建议标准 |
|---|---|
| 运行距离 | 连续完成 50–100 m 低速自主运行 |
| 安全距离 | 与深沟保持现场定义的最小安全距离，例如 ≥ 0.5–1.0 m |
| 撞障碍 | 20 次测试中 0 次实碰撞 |
| 人员检测 | 人员出现在前方时必须停车 |
| 人工接管 | 每 100 m 接管次数逐步降低 |
| 速度 | 初期建议 ≤ 0.3–0.5 m/s |
| 日志 | 所有传感器和控制命令可回放分析 |

---

## 15. 推荐技术栈

### 15.1 软件框架

| 模块 | 推荐 |
|---|---|
| Middleware | ROS 2 |
| Navigation Baseline | Nav2 |
| Local Map | Costmap / BEV Grid |
| Simulation | Isaac Sim / Isaac Lab / Gazebo |
| RL Training | Stable-Baselines3 / RLlib / Isaac Lab RL stack |
| Perception | PyTorch + TensorRT |
| Data Logging | rosbag2 |
| Visualization | RViz2 / Foxglove |
| Control | MPC / PID / low-level motor controller |

### 15.2 算法组件

| 模块 | 可选方案 |
|---|---|
| LiDAR-Inertial Odometry | FAST-LIO2 / LIO-SAM / LVI-SAM |
| Local Planner | DWB / TEB / MPPI / MPC |
| Segmentation | SAM / SAM2 / custom semantic segmentation |
| Open-vocabulary Detection | Grounding DINO |
| Fixed-class Detection | YOLO / RT-DETR |
| RL Policy | PPO / SAC / Imitation Learning |
| Trajectory Generation | Diffusion Policy / Diffusion Planner |
| Safety | CBF / MPC constraints / Rule-based safety filter |

---

## 16. 你作为实习生可以优先做的事情

如果你现在刚接手项目，建议不要一开始就训练 RL。更好的顺序是：

### 第一优先级：把问题定义清楚

你可以先输出一份：

- 任务定义
- 场景约束
- 危险源列表
- 可用传感器清单
- 车辆运动学约束
- 评价指标
- baseline 路线

这会让公司觉得你不是只会堆模型，而是能做系统工程。

### 第二优先级：做数据采集与 baseline

你可以推动：

- 在车上安装 camera / depth / LiDAR
- 录制 rosbag
- 做一个可视化 costmap
- 标出右侧深沟和可通行区域
- 做一个低速人工监管的自主前进 demo

### 第三优先级：把 RL / Diffusion 作为研究亮点

当 baseline 能跑后，再加入：

- RL local planner
- Diffusion trajectory proposal
- Semantic costmap
- MoE gating
- Safety filter ablation

---

## 17. 推荐最终汇报结构

你可以向公司这样汇报：

1. **Problem Definition**
   这不是巡线，而是隧道内安全约束局部导航。

2. **Why Pure RL Is Risky**
   纯 RL 很难保证深沟和人员安全。

3. **Proposed System**
   多传感器感知 + costmap + local planner + RL/diffusion 辅助 + safety filter。

4. **MVP Plan**
   先实现低速保守导航，再引入学习模块。

5. **Research Extension**
   Safety-Aware MoE / Diffusion Planner / VLM Supervisor。

6. **Evaluation Metrics**
   安全距离、碰撞率、接管率、成功率、人员停车距离。

7. **Risks and Requirements**
   需要 LiDAR / Depth / 右侧传感器 / 急停 / 现场测试规范。

---

## 18. 推荐项目标题

你可以把项目命名为：

> **Safety-Aware Learning-Augmented Navigation for Tunnel Inspection UGVs**

或者中文：

> **面向隧道工程场景的安全感知学习增强无人车导航系统**

更研究化一点：

> **A Safety-Aware MoE Navigation Framework for Tunnel UGVs under Open-Set Obstacles and High-Risk Boundaries**

---

## 19. 参考资料与可借鉴方向

> 以下资料用于支撑技术路线选择。实际项目落地时，需要结合公司硬件、现场规范和测试条件重新确认。

| 方向 | 资料 |
|---|---|
| ROS 2 Navigation / Nav2 | https://docs.nav2.org/ |
| Nav2 Costmap | https://docs.nav2.org/configuration/packages/configuring-costmaps.html |
| Nav2 Regulated Pure Pursuit | https://docs.nav2.org/configuration/packages/configuring-regulated-pp.html |
| Diffusion Policy | https://arxiv.org/abs/2303.04137 |
| RT-2 / VLA | https://arxiv.org/abs/2307.15818 |
| OpenVLA | https://arxiv.org/abs/2406.09246 |
| Segment Anything | https://arxiv.org/abs/2304.02643 |
| Grounding DINO | https://arxiv.org/abs/2303.05499 |
| Isaac Lab | https://isaac-sim.github.io/IsaacLab/ |
| FAST-LIO2 | https://arxiv.org/abs/2107.06829 |
| LIO-SAM | https://arxiv.org/abs/2007.00258 |
| Control Barrier Functions | https://coogan.ece.gatech.edu/papers/pdf/amesecc19.pdf |

---

## 20. 最终建议

### 推荐路线

**第一阶段：传统安全 baseline**

- 多传感器感知
- 局部 costmap
- 右侧深沟 keep-out zone
- 人员检测停车
- 低速 autonomous demo

**第二阶段：语义增强**

- 识别人、工具、工程车、未知障碍
- semantic costmap
- open-vocabulary detection + segmentation

**第三阶段：学习增强**

- RL local planner
- diffusion trajectory proposal
- MoE expert selection
- VLM high-level scene reasoning

**第四阶段：安全验证**

- safety filter
- FMEA
- field tests
- failure case replay
- quantitative metrics

### 不推荐路线

不建议把项目讲成：

> “用一个大模型 / 一个 RL policy 直接让车在隧道里自动开。”

这在论文 demo 中可能吸引人，但在右侧有深沟、场内有人和工程器械的真实项目中风险太高。

### 最合理的表达

建议你在公司内部把它表达为：

> **我们不是要用 RL 替代传统导航，而是用 RL / Diffusion / VLM 提升传统导航在开放障碍物、复杂隧道边界和动态施工场景中的鲁棒性；最终控制仍由安全约束层保证。**

这会更符合工程安全逻辑，也更容易获得项目认可。
