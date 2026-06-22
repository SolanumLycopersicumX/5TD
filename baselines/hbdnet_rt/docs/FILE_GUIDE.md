# HBD-Net-RT 代码导读

每个文件是干什么的、依赖谁、给谁用。面向接手代码的开发者。

---

## 总览：调用关系

```
scripts/           ← 入口脚本，你从这里开始跑
    │
src/hbdnet_rt/     ← 核心代码
    │
    ├── perception/    图像进，感知结果出
    ├── mapping/       感知结果 → 地面栅格
    ├── planning/      栅格 → 速度和转向角
    ├── safety/        对所有输出做安全修正
    ├── control/       统一控制命令格式
    └── utils/         配置加载、计时、日志

configs/            ← 所有参数（改参数改这里，不动代码）
tests/              ← 每个模块的测试
docs/               ← 文档
```

---

## 入口脚本（scripts/）

### run_pipeline.py
端到端管线。一帧图像进、控制命令出。串联了全部模块。

**用途**：验证全链路能跑通、测延迟。

**用法**：
```bash
python run_pipeline.py -n 50          # 跑50帧，输出延迟统计
python run_pipeline.py --input a.jpg   # 输入真实图片
```
调用了：perception / mapping / planning / safety / control（全部）。

---

### run_dashboard.py
实时调试视图。打开一个窗口，分 8 个面板显示每一步的中间结果。

**用途**：训练后看模型效果、定位是哪一步出了问题。

**用法**：
```bash
python run_dashboard.py                # 无摄像头模式（随机画面）
python run_dashboard.py --camera       # USB摄像头实时
python run_dashboard.py --video a.mp4  # 视频文件
```

面板含义：
1. Original — 原始画面
2. Detections — 检测框（框在哪、类别和置信度）
3. Ego-Passable — 本车可通行区域（轮廓线）
4. Hard-Boundary — 隔离沟/隔离带/隧道壁（轮廓线）
5. Edge — 边界锐化
6. Risk Grid — 风险栅格+规划轨迹
7. Occupancy — 占用栅格（白=可通/黑=不可通）
8. Stats — 数值面板（状态/速度/转向/置信度/警告）

调用了：全部模块。

---

### run_inference.py
仅跑感知部分：图像进，检测框+mask+置信度出。不跑规划和决策。

**用途**：单独调试模型和后处理，不关心 DWA 和安全。

**用法**：
```bash
python run_inference.py
```
调用了：perception 模块。

---

### run_planner_demo.py
仅跑 DWA 路径规划。使用手写 risk_grid（不依赖模型）。展示空旷/有障碍时的路径选择。

**用途**：验证 DWA 逻辑正确——空旷直行、遇障绕行、全堵停车。

**用法**：
```bash
python run_planner_demo.py
```
调用了：planning 模块。

---

### benchmark_latency.py
统计每步耗时：预处理、模型推理、后处理、栅格生成、DWA、安全状态机。

**用途**：定位性能瓶颈，确认是否满足 <100ms。

**用法**：
```bash
python benchmark_latency.py -n 200
```

---

### visualize_outputs.py
单帧可视化，输出为一张 jpg。跟 dashboard 的区别是：这个是静态的，适合写报告截图用。

**用法**：
```bash
python visualize_outputs.py -i input.jpg -o output.jpg
```

---

## 核心模块（src/hbdnet_rt/）

### perception/ — 感知

| 文件 | 做什么 | 关键类/函数 | 被谁调用 |
|------|--------|-----------|---------|
| `model.py` | 神经网络定义。RepVGG-lite + FPN + 5个Head。forward 输入图像 tensor，输出检测+分割+风险 | `HBDNetRT` | inference.py |
| `preprocessor.py` | 图像预处理：letterbox缩放 + CLAHE增强 + BGR→RGB→tensor | `ImagePreprocessor` | run_pipeline / run_dashboard |
| `postprocess.py` | 后处理：sigmoid/softmax + threshold + top-K筛选 | `PostProcessor` | inference.py |
| `inference.py` | 推理管线：model.forward + 后处理，统一对外接口 | `PerceptionInference` | run_pipeline / run_dashboard |

---

### mapping/ — 空间映射（图像坐标 → 地面坐标）

| 文件 | 做什么 | 关键类 | 被谁调用 |
|------|--------|--------|---------|
| `calibration.py` | 相机标定。从安装高度/俯仰/FOV 计算图像→地面 homography 矩阵 | `CameraCalibration` | bev_projector |
| `bev_projector.py` | BEV 投影器。把图像坐标的 mask 投影到地面栅格坐标系 | `BEVProjector` | occupancy / risk |
| `occupancy_grid.py` | 占用栅格。三规则融合：hard_boundary占、ego_passable外占、检测框区域占。输出 0/1 矩阵 | `OccupancyGrid` | run_pipeline |
| `risk_grid.py` | 风险栅格。在占用基础上，按障碍类别和置信度分配 0~1 连续风险值 | `RiskGrid` | run_pipeline |

---

### planning/ — 路径规划

| 文件 | 做什么 | 关键类/函数 | 被谁调用 |
|------|--------|-----------|---------|
| `trajectory.py` | 自行车模型前向模拟。给定起点、速度、转向角，模拟 N 步后的轨迹点 | `simulate_trajectory` | dwa.py |
| `cost_functions.py` | Risk-Adaptive 四因子评分：clearance + risk_cost + smoothness + progress。查询 risk_grid 做碰撞检测 | `compute_costs` | dwa.py |
| `dwa.py` | DWA 规划器。速度/转角采样 → 每条候选轨迹评分 → 选最优。输出速度和转向角 | `DWAPlanner` | run_pipeline |

---

### safety/ — 安全状态机

| 文件 | 做什么 | 关键类 | 被谁调用 |
|------|--------|--------|---------|
| `state_machine.py` | S0-S4 状态机。根据置信度/风险/距离确定安全状态，修正 DWA 输出 | `SafetyStateMachine` | run_pipeline |

状态含义：
- S0_NORMAL — 正常行驶
- S1_CAUTIOUS — 限速50%
- S2_SLOWDOWN — 限速25%
- S3_STOP — 速度为0
- S4_MANUAL_TAKEOVER — 人工接管

---

### control/ — 控制命令

| 文件 | 做什么 | 关键类 | 被谁调用 |
|------|--------|--------|---------|
| `command.py` | 统一输出格式：时间戳、速度、转向、刹车、安全状态、原因 | `ControlCommand` | run_pipeline |

---

### utils/ — 工具

| 文件 | 做什么 | 关键类/函数 | 被谁调用 |
|------|--------|-----------|---------|
| `config.py` | YAML 配置加载，把所有 configs/*.yaml 读入一个 Config 对象 | `load_config` | 所有模块 |
| `timing.py` | 各阶段延迟统计（上下文管理器用法） | `Timer` | benchmark / pipeline |
| `logger.py` | 统一日志格式 | `get_logger` | 所有模块 |

---

## 配置文件（configs/）

| 文件 | 内容 | 改什么 |
|------|------|--------|
| `fixed_scene.yaml` | 相机参数、模型输入尺寸、检测/分割阈值、BEV栅格范围 | 改相机分辨率、改置信度阈值 |
| `model.yaml` | 网络架构参数、Head配置、训练参数 | 改 backbone 类型、改 Head 输出类别数 |
| `planner.yaml` | DWA 速度/转角采样范围、代价权重、车辆尺寸、安全阈值 | 改车速范围、改安全余量 |
| `safety.yaml` | 置信度/风险/距离阈值、速度限制比例、恢复帧数 | 改降级灵敏度 |

---

## 测试（tests/）

| 文件 | 测什么 |
|------|--------|
| `test_config.py` | 配置文件是否能正常加载 |
| `test_model_shapes.py` | 模型 forward 输出维度是否正确 |
| `test_postprocess.py` | 后处理格式是否正确 |
| `test_occupancy_grid.py` | BEV 投影形状、占用栅格规则（全通/全堵）、风险栅格范围 |
| `test_dwa.py` | 四场景：空旷直行、前方障碍绕行、硬边界不穿越、全堵停车 |
| `test_safety_state_machine.py` | 16项：各级别触发、零速强制、恢复逻辑、接管触发 |
| `test_scenarios.py` | 8个端到端决策场景（独立运行，不依赖pytest） |

---

## 文档（docs/）

| 文件 | 内容 |
|------|------|
| `training_guide.md` | 完整训练方案：数据标注规范、损失函数、训练循环、部署导出 |
| `module_interfaces.md` | 每个模块的输入/输出格式定义 |
| `latency_budget.md` | 各模块延迟预算和实测数据 |
| `engineering_notes.md` | 工程假设、默认值、占位说明、后续TODO |
