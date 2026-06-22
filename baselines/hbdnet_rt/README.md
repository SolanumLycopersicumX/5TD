# HBD-Net-RT v1.0

隧道施工半幅通行纯视觉避障。固定场景，单目 RGB，实时闭环。

## 范围

**做**：单一隧道结构下的感知-规划-控制。RepVGG-lite Backbone + 5 Head 多任务网络 + Risk-Adaptive DWA + 安全状态机。

**不做**：多隧道泛化、复杂地形、仿生轮胎、多传感器融合、ROS2、真实底盘。

## 输出

检测框（4 类）、ego-passable mask、hard-boundary mask（4 类）、edge mask、surface risk map、BEV 占用/风险栅格、DWA 路径、控制命令。

## 约束

总延迟 < 100ms（实测 P95 ≈ 37ms）。hard-boundary 不可跨越。STOP/TAKEOVER 强制 speed=0。升态即时有降态需 5 帧确认。

## 快速开始

```bash
pip install torch numpy opencv-python pyyaml pytest
cd hbdnet_rt
export PYTHONPATH=src
```

### 端到端

```bash
python scripts/run_pipeline.py -n 50          # 完整管线
python scripts/run_dashboard.py               # 调试视图 (8面板)
python scripts/run_dashboard.py --camera      # 摄像头实时
```

### 单模块

```bash
python scripts/run_inference.py               # 感知推理
python scripts/run_planner_demo.py            # DWA 独立测试
python scripts/benchmark_latency.py -n 200    # 延迟统计
```

### 测试

```bash
pytest tests/ -v                              # 51 tests
python tests/test_scenarios.py                # 8 场景决策验证
```

## 结构

```
configs/             scene.yaml  model.yaml  planner.yaml  safety.yaml
src/hbdnet_rt/
  perception/        model  preprocessor  postprocess  inference
  mapping/           calibration  bev_projector  occupancy_grid  risk_grid
  planning/          dwa  trajectory  cost_functions
  safety/            state_machine
  control/           command
  utils/             config  timing  logger
scripts/             pipeline  dashboard  benchmark  visualize  inference  planner_demo
tests/               51 pytest + 8 scenarios
docs/                training_guide  module_interfaces  latency_budget  engineering_notes
```

## 模型

RepVGG-lite (~3.2M) → Lightweight FPN → 5 Heads（Detection / Ego-Passable / Hard-Boundary / Edge / SurfaceRisk）。所有 Head 共享 Backbone+FPN，一次前向输出全部结果。

## 状态

工程骨架完成。RepVGG-lite + 5 Head + Risk-Adaptive DWA + S0-S4 安全状态机全部就绪。51 tests + 8 场景全部通过。随机权重下全链路可跑。

**阻塞**：隧道标注数据。数据到齐 → 训练 → ONNX/TensorRT → 验收。

## 后续

训练策略见 `docs/training_guide.md`（8 章，含标注规范/损失函数/训练循环/部署导出/验收清单）。
