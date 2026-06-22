# 工程笔记

## 当前阶段: Round 1 — 工程骨架

所有模块使用 Placeholder 实现，forward / run / test 均可跑通。
不包含任何真实训练权重或生产级算法。

## 交付范围

固定隧道施工半幅通行场景。明确不交付:
- 复杂地形泛化、Surface Risk Head 高级版本
- 仿生轮胎能力配置 (VehicleCapabilityProfile)
- 多传感器融合、ROS2 集成、真实底盘驱动

## 模块占位说明

| 模块 | 当前实现 | 后续替换 |
|------|---------|---------|
| `perception/model.py::LightweightBackbone` | 4层简单卷积 | MobileNetV4-small / RepVGG-lite |
| `perception/postprocess.py::PostProcessor` | top-K 替代 NMS | 完整 NMS + IoU 过滤 |
| `mapping/bev_projector.py::BEVProjector` | 简化线性映射 | 相机标定 + homography |
| `planning/dwa.py::DWAPlanner` | 逐轨迹循环评估 | 向量化批量评估 |
| `safety/state_machine.py::SafetyStateMachine` | 规则逻辑已完整 | 阈值待实车调参 |

## 已知假设与默认值

- 相机高度: 3.5m (待实车测量)
- 相机俯仰角: 12° (待实车测量)
- 车身宽度: 2.0m, 长度: 3.0m, 轴距: 2.0m (待实车测量)
- 半幅车道宽: 3.5m (中国隧道标准估算)
- 安全余量: 0.25m (可调)
- 所有默认值均在 YAML 配置文件中，获取实车数据后可直接修改

## 安全逻辑

### 多层防护
1. **感知层**: hard-boundary mask 检测隔离沟/隔离带 → 漏检回退传统 CV
2. **栅格层**: hard-boundary 强制 risk=1.0, ego-passable 外 risk=1.0 → 安全膨胀
3. **规划层**: 穿越 risk≥0.99 轨迹直接判为不可行 → 全碰撞输出 NO_PATH
4. **安全状态机**: 低置信度/高风险/无路径 → 逐级升级至 STOP/TAKEOVER
5. **控制层**: STOP/TAKEOVER 强制 speed=0 + steering=0 + brake=True

### 状态转换
- 升级 (NORMAL→CAUTIOUS→...→STOP): 立即执行, 不防抖 (安全优先)
- 降级 (STOP→...→NORMAL): 需连续 5 帧确认 (防反复横跳)
- 连续 10 帧 STOP 或 30 帧异常 → MANUAL_TAKEOVER

## 后续 TODO

### Round 2 (感知增强)
- [ ] 替换 Backbone 为 MobileNetV4-small
- [ ] 实现完整 NMS
- [ ] 实现基于相机标定的精确 BEV 投影
- [ ] 采集 2000+ 张隧道场景数据 + 标注
- [ ] 三阶段训练 (冻结 backbone → 解冻 → 全网络微调)

### Round 3 (部署优化)
- [ ] 导出 ONNX + TensorRT FP16, 推理延迟 <10ms
- [ ] DWA 向量化
- [ ] Python → C++ 迁移后处理/BEV/DWA

### Round 4 (系统集成)
- [ ] 时域降噪 (3DNR) + 暗通道去雾
- [ ] 曝光状态感知模块
- [ ] 端到端延迟 P95 < 50ms
- [ ] 真实隧道视频端到端测试 + 验收
