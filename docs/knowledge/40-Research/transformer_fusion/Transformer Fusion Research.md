---
title: Transformer 融合研究
type: research
status: active
route: fusion
source:
  - research/transformer_fusion/README.md
  - configs/navigation/fusion.yaml
  - docs/project_evaluation/tunnel_ugv_lidar_rgb_transformer_navigation_plan.md
created: 2026-06-23
updated: 2026-06-23
tags:
  - #route/fusion
  - #risk/safety-critical
---

# Transformer 融合研究

Transformer 融合研究探索 LiDAR-RGB 融合和时序场景理解。

## 有价值的输出

- 风险图。
- 可通行图。
- 语义 costmap。
- 候选轨迹。
- 时序一致性信号。

## 安全边界

融合模块不替代 [[Safety Filter]]。它的输出应进入 [[Costmap 与 Risk Grid]]、[[DWAPlanner]] 或 [[SafetyStateMachine]]，不能直接控制电机。共享 fusion 配置是这条路线的重要来源。

## 相关

- [[双路线技术策略]]
- [[Semantic Risk Costmap]]
- [[Trench Keep-out]]

## 来源

- [Transformer fusion README](../../../../research/transformer_fusion/README.md)
- [融合配置](../../../../configs/navigation/fusion.yaml)
- [LiDAR-RGB Transformer 项目评估](../../../../docs/project_evaluation/tunnel_ugv_lidar_rgb_transformer_navigation_plan.md)
