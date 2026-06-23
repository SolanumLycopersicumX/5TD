---
title: Semantic Risk Costmap
type: glossary
status: active
route: shared
source:
  - baselines/hbdnet_rt/docs/module_interfaces.md
  - research/transformer_fusion/README.md
  - configs/navigation/fusion.yaml
  - docs/project_evaluation/tunnel_ugv_lidar_rgb_transformer_navigation_plan.md
created: 2026-06-23
updated: 2026-06-23
tags:
  - #module/mapping
---

# Semantic Risk Costmap

Semantic Risk Costmap 是融合几何占用和 RGB 语义风险的局部代价地图。

## 在 5TD 中

[[Perception 模块]] 提供工人、车辆、碎石、可通行区域和 [[Hard Boundary]] 等语义线索。[[Mapping 模块]] 把这些线索转换成风险值，供 [[DWAPlanner]] 和 [[SafetyStateMachine]] 使用。

## 相关

- [[Costmap 与 Risk Grid]]
- [[Transformer Fusion Research|Transformer 融合研究]]
- [[RGB Vision Research|RGB 视觉研究]]

## 来源

- [模块接口](../../../baselines/hbdnet_rt/docs/module_interfaces.md)
- [Transformer fusion README](../../../research/transformer_fusion/README.md)
- [融合配置](../../../configs/navigation/fusion.yaml)
- [LiDAR-RGB Transformer 项目评估](../../../docs/project_evaluation/tunnel_ugv_lidar_rgb_transformer_navigation_plan.md)
