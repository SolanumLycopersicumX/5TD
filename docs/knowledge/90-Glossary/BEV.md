---
title: BEV
type: glossary
status: active
route: shared
source:
  - baselines/hbdnet_rt/docs/module_interfaces.md
  - docs/project_evaluation/tunnel_ugv_lidar_rgb_transformer_navigation_plan.md
created: 2026-06-23
updated: 2026-06-23
tags:
  []
---

# BEV

BEV 是 bird's-eye view，即鸟瞰视角的地面坐标表达，用于描述占用、风险和局部轨迹。

## 在 5TD 中

[[Mapping 模块]] 把图像 mask 投影到 BEV 风格栅格，使 [[DWAPlanner]] 能在车辆坐标系附近评估运动。

## 相关

- [[Costmap 与 Risk Grid]]
- [[Semantic Risk Costmap]]
- [[DWA]]

## 来源

- [模块接口](../../../baselines/hbdnet_rt/docs/module_interfaces.md)
- [LiDAR-RGB Transformer 项目评估](../../../docs/project_evaluation/tunnel_ugv_lidar_rgb_transformer_navigation_plan.md)
