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

BEV means bird's-eye view: a ground-plane representation used to reason about occupancy, risk, and local trajectories.

## In 5TD

[[Mapping 模块]] projects image masks into BEV-style grids so [[DWAPlanner]] can score motion in vehicle-centric space.

## Related

- [[Costmap 与 Risk Grid]]
- [[Semantic Risk Costmap]]
- [[DWA]]

## Source

- [Module interfaces](../../../baselines/hbdnet_rt/docs/module_interfaces.md)
- [LiDAR-RGB Transformer project evaluation](../../../docs/project_evaluation/tunnel_ugv_lidar_rgb_transformer_navigation_plan.md)
