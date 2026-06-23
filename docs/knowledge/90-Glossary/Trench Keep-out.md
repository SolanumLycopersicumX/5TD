---
title: Trench Keep-out
type: glossary
status: active
route: shared
source:
  - configs/navigation/safety.yaml
  - docs/project_evaluation/2026-06-22-rgb-vision-route-addendum.md
  - docs/project_evaluation/tunnel_ugv_rl_navigation_project_evaluation.md
  - docs/project_evaluation/tunnel_ugv_lidar_rgb_transformer_navigation_plan.md
created: 2026-06-23
updated: 2026-06-23
tags:
  - #risk/safety-critical
---

# Trench Keep-out

Trench keep-out is the safety rule that the vehicle must not enter or cross the protected area around a trench or hard boundary.

## In 5TD

The exact distance requirement remains an open project-owner question. The implementation principle is clear: keep-out regions should be represented in [[Costmap 与 Risk Grid]] and enforced by [[DWAPlanner]] and [[Safety Filter]].

## Related

- [[Hard Boundary]]
- [[安全与风险 MOC]]
- [[验收指标]]

## Source

- [Navigation safety config](../../../configs/navigation/safety.yaml)
- [RGB pure-vision route addendum](../../../docs/project_evaluation/2026-06-22-rgb-vision-route-addendum.md)
- [RL navigation project evaluation](../../../docs/project_evaluation/tunnel_ugv_rl_navigation_project_evaluation.md)
- [LiDAR-RGB Transformer project evaluation](../../../docs/project_evaluation/tunnel_ugv_lidar_rgb_transformer_navigation_plan.md)
