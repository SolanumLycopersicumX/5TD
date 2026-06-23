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

Trench Keep-out 是围绕深沟或硬边界设置的禁入区和安全裕度规则。

## 在 5TD 中

具体距离仍需项目负责人确认。实现原则明确：keep-out 区域应进入 [[Costmap 与 Risk Grid]]，并由 [[DWAPlanner]] 和 [[Safety Filter]] 强制执行。

## 相关

- [[Hard Boundary]]
- [[安全与风险 MOC]]
- [[验收指标]]

## 来源

- [导航安全配置](../../../configs/navigation/safety.yaml)
- [RGB 纯视觉路线补充说明](../../../docs/project_evaluation/2026-06-22-rgb-vision-route-addendum.md)
- [RL 导航项目评估](../../../docs/project_evaluation/tunnel_ugv_rl_navigation_project_evaluation.md)
- [LiDAR-RGB Transformer 项目评估](../../../docs/project_evaluation/tunnel_ugv_lidar_rgb_transformer_navigation_plan.md)
