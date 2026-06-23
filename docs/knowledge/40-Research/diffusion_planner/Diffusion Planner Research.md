---
title: Diffusion 轨迹规划研究
type: research
status: active
route: diffusion
source:
  - research/diffusion_planner/README.md
  - configs/experiments/diffusion_default.yaml
  - docs/project_evaluation/tunnel_ugv_lidar_rgb_transformer_navigation_plan.md
created: 2026-06-23
updated: 2026-06-23
tags:
  - #route/diffusion
  - #module/planning
  - #risk/safety-critical
---

# Diffusion 轨迹规划研究

Diffusion 轨迹规划研究用于围绕障碍物生成多条局部候选轨迹。

## 合同边界

生成轨迹只是 proposal。它可以基于风险 costmap 或短历史进行条件生成，但最终必须检查占用、风险、沟边 keep-out、车辆限制和人员安全约束。

## 相关

- [[DWAPlanner]]
- [[Costmap 与 Risk Grid]]
- [[Trench Keep-out]]
- [[Safety Filter]]

## 来源

- [Diffusion planner README](../../../../research/diffusion_planner/README.md)
- [Diffusion 实验配置](../../../../configs/experiments/diffusion_default.yaml)
- [LiDAR-RGB Transformer 项目评估](../../../../docs/project_evaluation/tunnel_ugv_lidar_rgb_transformer_navigation_plan.md)
