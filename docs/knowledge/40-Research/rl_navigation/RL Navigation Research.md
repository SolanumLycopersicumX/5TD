---
title: RL 导航研究
type: research
status: active
route: rl
source:
  - research/rl_navigation/README.md
  - configs/experiments/rl_default.yaml
  - configs/navigation/planner.yaml
  - docs/project_evaluation/tunnel_ugv_rl_navigation_project_evaluation.md
  - docs/project_evaluation/tunnel_ugv_lidar_rgb_transformer_navigation_plan.md
created: 2026-06-23
updated: 2026-06-23
tags:
  - #route/rl
  - #risk/safety-critical
---

# RL 导航研究

RL 导航研究关注安全约束下的局部导航策略。

## 允许的输出形态

RL 可以提出 waypoint、速度建议、候选动作或策略提示。语义风险 costmap 是推荐输入形态。这些输出必须由 [[DWAPlanner]]、[[SafetyStateMachine]] 和 [[Safety Filter]] 检查。

## 不允许的行为

RL 不应直接控制电机，也不能绕过 [[Hard Boundary]] 和 [[Trench Keep-out]] 等硬约束。

## 来源

- [RL navigation README](../../../../research/rl_navigation/README.md)
- [RL 实验配置](../../../../configs/experiments/rl_default.yaml)
- [规划配置](../../../../configs/navigation/planner.yaml)
- [RL 导航项目评估](../../../../docs/project_evaluation/tunnel_ugv_rl_navigation_project_evaluation.md)
- [LiDAR-RGB Transformer 项目评估](../../../../docs/project_evaluation/tunnel_ugv_lidar_rgb_transformer_navigation_plan.md)
