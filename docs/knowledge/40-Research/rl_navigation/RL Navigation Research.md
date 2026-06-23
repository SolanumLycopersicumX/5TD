---
title: RL Navigation Research
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

# RL Navigation Research

RL Navigation Research covers safety-constrained reinforcement learning for local navigation.

## Allowed Output Shape

RL can propose waypoints, velocity suggestions, candidate actions, or policy hints. A semantic risk costmap is the preferred input shape. These outputs must be checked by [[DWAPlanner]], [[SafetyStateMachine]], and [[Safety Filter]].

## Not Allowed

RL should not directly control motors or bypass hard constraints such as [[Hard Boundary]] and [[Trench Keep-out]].

## Source

- [RL navigation README](../../../../research/rl_navigation/README.md)
- [RL experiment config](../../../../configs/experiments/rl_default.yaml)
- [Planner config](../../../../configs/navigation/planner.yaml)
- [RL navigation project evaluation](../../../../docs/project_evaluation/tunnel_ugv_rl_navigation_project_evaluation.md)
- [LiDAR-RGB Transformer project evaluation](../../../../docs/project_evaluation/tunnel_ugv_lidar_rgb_transformer_navigation_plan.md)
