---
title: Diffusion Planner Research
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

# Diffusion Planner Research

Diffusion Planner Research generates multiple local trajectory proposals around obstacles.

## Contract

Generated trajectories are proposals. They can be conditioned on risk costmaps or short history, but they must be checked against occupancy, risk, trench keep-out zones, vehicle limits, and human-safety constraints before selection.

## Related

- [[DWAPlanner]]
- [[Costmap 与 Risk Grid]]
- [[Trench Keep-out]]
- [[Safety Filter]]

## Source

- [Diffusion planner README](../../../../research/diffusion_planner/README.md)
- [Diffusion experiment config](../../../../configs/experiments/diffusion_default.yaml)
- [LiDAR-RGB Transformer project evaluation](../../../../docs/project_evaluation/tunnel_ugv_lidar_rgb_transformer_navigation_plan.md)
