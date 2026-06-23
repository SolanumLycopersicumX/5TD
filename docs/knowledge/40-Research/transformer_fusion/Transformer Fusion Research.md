---
title: Transformer Fusion Research
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

# Transformer Fusion Research

Transformer Fusion Research explores LiDAR-RGB fusion and temporal scene understanding.

## Useful Outputs

- Risk maps.
- Traversability maps.
- Semantic costmaps.
- Candidate trajectories.
- Temporal consistency signals.

## Safety Boundary

Fusion does not replace [[Safety Filter]]. Its outputs should feed [[Costmap 与 Risk Grid]], [[DWAPlanner]], or [[SafetyStateMachine]], not direct motor control. The shared fusion config is part of the source trail for this route.

## Related

- [[双路线技术策略]]
- [[Semantic Risk Costmap]]
- [[Trench Keep-out]]

## Source

- [Transformer fusion README](../../../../research/transformer_fusion/README.md)
- [Fusion config](../../../../configs/navigation/fusion.yaml)
- [LiDAR-RGB Transformer project evaluation](../../../../docs/project_evaluation/tunnel_ugv_lidar_rgb_transformer_navigation_plan.md)
