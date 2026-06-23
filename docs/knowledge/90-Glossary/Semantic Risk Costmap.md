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

A semantic risk costmap represents traversability and obstacle risk using semantic perception signals, not only geometry.

## In 5TD

[[Perception 模块]] supplies semantic cues such as workers, vehicles, debris, passable area, and [[Hard Boundary]]. [[Mapping 模块]] turns those cues into risk values used by [[DWAPlanner]] and [[SafetyStateMachine]].

## Related

- [[Costmap 与 Risk Grid]]
- [[Transformer Fusion Research]]
- [[RGB Vision Research]]

## Source

- [Module interfaces](../../../baselines/hbdnet_rt/docs/module_interfaces.md)
- [Transformer fusion README](../../../research/transformer_fusion/README.md)
- [Fusion config](../../../configs/navigation/fusion.yaml)
- [LiDAR-RGB Transformer project evaluation](../../../docs/project_evaluation/tunnel_ugv_lidar_rgb_transformer_navigation_plan.md)
