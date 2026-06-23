---
title: Costmap 与 Risk Grid
type: architecture
status: active
route: shared
source:
  - baselines/hbdnet_rt/docs/module_interfaces.md
  - baselines/hbdnet_rt/docs/engineering_notes.md
created: 2026-06-23
updated: 2026-06-23
tags:
  - #module/mapping
  - #risk/safety-critical
---

# Costmap 与 Risk Grid

The costmap/risk-grid layer is the contract between perception and planning. It converts image-level results into ground-plane costs that the planner can reject or prefer.

## Inputs

- Hard-boundary mask from [[Perception 模块]]
- Ego-passable mask from [[Perception 模块]]
- Detection boxes and classes
- Confidence values

## Outputs

- Occupancy grid: binary blocked/free space used by [[Mapping 模块]].
- Risk grid: continuous risk values used by [[DWAPlanner]] and [[SafetyStateMachine]].

## Rule of Thumb

[[Hard Boundary]] and space outside ego-passable regions should become high-risk or occupied space. Candidate trajectories crossing risk values near `1.0` should be treated as collision candidates.

## Related

- [[BEV]]
- [[Semantic Risk Costmap]]
- [[Trench Keep-out]]

## Source

- [Module interfaces](../../../baselines/hbdnet_rt/docs/module_interfaces.md)
- [Engineering notes](../../../baselines/hbdnet_rt/docs/engineering_notes.md)
