---
title: DWAPlanner
type: interface
status: active
route: rgb-only
source:
  - baselines/hbdnet_rt/docs/module_interfaces.md
  - baselines/hbdnet_rt/docs/FILE_GUIDE.md
created: 2026-06-23
updated: 2026-06-23
tags:
  - #module/planning
  - #risk/safety-critical
---

# DWAPlanner

`DWAPlanner` samples local velocity and steering candidates, simulates trajectories, rejects collisions, and selects a target speed and steering angle.

## Inputs

- Current pose and velocity.
- Risk grid and grid extent from [[Mapping 模块]].
- Optional goal direction.

## Outputs

- Target speed.
- Target steering.
- Selected trajectory.
- Planner status such as `OK` or `NO_PATH`.
- Cost breakdown and candidate counts.

## Safety Role

Trajectories crossing high-risk regions, especially [[Hard Boundary]] and [[Trench Keep-out]], are infeasible. `NO_PATH` becomes a safety signal for [[SafetyStateMachine]].

## Related

- [[DWA]]
- [[Costmap 与 Risk Grid]]
- [[ControlCommand]]

## Source

- [Module interfaces](../../../../baselines/hbdnet_rt/docs/module_interfaces.md)
- [File guide](../../../../baselines/hbdnet_rt/docs/FILE_GUIDE.md)
