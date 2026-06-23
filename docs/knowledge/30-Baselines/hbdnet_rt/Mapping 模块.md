---
title: Mapping 模块
type: interface
status: active
route: rgb-only
source:
  - baselines/hbdnet_rt/docs/module_interfaces.md
  - baselines/hbdnet_rt/docs/FILE_GUIDE.md
created: 2026-06-23
updated: 2026-06-23
tags:
  - #module/mapping
---

# Mapping 模块

The mapping module projects perception outputs into ground-plane occupancy and risk grids.

## Responsibilities

- Use calibration and BEV projection to map image masks to grid coordinates.
- Create occupancy from hard boundaries, non-passable regions, and detection areas.
- Create continuous risk for [[DWAPlanner]] and [[SafetyStateMachine]].

## Key Files

- `mapping/calibration.py`
- `mapping/bev_projector.py`
- `mapping/occupancy_grid.py`
- `mapping/risk_grid.py`

## Related

- [[BEV]]
- [[Costmap 与 Risk Grid]]
- [[Semantic Risk Costmap]]

## Source

- [Module interfaces](../../../../baselines/hbdnet_rt/docs/module_interfaces.md)
- [File guide](../../../../baselines/hbdnet_rt/docs/FILE_GUIDE.md)
