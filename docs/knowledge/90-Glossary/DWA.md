---
title: DWA
type: glossary
status: active
route: shared
source:
  - baselines/hbdnet_rt/docs/module_interfaces.md
  - baselines/hbdnet_rt/docs/FILE_GUIDE.md
  - baselines/hbdnet_rt/configs/planner.yaml
  - baselines/hbdnet_rt/tests/test_dwa.py
created: 2026-06-23
updated: 2026-06-23
tags:
  []
---

# DWA

DWA means Dynamic Window Approach, a local planning method that samples feasible velocity and steering commands and scores simulated trajectories.

## In 5TD

[[DWAPlanner]] uses risk-grid information from [[Mapping 模块]] to avoid collisions and choose a local command proposal.

## Related

- [[DWAPlanner]]
- [[Safety Filter]]
- [[Costmap 与 Risk Grid]]

## Source

- [Module interfaces](../../../baselines/hbdnet_rt/docs/module_interfaces.md)
- [File guide](../../../baselines/hbdnet_rt/docs/FILE_GUIDE.md)
- [Planner config](../../../baselines/hbdnet_rt/configs/planner.yaml)
- [DWA tests](../../../baselines/hbdnet_rt/tests/test_dwa.py)
