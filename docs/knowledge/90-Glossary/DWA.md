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

DWA 是 Dynamic Window Approach，一种局部规划方法，通过采样可行速度和转角并评分模拟轨迹来选动作。

## 在 5TD 中

[[DWAPlanner]] 使用 [[Mapping 模块]] 输出的 risk grid 避免碰撞，并产生局部控制建议。

## 相关

- [[DWAPlanner]]
- [[Safety Filter]]
- [[Costmap 与 Risk Grid]]

## 来源

- [模块接口](../../../baselines/hbdnet_rt/docs/module_interfaces.md)
- [文件导读](../../../baselines/hbdnet_rt/docs/FILE_GUIDE.md)
- [规划配置](../../../baselines/hbdnet_rt/configs/planner.yaml)
- [DWA 测试](../../../baselines/hbdnet_rt/tests/test_dwa.py)
