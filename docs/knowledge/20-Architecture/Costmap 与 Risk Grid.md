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

Costmap / Risk Grid 是感知和规划之间的契约层。它把图像级结果转换成地面坐标系下的代价，让规划器判断哪些轨迹应拒绝、哪些更优。

## 输入

- [[Perception 模块]] 输出的 hard-boundary mask。
- [[Perception 模块]] 输出的 ego-passable mask。
- 检测框和类别。
- 置信度。

## 输出

- 占用栅格：[[Mapping 模块]] 使用的二值 blocked/free 表达。
- 风险栅格：[[DWAPlanner]] 和 [[SafetyStateMachine]] 使用的连续风险值。

## 规则要点

[[Hard Boundary]] 和 ego-passable 之外的区域应成为高风险或占用区域。穿越接近 `1.0` 风险值的候选轨迹应按碰撞处理。

## 相关

- [[BEV]]
- [[Semantic Risk Costmap]]
- [[Trench Keep-out]]

## 来源

- [模块接口](../../../baselines/hbdnet_rt/docs/module_interfaces.md)
- [工程笔记](../../../baselines/hbdnet_rt/docs/engineering_notes.md)
