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

`DWAPlanner` 采样局部速度和转角候选，模拟轨迹，拒绝碰撞轨迹，并选择目标速度和转角。

## 输入

- 当前位姿和速度。
- 来自 [[Mapping 模块]] 的 risk grid 和 grid extent。
- 可选目标方向。

## 输出

- 目标速度。
- 目标转角。
- 被选中的轨迹。
- `OK` 或 `NO_PATH` 等规划状态。
- 代价拆分和候选数量。

## 安全作用

穿越高风险区域，尤其是 [[Hard Boundary]] 和 [[Trench Keep-out]] 的轨迹不可行。`NO_PATH` 会成为 [[SafetyStateMachine]] 的安全输入。

## 相关

- [[DWA]]
- [[Costmap 与 Risk Grid]]
- [[ControlCommand]]

## 来源

- [模块接口](../../../../baselines/hbdnet_rt/docs/module_interfaces.md)
- [文件导读](../../../../baselines/hbdnet_rt/docs/FILE_GUIDE.md)
