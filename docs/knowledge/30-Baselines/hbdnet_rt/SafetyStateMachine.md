---
title: SafetyStateMachine
type: interface
status: active
route: rgb-only
source:
  - baselines/hbdnet_rt/docs/module_interfaces.md
  - baselines/hbdnet_rt/docs/engineering_notes.md
created: 2026-06-23
updated: 2026-06-23
tags:
  - #module/safety
  - #risk/safety-critical
---

# SafetyStateMachine

`SafetyStateMachine` 把置信度、风险、距离和路径可行性信号转换为安全状态和限速比例。

## 输入

- 感知综合置信度。
- 最大风险。
- 边界距离。
- 工人距离。
- [[DWAPlanner]] 是否找到可行路径。

## 输出

- 从 S0 normal 到 S4 manual takeover 的安全状态。
- 速度限制比例。
- 刹车标志。
- 原因文本。

## 行为

安全升级立即执行。恢复需要连续稳定帧。STOP 和 TAKEOVER 通过 [[ControlCommand]] 强制零速。

## 相关

- [[Safety Filter]]
- [[安全约束导航]]
- [[Hard Boundary]]

## 来源

- [模块接口](../../../../baselines/hbdnet_rt/docs/module_interfaces.md)
- [工程笔记](../../../../baselines/hbdnet_rt/docs/engineering_notes.md)
