---
title: ControlCommand
type: interface
status: active
route: rgb-only
source:
  - baselines/hbdnet_rt/docs/module_interfaces.md
  - baselines/hbdnet_rt/docs/FILE_GUIDE.md
created: 2026-06-23
updated: 2026-06-23
tags:
  - #module/control
  - #risk/safety-critical
---

# ControlCommand

`ControlCommand` 是基线最终输出格式，包含速度、转角、刹车、安全状态、原因和调试信息。

## 作用

它把 [[DWAPlanner]] 输出和 [[SafetyStateMachine]] 输出合并。如果安全状态是 STOP 或 TAKEOVER，则速度和转角强制为零，并启用刹车。

## 字段

- 时间戳。
- 目标速度。
- 目标转角。
- 刹车标志。
- 安全状态。
- 原因。
- 调试负载。

## 相关

- [[端到端运行流程]]
- [[Safety Filter]]
- [[部署边界]]

## 来源

- [模块接口](../../../../baselines/hbdnet_rt/docs/module_interfaces.md)
- [文件导读](../../../../baselines/hbdnet_rt/docs/FILE_GUIDE.md)
