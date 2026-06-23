---
title: HBD-Net-RT 基线
type: baseline
status: active
route: rgb-only
source:
  - baselines/hbdnet_rt/README.md
  - baselines/hbdnet_rt/docs/engineering_notes.md
  - baselines/hbdnet_rt/docs/latency_budget.md
created: 2026-06-23
updated: 2026-06-23
tags:
  - #route/rgb-only
  - #status/active
---

# HBD-Net-RT 基线

HBD-Net-RT 是近期 MVP 的活跃 RGB-only 工程基线。它提供可运行的感知、映射、规划、安全状态处理和控制命令输出链路。

## 范围

它面向固定的后土建隧道半幅通行场景，输入为单目 RGB。当前不声称多隧道泛化、多传感器融合、ROS 2 集成或真实底盘控制。

## 管线

- [[Perception 模块]]
- [[Mapping 模块]]
- [[DWAPlanner]]
- [[SafetyStateMachine]]
- [[ControlCommand]]

## 运行状态

- 当前实现是工程骨架，模型部分仍包含 placeholder。
- 基线文档记录的目标是端到端延迟低于 100 ms，当前模拟 P95 约 37 ms。
- 主要阻塞是隧道标注数据以及真实训练/验证。

## 来源

- [HBD-Net-RT README](../../../../baselines/hbdnet_rt/README.md)
- [工程笔记](../../../../baselines/hbdnet_rt/docs/engineering_notes.md)
- [延迟预算](../../../../baselines/hbdnet_rt/docs/latency_budget.md)
