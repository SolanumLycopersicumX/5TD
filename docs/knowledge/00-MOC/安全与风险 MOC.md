---
title: 安全与风险 MOC
type: moc
status: active
route: shared
source:
  []
created: 2026-06-23
updated: 2026-06-23
tags:
  - #risk/safety-critical
---

# 安全与风险 MOC

这张地图集中整理安全关键约束和故障控制概念。

## 架构入口

- [[安全约束导航]]
- [[Costmap 与 Risk Grid]]
- [[端到端运行流程]]

## 基线模块

- [[SafetyStateMachine]]
- [[ControlCommand]]
- [[DWAPlanner]]
- [[Mapping 模块]]

## 术语

- [[Safety Filter]]
- [[Hard Boundary]]
- [[Trench Keep-out]]
- [[Semantic Risk Costmap]]
- [[DWA]]

## 待验证问题

- RGB-only 感知是否能稳定看清右侧沟边？
- 项目要求的最小沟边安全距离是多少？
- 是否允许右侧 ToF、LiDAR 或测距传感器作为独立安全通道？
