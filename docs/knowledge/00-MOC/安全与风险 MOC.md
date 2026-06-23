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

This map collects safety-critical constraints and failure-control concepts.

## Architecture

- [[安全约束导航]]
- [[Costmap 与 Risk Grid]]
- [[端到端运行流程]]

## Baseline Modules

- [[SafetyStateMachine]]
- [[ControlCommand]]
- [[DWAPlanner]]
- [[Mapping 模块]]

## Glossary

- [[Safety Filter]]
- [[Hard Boundary]]
- [[Trench Keep-out]]
- [[Semantic Risk Costmap]]
- [[DWA]]

## Open Validation Questions

- Is the right-side trench edge reliably visible with RGB-only perception?
- What minimum trench safety distance is required?
- Is a right-side ToF, LiDAR, or distance sensor allowed as an independent safety channel?
