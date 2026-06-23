---
title: 研究路线 MOC
type: moc
status: active
route: shared
source:
  []
created: 2026-06-23
updated: 2026-06-23
tags:
  []
---

# 研究路线 MOC

This map tracks research routes that can improve the baseline while staying behind planner and safety validation.

## Tracks

- [[RGB Vision Research]]
- [[Transformer Fusion Research]]
- [[RL Navigation Research]]
- [[Diffusion Planner Research]]
- [[VLM Supervisor Research]]

## Required Safety Boundary

Research modules may produce perception outputs, risk maps, semantic costmaps, candidate trajectories, waypoints, or supervisory signals. They must not bypass [[DWAPlanner]], [[SafetyStateMachine]], or [[Safety Filter]] to control motors directly.

## Related Architecture

- [[双路线技术策略]]
- [[Costmap 与 Risk Grid]]
- [[基线模块晋升到 tunnel_nav]]
