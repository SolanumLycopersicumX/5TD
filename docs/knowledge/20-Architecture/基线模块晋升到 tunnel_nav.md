---
title: 基线模块晋升到 tunnel_nav
type: architecture
status: active
route: shared
source:
  - src/tunnel_nav/README.md
created: 2026-06-23
updated: 2026-06-23
tags:
  []
---

# 基线模块晋升到 tunnel_nav

`src/tunnel_nav/` is the future integrated engineering package. Stable modules can be promoted there after their interfaces, tests, and safety behavior are clear.

## Candidate Responsibilities

- Sensor adapters
- Calibration
- Perception
- Mapping and costmaps
- Planning
- Safety filtering
- Control-command interfaces
- Evaluation utilities
- ROS 2 interfaces

## Promotion Rule

Keep experimental work in `baselines/` or `research/` until the interface is stable. Promote only modules that have clear inputs, outputs, validation commands, and safety boundaries.

## Related

- [[HBD-Net-RT Baseline]]
- [[双路线技术策略]]
- [[部署边界]]

## Source

- [tunnel_nav README](../../../src/tunnel_nav/README.md)
