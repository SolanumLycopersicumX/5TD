---
title: Safety Filter
type: glossary
status: active
route: shared
source:
  - README.md
  - baselines/hbdnet_rt/docs/engineering_notes.md
  - configs/navigation/safety.yaml
  - configs/navigation/planner.yaml
  - configs/experiments/rl_default.yaml
  - configs/experiments/diffusion_default.yaml
  - baselines/hbdnet_rt/tests/test_safety_state_machine.py
created: 2026-06-23
updated: 2026-06-23
tags:
  - #risk/safety-critical
---

# Safety Filter

Safety filter is the project rule that final motion must be constrained by safety checks after perception, planning, or research modules propose outputs.

## In 5TD

The practical safety filter combines [[Costmap 与 Risk Grid]], [[DWAPlanner]], [[SafetyStateMachine]], and [[ControlCommand]]. Research modules can propose maps, trajectories, or hints, but cannot bypass this layer.

## Related

- [[安全约束导航]]
- [[双路线技术策略]]
- [[RL Navigation Research]]
- [[Diffusion Planner Research]]
- [[Transformer Fusion Research]]

## Source

- [Repository README](../../../README.md)
- [Engineering notes](../../../baselines/hbdnet_rt/docs/engineering_notes.md)
- [Navigation safety config](../../../configs/navigation/safety.yaml)
- [Planner config](../../../configs/navigation/planner.yaml)
- [RL experiment config](../../../configs/experiments/rl_default.yaml)
- [Diffusion experiment config](../../../configs/experiments/diffusion_default.yaml)
- [Safety state machine tests](../../../baselines/hbdnet_rt/tests/test_safety_state_machine.py)
