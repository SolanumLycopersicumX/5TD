---
title: HBD-Net-RT Baseline
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

# HBD-Net-RT Baseline

HBD-Net-RT is the active RGB-only engineering baseline for the near-term MVP. It provides a runnable chain for perception, mapping, planning, safety-state handling, and control-command output.

## Scope

It targets a fixed post-civil tunnel half-lane scenario with single-camera RGB input. It does not currently claim multi-tunnel generalization, multisensor fusion, ROS 2 integration, or real chassis control.

## Pipeline

- [[Perception 模块]]
- [[Mapping 模块]]
- [[DWAPlanner]]
- [[SafetyStateMachine]]
- [[ControlCommand]]

## Operating Notes

- Current implementation is an engineering skeleton with placeholder model pieces.
- Reported benchmark target is end-to-end latency below 100 ms, with current simulated P95 around 37 ms in the baseline docs.
- The main blocker is tunnel annotation data and real training/validation.

## Sources

- [HBD-Net-RT README](../../../../baselines/hbdnet_rt/README.md)
- [Engineering notes](../../../../baselines/hbdnet_rt/docs/engineering_notes.md)
- [Latency budget](../../../../baselines/hbdnet_rt/docs/latency_budget.md)
