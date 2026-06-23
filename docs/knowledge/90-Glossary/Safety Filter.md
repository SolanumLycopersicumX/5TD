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

Safety Filter 是项目约定的最终安全校验层，用来约束感知、规划或研究模块提出的运动结果。

## 在 5TD 中

实际安全过滤由 [[Costmap 与 Risk Grid]]、[[DWAPlanner]]、[[SafetyStateMachine]] 和 [[ControlCommand]] 共同构成。研究模块可以提出地图、轨迹或提示，但不能绕过这一层。

## 相关

- [[安全约束导航]]
- [[双路线技术策略]]
- [[RL Navigation Research|RL 导航研究]]
- [[Diffusion Planner Research|Diffusion 轨迹规划研究]]
- [[Transformer Fusion Research|Transformer 融合研究]]

## 来源

- [仓库 README](../../../README.md)
- [工程笔记](../../../baselines/hbdnet_rt/docs/engineering_notes.md)
- [导航安全配置](../../../configs/navigation/safety.yaml)
- [规划配置](../../../configs/navigation/planner.yaml)
- [RL 实验配置](../../../configs/experiments/rl_default.yaml)
- [Diffusion 实验配置](../../../configs/experiments/diffusion_default.yaml)
- [安全状态机测试](../../../baselines/hbdnet_rt/tests/test_safety_state_machine.py)
