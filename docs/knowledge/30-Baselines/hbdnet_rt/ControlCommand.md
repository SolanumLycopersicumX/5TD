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

`ControlCommand` is the final baseline output format for speed, steering, braking, safety state, reason, and debug data.

## Role

It combines [[DWAPlanner]] output with [[SafetyStateMachine]] output. If the safety state is STOP or TAKEOVER, speed and steering are forced to zero and braking is enabled.

## Fields

- Timestamp.
- Target speed.
- Target steering.
- Brake flag.
- Safety state.
- Reason.
- Debug payload.

## Related

- [[端到端运行流程]]
- [[Safety Filter]]
- [[部署边界]]

## Source

- [Module interfaces](../../../../baselines/hbdnet_rt/docs/module_interfaces.md)
- [File guide](../../../../baselines/hbdnet_rt/docs/FILE_GUIDE.md)
