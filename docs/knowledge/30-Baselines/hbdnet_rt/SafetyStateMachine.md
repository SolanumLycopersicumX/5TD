---
title: SafetyStateMachine
type: interface
status: active
route: rgb-only
source:
  - baselines/hbdnet_rt/docs/module_interfaces.md
  - baselines/hbdnet_rt/docs/engineering_notes.md
created: 2026-06-23
updated: 2026-06-23
tags:
  - #module/safety
  - #risk/safety-critical
---

# SafetyStateMachine

`SafetyStateMachine` converts confidence, risk, distance, and path-feasibility signals into a safety state and speed limit.

## Inputs

- Overall perception confidence.
- Maximum risk.
- Boundary distance.
- Worker distance.
- Whether [[DWAPlanner]] found a feasible path.

## Outputs

- Safety state from S0 normal to S4 manual takeover.
- Speed limit ratio.
- Brake flag.
- Reason text.

## Behavior

Escalation is immediate for safety. Recovery requires consecutive stable frames. STOP and TAKEOVER force zero speed through [[ControlCommand]].

## Related

- [[Safety Filter]]
- [[安全约束导航]]
- [[Hard Boundary]]

## Source

- [Module interfaces](../../../../baselines/hbdnet_rt/docs/module_interfaces.md)
- [Engineering notes](../../../../baselines/hbdnet_rt/docs/engineering_notes.md)
