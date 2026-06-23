---
title: Perception 模块
type: interface
status: active
route: rgb-only
source:
  - baselines/hbdnet_rt/docs/module_interfaces.md
  - baselines/hbdnet_rt/docs/FILE_GUIDE.md
created: 2026-06-23
updated: 2026-06-23
tags:
  - #module/perception
---

# Perception 模块

The perception module converts RGB images into detection, segmentation, boundary, edge, risk, and confidence outputs for the rest of the pipeline.

## Inputs

- RGB image tensor shaped for the HBD-Net-RT model.

## Outputs

- Object detections for construction vehicles, workers, suspended objects, and debris.
- Ego-passable mask.
- Hard-boundary mask and edge.
- Surface risk map.
- Confidence values used by [[SafetyStateMachine]].

## Key Files

- `perception/model.py`
- `perception/preprocessor.py`
- `perception/postprocess.py`
- `perception/inference.py`

## Related

- [[HBD-Net-RT Baseline]]
- [[Mapping 模块]]
- [[Hard Boundary]]

## Source

- [Module interfaces](../../../../baselines/hbdnet_rt/docs/module_interfaces.md)
- [File guide](../../../../baselines/hbdnet_rt/docs/FILE_GUIDE.md)
