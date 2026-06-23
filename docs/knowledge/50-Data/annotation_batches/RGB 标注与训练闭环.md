---
title: RGB 标注与训练闭环
type: dataset
status: active
route: rgb-only
source:
  - data/README.md
  - baselines/hbdnet_rt/docs/training_guide.md
  - data/annotation_batches/rgb_keyframes_2026-06-22/annotation_rules.md
created: 2026-06-23
updated: 2026-06-23
tags:
  - #data/annotation
  - #route/rgb-only
---

# RGB 标注与训练闭环

The RGB annotation and training loop turns tunnel frames into labels that can train and validate the RGB-only perception baseline.

## Loop

1. Extract and curate representative tunnel frames.
2. Annotate free space, [[Hard Boundary]], obstacles, workers, engineering vehicles, and risk-relevant regions.
3. Train or fine-tune [[Perception 模块]].
4. Evaluate outputs through [[Mapping 模块]], [[DWAPlanner]], and [[SafetyStateMachine]].
5. Record results using [[实验记录格式]] and compare against [[验收指标]].

## Active Batch

- [[rgb_keyframes_2026-06-22]]

## Source

- [Data README](../../../../data/README.md)
- [Training guide](../../../../baselines/hbdnet_rt/docs/training_guide.md)
- [Annotation rules](../../../../data/annotation_batches/rgb_keyframes_2026-06-22/annotation_rules.md)
