---
title: Hard Boundary
type: glossary
status: active
route: shared
source:
  - baselines/hbdnet_rt/README.md
  - baselines/hbdnet_rt/docs/engineering_notes.md
  - baselines/hbdnet_rt/docs/training_guide.md
  - data/annotation_batches/rgb_keyframes_2026-06-22/annotation_rules.md
created: 2026-06-23
updated: 2026-06-23
tags:
  - #risk/safety-critical
---

# Hard Boundary

Hard Boundary 是不可跨越的物理或安全边界，例如沟边、隔离结构或隧道墙根。

## 在 5TD 中

[[Perception 模块]] 估计 hard-boundary mask。[[Mapping 模块]] 把它们转成高风险或占用栅格，[[DWAPlanner]] 应拒绝穿越这些区域的轨迹。

## 相关

- [[Trench Keep-out]]
- [[Safety Filter]]
- [[Semantic Risk Costmap]]

## 来源

- [HBD-Net-RT README](../../../baselines/hbdnet_rt/README.md)
- [工程笔记](../../../baselines/hbdnet_rt/docs/engineering_notes.md)
- [训练指南](../../../baselines/hbdnet_rt/docs/training_guide.md)
- [标注规则](../../../data/annotation_batches/rgb_keyframes_2026-06-22/annotation_rules.md)
