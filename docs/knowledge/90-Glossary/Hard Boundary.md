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

A hard boundary is a non-crossable physical or safety boundary, such as trench edge, separation structure, or tunnel wall.

## In 5TD

[[Perception 模块]] estimates hard-boundary masks. [[Mapping 模块]] converts them into high-risk or occupied cells, and [[DWAPlanner]] should reject trajectories crossing them.

## Related

- [[Trench Keep-out]]
- [[Safety Filter]]
- [[Semantic Risk Costmap]]

## Source

- [HBD-Net-RT README](../../../baselines/hbdnet_rt/README.md)
- [Engineering notes](../../../baselines/hbdnet_rt/docs/engineering_notes.md)
- [Training guide](../../../baselines/hbdnet_rt/docs/training_guide.md)
- [Annotation rules](../../../data/annotation_batches/rgb_keyframes_2026-06-22/annotation_rules.md)
