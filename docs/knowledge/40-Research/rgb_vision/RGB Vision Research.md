---
title: RGB Vision Research
type: research
status: active
route: rgb-only
source:
  - research/rgb_vision/README.md
  - docs/project_evaluation/2026-06-22-rgb-vision-route-addendum.md
  - configs/navigation/rgb_only.yaml
  - configs/navigation/perception.yaml
created: 2026-06-23
updated: 2026-06-23
tags:
  - #route/rgb-only
  - #module/perception
---

# RGB Vision Research

RGB Vision Research covers pure-RGB improvements beyond the current HBD-Net-RT baseline.

## Focus

- Free-space segmentation.
- [[Hard Boundary]] and trench-edge detection.
- Obstacle, person, and engineering-vehicle perception.
- Temporal consistency and failure detection.
- Comparison against [[HBD-Net-RT Baseline]].

## Validation Trigger

If RGB-only trench or right-boundary visibility is not reliable enough, the route should add right-side ToF, LiDAR, or fusion support rather than weakening [[Trench Keep-out]].

## Safety Boundary

Outputs should improve [[Perception 模块]] or [[Costmap 与 Risk Grid]]. They still need [[Safety Filter]] validation before affecting final motion.

## Source

- [RGB vision README](../../../../research/rgb_vision/README.md)
- [RGB pure-vision addendum](../../../../docs/project_evaluation/2026-06-22-rgb-vision-route-addendum.md)
- [RGB-only navigation config](../../../../configs/navigation/rgb_only.yaml)
- [Perception config](../../../../configs/navigation/perception.yaml)
