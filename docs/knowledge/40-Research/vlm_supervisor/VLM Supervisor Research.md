---
title: VLM Supervisor Research
type: research
status: active
route: vlm
source:
  - research/vlm_supervisor/README.md
  - docs/project_evaluation/tunnel_ugv_rl_navigation_project_evaluation.md
  - docs/project_evaluation/tunnel_ugv_lidar_rgb_transformer_navigation_plan.md
created: 2026-06-23
updated: 2026-06-23
tags:
  - #route/vlm
  - #data/annotation
---

# VLM Supervisor Research

VLM Supervisor Research uses open-vocabulary and large-model support for annotation, risk explanation, and low-frequency supervision.

## Good Uses

- Help annotation and label review.
- Explain risk or failure cases.
- Provide low-frequency supervisory signals.

## Safety Boundary

VLM output is not a real-time motor-control command. It can support [[RGB 标注与训练闭环]], [[实验记录格式]], or supervisory review, but final motion still goes through [[Safety Filter]].

## Source

- [VLM supervisor README](../../../../research/vlm_supervisor/README.md)
- [RL navigation project evaluation](../../../../docs/project_evaluation/tunnel_ugv_rl_navigation_project_evaluation.md)
- [LiDAR-RGB Transformer project evaluation](../../../../docs/project_evaluation/tunnel_ugv_lidar_rgb_transformer_navigation_plan.md)
