---
title: VLM 监督研究
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

# VLM 监督研究

VLM 监督研究用开放词汇和大模型能力支持标注、风险解释和低频监督。

## 合适用途

- 辅助标注和标签复核。
- 解释风险或失败案例。
- 提供低频监督信号。

## 安全边界

VLM 输出不是实时电机控制命令。它可以支持 [[RGB 标注与训练闭环]]、[[实验记录格式]] 或监督复核，但最终运动仍必须通过 [[Safety Filter]]。

## 来源

- [VLM supervisor README](../../../../research/vlm_supervisor/README.md)
- [RL 导航项目评估](../../../../docs/project_evaluation/tunnel_ugv_rl_navigation_project_evaluation.md)
- [LiDAR-RGB Transformer 项目评估](../../../../docs/project_evaluation/tunnel_ugv_lidar_rgb_transformer_navigation_plan.md)
