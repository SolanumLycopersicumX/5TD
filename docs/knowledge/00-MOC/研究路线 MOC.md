---
title: 研究路线 MOC
type: moc
status: active
route: shared
source:
  []
created: 2026-06-23
updated: 2026-06-23
tags:
  []
---

# 研究路线 MOC

这张地图追踪可增强基线能力的研究路线。所有研究输出都必须留在规划器和安全过滤之后接受验证。

## 研究方向

- [[RGB Vision Research|RGB 视觉研究]]
- [[Transformer Fusion Research|Transformer 融合研究]]
- [[RL Navigation Research|RL 导航研究]]
- [[Diffusion Planner Research|Diffusion 轨迹规划研究]]
- [[VLM Supervisor Research|VLM 监督研究]]

## 必须遵守的安全边界

研究模块可以输出感知结果、风险图、语义代价地图、候选轨迹、waypoint 或低频监督信号。它们不能绕过 [[DWAPlanner]]、[[SafetyStateMachine]] 或 [[Safety Filter]] 直接控制电机。

## 相关架构

- [[双路线技术策略]]
- [[Costmap 与 Risk Grid]]
- [[基线模块晋升到 tunnel_nav]]
