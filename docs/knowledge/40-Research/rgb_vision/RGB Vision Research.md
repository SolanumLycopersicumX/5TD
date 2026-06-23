---
title: RGB 视觉研究
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

# RGB 视觉研究

RGB 视觉研究面向后土建阶段隧道的纯 RGB 感知增强，覆盖可通行区域、硬边界、沟边、障碍物和人员检测。

## 重点

- 自由空间分割。
- [[Hard Boundary]] 和沟边检测。
- 障碍物、人员、工程车辆感知。
- 时序一致性和失败检测。
- 与 [[HBD-Net-RT Baseline|HBD-Net-RT 基线]] 对比。

## 验证触发条件

如果 RGB-only 无法可靠看到沟边或右边界，应补充右侧 ToF、LiDAR 或融合支持，而不是放松 [[Trench Keep-out]]。

## 安全边界

输出应改进 [[Perception 模块]] 或 [[Costmap 与 Risk Grid]]。影响最终运动前仍必须经过 [[Safety Filter]] 验证。

## 来源

- [RGB vision README](../../../../research/rgb_vision/README.md)
- [RGB 纯视觉路线补充说明](../../../../docs/project_evaluation/2026-06-22-rgb-vision-route-addendum.md)
- [RGB-only 导航配置](../../../../configs/navigation/rgb_only.yaml)
- [感知配置](../../../../configs/navigation/perception.yaml)
