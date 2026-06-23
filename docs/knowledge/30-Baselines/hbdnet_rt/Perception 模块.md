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

Perception 模块把 RGB 图像转换为检测、分割、边界、边缘、风险和置信度输出，供后续管线使用。

## 输入

- 按 HBD-Net-RT 模型尺寸整理后的 RGB 图像 tensor。

## 输出

- 工程车辆、工人、悬挂物、碎石/杂物等检测结果。
- ego-passable mask。
- hard-boundary mask 和 edge。
- surface risk map。
- 供 [[SafetyStateMachine]] 使用的置信度。

## 关键文件

- `perception/model.py`
- `perception/preprocessor.py`
- `perception/postprocess.py`
- `perception/inference.py`

## 相关

- [[HBD-Net-RT Baseline|HBD-Net-RT 基线]]
- [[Mapping 模块]]
- [[Hard Boundary]]

## 来源

- [模块接口](../../../../baselines/hbdnet_rt/docs/module_interfaces.md)
- [文件导读](../../../../baselines/hbdnet_rt/docs/FILE_GUIDE.md)
