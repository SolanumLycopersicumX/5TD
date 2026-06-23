---
title: Mapping 模块
type: interface
status: active
route: rgb-only
source:
  - baselines/hbdnet_rt/docs/module_interfaces.md
  - baselines/hbdnet_rt/docs/FILE_GUIDE.md
created: 2026-06-23
updated: 2026-06-23
tags:
  - #module/mapping
---

# Mapping 模块

Mapping 模块把感知输出投影为地面坐标系下的占用栅格和风险栅格。

## 职责

- 使用标定和 BEV 投影把图像 mask 映射到栅格坐标。
- 根据硬边界、不可通行区域和检测区域生成占用。
- 为 [[DWAPlanner]] 和 [[SafetyStateMachine]] 生成连续风险。

## 关键文件

- `mapping/calibration.py`
- `mapping/bev_projector.py`
- `mapping/occupancy_grid.py`
- `mapping/risk_grid.py`

## 相关

- [[BEV]]
- [[Costmap 与 Risk Grid]]
- [[Semantic Risk Costmap]]

## 来源

- [模块接口](../../../../baselines/hbdnet_rt/docs/module_interfaces.md)
- [文件导读](../../../../baselines/hbdnet_rt/docs/FILE_GUIDE.md)
