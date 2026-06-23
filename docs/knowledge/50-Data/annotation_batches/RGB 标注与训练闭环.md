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

RGB 标注与训练闭环把隧道帧转成标签，用于训练和验证 RGB-only 感知基线。

## 闭环步骤

1. 抽取并筛选有代表性的隧道帧。
2. 标注自由空间、[[Hard Boundary]]、障碍物、工人、工程车辆和风险相关区域。
3. 训练或微调 [[Perception 模块]]。
4. 通过 [[Mapping 模块]]、[[DWAPlanner]] 和 [[SafetyStateMachine]] 评估输出。
5. 用 [[实验记录格式]] 记录结果，并与 [[验收指标]] 对比。

## 活跃批次

- [[rgb_keyframes_2026-06-22]]

## 来源

- [数据目录 README](../../../../data/README.md)
- [训练指南](../../../../baselines/hbdnet_rt/docs/training_guide.md)
- [标注规则](../../../../data/annotation_batches/rgb_keyframes_2026-06-22/annotation_rules.md)
