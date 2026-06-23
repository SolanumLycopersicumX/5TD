---
title: rgb_keyframes_2026-06-22
type: dataset
status: active
route: rgb-only
source:
  - data/annotation_batches/rgb_keyframes_2026-06-22/README.md
  - data/annotation_batches/rgb_keyframes_2026-06-22/labels.txt
  - data/annotation_batches/rgb_keyframes_2026-06-22/annotation_rules.md
created: 2026-06-23
updated: 2026-06-23
tags:
  - #data/annotation
---

# rgb_keyframes_2026-06-22

`rgb_keyframes_2026-06-22` 是当前共享的隧道 RGB 关键帧标注批次。

## 内容

- 批次 README。
- 标签列表。
- 标注规则。
- 145 张 JPEG 关键帧。
- 当前存在 10 个 JSON 标注。
- 无目标图像也是有效负样本，只要正确标注 mask 和硬边界。

## 作用

这个批次服务 [[RGB 标注与训练闭环]]，应作为 [[Perception 模块]] 的训练/评估材料。即使沟对面地面看起来平整，也不要标为 ego-passable。

## 来源

- [批次 README](../../../../data/annotation_batches/rgb_keyframes_2026-06-22/README.md)
- [标签列表](../../../../data/annotation_batches/rgb_keyframes_2026-06-22/labels.txt)
- [标注规则](../../../../data/annotation_batches/rgb_keyframes_2026-06-22/annotation_rules.md)
