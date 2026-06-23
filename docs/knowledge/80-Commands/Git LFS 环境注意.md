---
title: Git LFS 环境注意
type: command
status: active
route: shared
source:
  - README.md
  - docs/progress/LOG.md
created: 2026-06-23
updated: 2026-06-23
tags:
  - #data/asset
---

# Git LFS 环境注意

仓库使用 Git LFS 跟踪大型资产。本地机器需要安装 Git LFS 才能正常处理大型压缩包、视频、模型权重和数据库。

## 命令

```bash
git lfs install
git lfs pull
```

## 本机注意

如果本机没有 `git-lfs`，普通 `git status` 可能失败。只读查看状态时可以使用：

```bash
git -c filter.lfs.process= -c filter.lfs.required=false status --short
```

## 相关

- [[大资产与 Git LFS]]
- [[Legacy Pure Vision Baseline|Legacy 纯视觉基线]]

## 来源

- [仓库 README](../../../README.md)
- [项目进展日志](../../../docs/progress/LOG.md)
