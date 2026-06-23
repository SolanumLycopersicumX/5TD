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

The repository tracks large assets with Git LFS. Local machines need Git LFS installed to handle large archives, videos, model checkpoints, and databases normally.

## Commands

```bash
git lfs install
git lfs pull
```

## Local Caveat

On this machine, plain `git status` may fail when `git-lfs` is unavailable. A read-only workaround for status is:

```bash
git -c filter.lfs.process= -c filter.lfs.required=false status --short
```

## Related

- [[大资产与 Git LFS]]
- [[Legacy Pure Vision Baseline]]

## Source

- [Repository README](../../../README.md)
- [Progress log](../../../docs/progress/LOG.md)
