---
title: HBD-Net-RT 快速开始
type: command
status: active
route: rgb-only
source:
  - baselines/hbdnet_rt/README.md
created: 2026-06-23
updated: 2026-06-23
tags:
  - #route/rgb-only
---

# HBD-Net-RT 快速开始

活跃 RGB-only 基线的快速开始命令。

```bash
pip install torch numpy opencv-python pyyaml pytest
cd baselines/hbdnet_rt
export PYTHONPATH=src
```

## 冒烟命令

```bash
python scripts/run_pipeline.py -n 50
python scripts/run_dashboard.py
python scripts/run_inference.py
python scripts/run_planner_demo.py
python scripts/benchmark_latency.py -n 200
pytest tests/ -v
```

## 相关

- [[HBD-Net-RT Baseline|HBD-Net-RT 基线]]
- [[常用脚本命令]]
- [[测试矩阵]]

## 来源

- [HBD-Net-RT README](../../../baselines/hbdnet_rt/README.md)
