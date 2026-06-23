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

Quick start for the active RGB-only baseline.

```bash
pip install torch numpy opencv-python pyyaml pytest
cd baselines/hbdnet_rt
export PYTHONPATH=src
```

## Smoke Commands

```bash
python scripts/run_pipeline.py -n 50
python scripts/run_dashboard.py
python scripts/run_inference.py
python scripts/run_planner_demo.py
python scripts/benchmark_latency.py -n 200
pytest tests/ -v
```

## Related

- [[HBD-Net-RT Baseline]]
- [[常用脚本命令]]
- [[测试矩阵]]

## Source

- [HBD-Net-RT README](../../../baselines/hbdnet_rt/README.md)
