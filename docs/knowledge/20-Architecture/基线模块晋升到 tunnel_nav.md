---
title: 基线模块晋升到 tunnel_nav
type: architecture
status: active
route: shared
source:
  - src/tunnel_nav/README.md
created: 2026-06-23
updated: 2026-06-23
tags:
  []
---

# 基线模块晋升到 tunnel_nav

`src/tunnel_nav/` 是未来集成工程包。只有接口、测试和安全行为稳定的模块才应晋升到这里。

## 候选职责

- 传感器适配。
- 标定。
- 感知。
- Mapping 和 costmap。
- 规划。
- 安全过滤。
- 控制命令接口。
- 评估工具。
- ROS 2 接口。

## 晋升规则

在接口稳定前，实验性工作应留在 `baselines/` 或 `research/`。只有具备清晰输入输出、验证命令和安全边界的模块才应进入稳定包。

## 相关

- [[HBD-Net-RT Baseline|HBD-Net-RT 基线]]
- [[双路线技术策略]]
- [[部署边界]]

## 来源

- [tunnel_nav README](../../../src/tunnel_nav/README.md)
