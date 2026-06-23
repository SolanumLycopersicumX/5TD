# 5TD Obsidian 知识库设计

日期：2026-06-23

## 目标

在 `docs/knowledge/` 下创建一个 Obsidian 风格知识库，用于帮助维护者按系统关系理解 5TD Tunnel UGV Navigation 仓库。

知识库应突出当前 RGB-only 工程路线、安全关键导航管线、增强研究路线，以及数据/标注闭环。它不是现有 README 的复制品，而是面向 Obsidian 双链的项目知识入口。

## 背景

该仓库是隧道 UGV 导航 monorepo。当前可落地路线是 `baselines/hbdnet_rt/` 下的 RGB-only HBD-Net-RT 基线。增强研究路线位于 `research/`，包含 RGB 改进、LiDAR-RGB Transformer 融合、RL 导航、diffusion 轨迹 proposal 和 VLM 监督。

重要源材料：

- `README.md`
- `docs/progress/LOG.md`
- `docs/project_evaluation/`
- `baselines/hbdnet_rt/README.md`
- `baselines/hbdnet_rt/docs/`
- `data/annotation_batches/rgb_keyframes_2026-06-22/`
- `research/*/README.md`
- `src/tunnel_nav/README.md`

知识库应总结并连接这些源材料，不复制长篇原文。

## 选定方案

采用 MOC 驱动的 vault：短笔记加源文件链接。

相较于镜像仓库目录，这种方式更适合 Obsidian，因为入口是概念和工作流。相较于研究优先的 vault，这种方式更符合项目现状，因为活跃基线和安全管线比多数研究骨架更具体。

## Vault 位置

创建位置：

```text
docs/knowledge/
```

这个目录可以直接作为 Obsidian vault 打开。

## 目录结构

```text
docs/knowledge/
  Home.md
  00-MOC/
    5TD 项目总览 MOC.md
    RGB-only 工程路线 MOC.md
    安全与风险 MOC.md
    数据与标注 MOC.md
    研究路线 MOC.md
    实验与验收 MOC.md
    Legacy 参考 MOC.md
  10-Decisions/
  20-Architecture/
  30-Baselines/
    hbdnet_rt/
    legacy_vision/
  40-Research/
    rgb_vision/
    transformer_fusion/
    rl_navigation/
    diffusion_planner/
    vlm_supervisor/
  50-Data/
    annotation_batches/
    assets/
  60-Experiments/
  70-Deployment/
  80-Commands/
  90-Glossary/
```

## 第一版范围

第一版应提供足够的项目导航价值，同时不把尚未完成的研究 track 写成已经完成的系统。

需要包含：

- `Home.md`。
- 项目总览、RGB-only 工程路线、安全/风险、数据/标注、研究路线、实验/验收和 legacy 参考 MOC。
- 双路线技术策略和当前优先级决策笔记。
- 端到端数据流、安全约束导航、costmap/risk-grid 模型、基线模块晋升到 `src/tunnel_nav` 的架构笔记。
- HBD-Net-RT、perception、mapping、DWA、safety state machine、control command、延迟预算、测试矩阵和入口脚本笔记。
- RGB vision、Transformer fusion、RL navigation、diffusion planner、VLM supervisor 研究笔记；内容保持简洁并链接回源 README。
- RGB 标注/训练闭环和当前 `rgb_keyframes_2026-06-22` 批次笔记。
- HBD-Net-RT 快速开始、常用脚本、Git LFS 环境注意笔记。
- hard boundary、trench keep-out、safety filter、semantic risk costmap、BEV、DWA、MOC 术语笔记。

## 笔记风格

每篇笔记应短、可链接、可追溯：

- 使用最小 YAML frontmatter。
- 使用 `[[Wiki Links]]` 连接相关 vault 笔记。
- 引用仓库材料时使用相对 Markdown 链接。
- 优先写摘要、接口和决策，不复制已有文档长段落。
- 不确定的项目事实写成待验证问题，不编造细节。
- 正文默认使用中文；技术名词、类名、命令、路径和 schema 值按原文保留。

## 元数据约定

使用最小元数据 schema：

```yaml
---
title:
type: moc | decision | architecture | baseline | interface | research | experiment | dataset | command | legacy | glossary
status: draft | active | validated | archived
route: rgb-only | fusion | rl | diffusion | vlm | legacy | shared
source:
created: 2026-06-23
updated: 2026-06-23
tags:
---
```

`source` 在笔记来自现有文件时填写相对源路径列表。纯索引笔记可使用空列表。

## 标签约定

标签只在有过滤价值时使用：

- `#route/rgb-only`
- `#route/fusion`
- `#route/rl`
- `#route/diffusion`
- `#route/vlm`
- `#module/perception`
- `#module/mapping`
- `#module/planning`
- `#module/safety`
- `#module/control`
- `#data/annotation`
- `#data/asset`
- `#status/draft`
- `#status/active`
- `#status/validated`
- `#status/archived`
- `#risk/safety-critical`
- `#legacy/reference`

## 内容边界

知识库不应：

- 嵌入大型资产、视频、模型 checkpoint、zip、数据库或原始图片。
- 从现有评估报告或 baseline 文档复制长段落。
- 把 `baselines/vision_obstacle_avoidance_legacy/` 当作活跃工程路线。
- 把 RL、diffusion、VLM 或 Transformer 模块写成可直接控制电机的路径。
- 完整重写已有实现文档。

## 安全重点

第一版必须突出这些安全约束：

- 研究模块可以提出风险图、语义 costmap、候选轨迹或监督信号，但输出必须经过 planner 和 safety filter 验证。
- Hard-boundary 和 trench keep-out 是不可跨越约束。
- STOP 和 TAKEOVER 状态强制零速和刹车。
- RGB-only 是近期 MVP，但隧道光照、反光、沟边可见性和传感器许可仍是验证问题。

## 验证计划

实施后检查：

1. `docs/knowledge/` 下所有预期 Markdown 文件存在。
2. 扫描未完成占位符和空 source section。
3. 确认内部 wiki links 指向 vault 中存在的笔记。
4. 确认源文件 Markdown 链接为相对路径，并指向仓库中存在的文件。
5. 确认没有大型二进制资产进入 vault。
6. 确认正文以中文为主，只保留必要技术名词、命令、路径和 schema 值。

## 待确认问题

第一版应显式保留这些待确认问题：

- 后土建阶段真实隧道光照和反光条件是什么？
- 右侧沟边是否能被 RGB-only 感知稳定看清？
- 项目要求的最小沟边安全距离是多少？
- 是否允许右侧 ToF、LiDAR 或测距传感器作为独立安全通道？
- 最终 ROS 2 和真实底盘集成边界是什么？

## 实施说明

生成 markdown-only 项目文档。除非后续明确要求，不修改源码、数据资产、legacy 资产、模型权重、视频、数据库或既有项目文档。
