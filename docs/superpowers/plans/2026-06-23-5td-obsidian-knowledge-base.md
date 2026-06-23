# 5TD Obsidian 知识库实施计划

> **给 agentic workers：** 必须使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 按任务执行本计划。步骤使用 checkbox（`- [ ]`）语法便于追踪。

**目标：** 在 `docs/knowledge/` 下构建第一版 Obsidian 风格 Markdown 知识库。

**架构：** 采用 MOC 驱动结构：`Home.md` 和 `00-MOC/` 提供导航，领域笔记总结项目决策、架构、基线、研究路线、数据工作流、命令和术语。笔记使用 YAML frontmatter、Obsidian wiki links，并用相对路径链接回源文件。

**技术栈：** Markdown、Obsidian wiki links、YAML frontmatter、基于 `find`、`rg`、`git` 的 shell 验证。

---

## 文件地图

创建：

- `docs/knowledge/Home.md`
- `docs/knowledge/00-MOC/5TD 项目总览 MOC.md`
- `docs/knowledge/00-MOC/RGB-only 工程路线 MOC.md`
- `docs/knowledge/00-MOC/安全与风险 MOC.md`
- `docs/knowledge/00-MOC/数据与标注 MOC.md`
- `docs/knowledge/00-MOC/研究路线 MOC.md`
- `docs/knowledge/00-MOC/实验与验收 MOC.md`
- `docs/knowledge/00-MOC/Legacy 参考 MOC.md`
- `docs/knowledge/10-Decisions/当前优先级与路线决策.md`
- `docs/knowledge/10-Decisions/双路线技术策略.md`
- `docs/knowledge/20-Architecture/端到端运行流程.md`
- `docs/knowledge/20-Architecture/安全约束导航.md`
- `docs/knowledge/20-Architecture/Costmap 与 Risk Grid.md`
- `docs/knowledge/20-Architecture/基线模块晋升到 tunnel_nav.md`
- `docs/knowledge/30-Baselines/hbdnet_rt/HBD-Net-RT Baseline.md`
- `docs/knowledge/30-Baselines/hbdnet_rt/Perception 模块.md`
- `docs/knowledge/30-Baselines/hbdnet_rt/Mapping 模块.md`
- `docs/knowledge/30-Baselines/hbdnet_rt/DWAPlanner.md`
- `docs/knowledge/30-Baselines/hbdnet_rt/SafetyStateMachine.md`
- `docs/knowledge/30-Baselines/hbdnet_rt/ControlCommand.md`
- `docs/knowledge/30-Baselines/hbdnet_rt/入口脚本索引.md`
- `docs/knowledge/30-Baselines/hbdnet_rt/测试矩阵.md`
- `docs/knowledge/30-Baselines/hbdnet_rt/延迟基准.md`
- `docs/knowledge/30-Baselines/legacy_vision/Legacy Pure Vision Baseline.md`
- `docs/knowledge/40-Research/rgb_vision/RGB Vision Research.md`
- `docs/knowledge/40-Research/transformer_fusion/Transformer Fusion Research.md`
- `docs/knowledge/40-Research/rl_navigation/RL Navigation Research.md`
- `docs/knowledge/40-Research/diffusion_planner/Diffusion Planner Research.md`
- `docs/knowledge/40-Research/vlm_supervisor/VLM Supervisor Research.md`
- `docs/knowledge/50-Data/annotation_batches/RGB 标注与训练闭环.md`
- `docs/knowledge/50-Data/annotation_batches/rgb_keyframes_2026-06-22.md`
- `docs/knowledge/50-Data/assets/大资产与 Git LFS.md`
- `docs/knowledge/60-Experiments/实验记录格式.md`
- `docs/knowledge/60-Experiments/验收指标.md`
- `docs/knowledge/70-Deployment/部署边界.md`
- `docs/knowledge/80-Commands/HBD-Net-RT 快速开始.md`
- `docs/knowledge/80-Commands/常用脚本命令.md`
- `docs/knowledge/80-Commands/Git LFS 环境注意.md`
- `docs/knowledge/90-Glossary/BEV.md`
- `docs/knowledge/90-Glossary/DWA.md`
- `docs/knowledge/90-Glossary/Hard Boundary.md`
- `docs/knowledge/90-Glossary/Trench Keep-out.md`
- `docs/knowledge/90-Glossary/Safety Filter.md`
- `docs/knowledge/90-Glossary/Semantic Risk Costmap.md`
- `docs/knowledge/90-Glossary/MOC.md`

修改：

- 不修改既有项目文件。

## 任务 1：先写红色验证命令

**文件：**
- 读取：`docs/superpowers/specs/2026-06-23-5td-obsidian-knowledge-base-design.md`
- 创建：无
- 测试：仅 shell 命令

- [ ] **步骤 1：创建 vault 前运行预期文件检查**

运行：

```bash
missing=0
for file in   "docs/knowledge/Home.md"   "docs/knowledge/00-MOC/5TD 项目总览 MOC.md"   "docs/knowledge/30-Baselines/hbdnet_rt/HBD-Net-RT Baseline.md"   "docs/knowledge/40-Research/transformer_fusion/Transformer Fusion Research.md"   "docs/knowledge/50-Data/annotation_batches/RGB 标注与训练闭环.md"   "docs/knowledge/80-Commands/HBD-Net-RT 快速开始.md"   "docs/knowledge/90-Glossary/Safety Filter.md"
do
  test -f "$file" || { echo "missing: $file"; missing=1; }
done
exit "$missing"
```

实施前预期：退出码 `1`，并输出缺失文件。

- [ ] **步骤 2：在实施记录中记录红色检查结果**

预期记录：命令失败是因为 vault 尚未创建。

## 任务 2：创建首页和 MOC 层

**文件：**
- 创建：`docs/knowledge/Home.md`
- 创建：`docs/knowledge/00-MOC/` 下所有文件

- [ ] **步骤 1：创建 `Home.md`**

包含所有 MOC 和核心笔记链接：

```markdown
[[5TD 项目总览 MOC]]
[[RGB-only 工程路线 MOC]]
[[安全与风险 MOC]]
[[数据与标注 MOC]]
[[研究路线 MOC]]
[[实验与验收 MOC]]
[[Legacy 参考 MOC]]
```

- [ ] **步骤 2：创建 MOC 文件**

每个 MOC 都应包含 `type: moc`、`status: active`、`route: shared` 的 frontmatter，并用简短链接组连接领域笔记。

- [ ] **步骤 3：验证 MOC 文件存在**

运行：

```bash
find docs/knowledge/00-MOC -maxdepth 1 -type f -name '*.md' | sort
```

预期：列出 7 个 MOC 文件。

## 任务 3：创建决策和架构笔记

**文件：**
- 创建：`docs/knowledge/10-Decisions/当前优先级与路线决策.md`
- 创建：`docs/knowledge/10-Decisions/双路线技术策略.md`
- 创建：`docs/knowledge/20-Architecture/` 下所有文件

- [ ] **步骤 1：创建决策笔记**

总结当前 RGB-only MVP 优先级、LiDAR-RGB 增强路线和 safety-filter 规则。链接到：

```markdown
[README.md](../../../README.md)
[RGB 纯视觉路线补充说明](../../../docs/project_evaluation/2026-06-22-rgb-vision-route-addendum.md)
[项目进展日志](../../../docs/progress/LOG.md)
```

- [ ] **步骤 2：创建架构笔记**

记录管线：

```text
RGB frame -> Perception -> OccupancyGrid/RiskGrid -> DWAPlanner -> SafetyStateMachine -> ControlCommand
```

每个阶段都用 wiki link 指向对应 baseline 笔记。

- [ ] **步骤 3：验证核心架构链接**

运行：

```bash
rg -n "\[\[(Perception 模块|Mapping 模块|DWAPlanner|SafetyStateMachine|ControlCommand)" docs/knowledge/20-Architecture docs/knowledge/10-Decisions
```

预期：每个核心模块至少出现一次。

## 任务 4：创建基线和命令笔记

**文件：**
- 创建：`docs/knowledge/30-Baselines/hbdnet_rt/` 下所有文件
- 创建：`docs/knowledge/30-Baselines/legacy_vision/Legacy Pure Vision Baseline.md`
- 创建：`docs/knowledge/80-Commands/` 下所有文件

- [ ] **步骤 1：创建 HBD-Net-RT 基线笔记**

使用 `baselines/hbdnet_rt/README.md`、`baselines/hbdnet_rt/docs/module_interfaces.md`、`baselines/hbdnet_rt/docs/FILE_GUIDE.md`、`baselines/hbdnet_rt/docs/latency_budget.md` 和 `baselines/hbdnet_rt/docs/engineering_notes.md` 作为来源。

- [ ] **步骤 2：创建命令笔记**

保留仓库文档中的命令：

```bash
cd baselines/hbdnet_rt
export PYTHONPATH=src
python scripts/run_pipeline.py -n 50
python scripts/run_dashboard.py
python scripts/run_inference.py
python scripts/run_planner_demo.py
python scripts/benchmark_latency.py -n 200
pytest tests/ -v
```

- [ ] **步骤 3：验证命令笔记引用可运行路径**

运行：

```bash
rg -n "scripts/run_pipeline.py|scripts/run_dashboard.py|pytest tests" docs/knowledge/80-Commands docs/knowledge/30-Baselines/hbdnet_rt
```

预期：命令或 baseline 笔记中出现相关引用。

## 任务 5：创建研究、数据、实验、部署和术语笔记

**文件：**
- 创建：`docs/knowledge/40-Research/` 下所有文件
- 创建：`docs/knowledge/50-Data/` 下所有文件
- 创建：`docs/knowledge/60-Experiments/` 下所有文件
- 创建：`docs/knowledge/70-Deployment/` 下所有文件
- 创建：`docs/knowledge/90-Glossary/` 下所有文件

- [ ] **步骤 1：创建研究笔记**

研究笔记保持简洁，并明确研究输出必须经过 planner 和 safety filter 验证。

- [ ] **步骤 2：创建数据和资产笔记**

链接当前标注批次，并说明大型资产留在 Git LFS，不进入 vault。

- [ ] **步骤 3：创建实验和部署笔记**

从现有 README 总结实验报告字段和部署边界。

- [ ] **步骤 4：创建术语笔记**

每个术语笔记都应定义概念，链接至少一个相关架构或 baseline 笔记，并在适用时引用源文件。

## 任务 6：中文化修正

**文件：**
- 修改：`docs/knowledge/**/*.md`
- 修改：本计划文件
- 修改：设计文档

- [ ] **步骤 1：把正文改为中文**

正文、说明性标题、决策说明、验证说明统一使用中文。技术名词、命令、路径、类名、frontmatter 枚举值保持原文。

- [ ] **步骤 2：验证英文式正文已减少到技术名词范围**

运行中文化检查脚本，确认不再出现明显英文说明句。

## 任务 7：完整验证和提交

**文件：**
- 读取：`docs/knowledge/**/*.md`
- 修改：不修改既有项目文件

- [ ] **步骤 1：运行预期文件验证**

再次运行任务 1 的命令。

实施后预期：退出码 `0`，没有缺失文件输出。

- [ ] **步骤 2：验证没有空笔记**

运行：

```bash
find docs/knowledge -name '*.md' -type f -size 0 -print
```

预期：无输出。

- [ ] **步骤 3：验证 source links 使用相对路径**

运行：

```bash
rg -n "file://|/home/jiaming|https://github.com" docs/knowledge
```

预期：无输出。

- [ ] **步骤 4：验证 vault 中没有二进制资产**

运行：

```bash
find docs/knowledge -type f ! -name '*.md' -print
```

预期：无输出。

- [ ] **步骤 5：验证 wiki links 可按笔记 basename 解析**

运行：

```bash
tmp_notes="$(mktemp)"
tmp_links="$(mktemp)"
find docs/knowledge -name '*.md' -type f -printf '%f
' | sed 's/\.md$//' | sort -u > "$tmp_notes"
rg -o "\[\[[^]]+\]\]" docs/knowledge   | sed 's/^.*\[\[//; s/\]\]$//; s/|.*$//; s/#.*$//'   | sort -u > "$tmp_links"
missing=0
while IFS= read -r link
do
  test -z "$link" && continue
  if ! grep -Fxq "$link" "$tmp_notes"; then
    echo "unresolved wiki link: $link"
    missing=1
  fi
done < "$tmp_links"
rm -f "$tmp_notes" "$tmp_links"
exit "$missing"
```

预期：没有 unresolved wiki link。

- [ ] **步骤 6：检查 git diff**

运行：

```bash
git -c filter.lfs.process= -c filter.lfs.required=false status --short
git -c filter.lfs.process= -c filter.lfs.required=false diff -- docs/knowledge docs/superpowers/plans/2026-06-23-5td-obsidian-knowledge-base.md docs/superpowers/specs/2026-06-23-5td-obsidian-knowledge-base-design.md
```

预期：只包含知识库、计划和设计文档的中文化变更；预先存在的无关未跟踪文件不纳入提交。

- [ ] **步骤 7：提交中文化修正**

运行：

```bash
git -c filter.lfs.process= -c filter.lfs.required=false add docs/knowledge docs/superpowers/plans/2026-06-23-5td-obsidian-knowledge-base.md docs/superpowers/specs/2026-06-23-5td-obsidian-knowledge-base-design.md
git -c filter.lfs.process= -c filter.lfs.required=false commit -m "Localize knowledge vault docs to Chinese"
```
