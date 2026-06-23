# 5TD Obsidian Knowledge Base Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a first-version Obsidian-style markdown knowledge vault for the 5TD repository under `docs/knowledge/`.

**Architecture:** The vault is MOC-driven: `Home.md` and `00-MOC/` provide navigation, while focused notes summarize project decisions, architecture, baselines, research tracks, data workflows, commands, and glossary concepts. Notes use YAML frontmatter, Obsidian wiki links, and relative markdown links back to source files.

**Tech Stack:** Markdown, Obsidian wiki links, YAML frontmatter, shell-based verification with `find`, `rg`, and `git`.

---

## File Map

Create:

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

Modify:

- No existing project files.

## Task 1: Write Red Verification Command

**Files:**
- Read: `docs/superpowers/specs/2026-06-23-5td-obsidian-knowledge-base-design.md`
- Create: no files
- Test: shell command only

- [ ] **Step 1: Run expected-file check before vault creation**

Run:

```bash
missing=0
for file in \
  "docs/knowledge/Home.md" \
  "docs/knowledge/00-MOC/5TD 项目总览 MOC.md" \
  "docs/knowledge/30-Baselines/hbdnet_rt/HBD-Net-RT Baseline.md" \
  "docs/knowledge/40-Research/transformer_fusion/Transformer Fusion Research.md" \
  "docs/knowledge/50-Data/annotation_batches/RGB 标注与训练闭环.md" \
  "docs/knowledge/80-Commands/HBD-Net-RT 快速开始.md" \
  "docs/knowledge/90-Glossary/Safety Filter.md"
do
  test -f "$file" || { echo "missing: $file"; missing=1; }
done
exit "$missing"
```

Expected before implementation: exit code `1` with missing file lines.

- [ ] **Step 2: Record red check result in the implementation notes**

Expected note: the command fails because the vault has not been created yet.

## Task 2: Create Home and MOC Layer

**Files:**
- Create: `docs/knowledge/Home.md`
- Create: all files under `docs/knowledge/00-MOC/`

- [ ] **Step 1: Create `Home.md`**

Include links to all MOC files and core notes:

```markdown
[[5TD 项目总览 MOC]]
[[RGB-only 工程路线 MOC]]
[[安全与风险 MOC]]
[[数据与标注 MOC]]
[[研究路线 MOC]]
[[实验与验收 MOC]]
[[Legacy 参考 MOC]]
```

- [ ] **Step 2: Create MOC files**

Each MOC must include frontmatter with `type: moc`, `status: active`, and `route: shared`, then short link groups to the notes in its domain.

- [ ] **Step 3: Verify MOC files exist**

Run:

```bash
find docs/knowledge/00-MOC -maxdepth 1 -type f -name '*.md' | sort
```

Expected: seven MOC files listed.

## Task 3: Create Decision and Architecture Notes

**Files:**
- Create: `docs/knowledge/10-Decisions/当前优先级与路线决策.md`
- Create: `docs/knowledge/10-Decisions/双路线技术策略.md`
- Create: all files under `docs/knowledge/20-Architecture/`

- [ ] **Step 1: Create decision notes**

Summarize the current RGB-only MVP priority, LiDAR-RGB enhancement path, and safety-filter rule. Link to:

```markdown
[README.md](../../../README.md)
[RGB Pure-Vision Route Addendum](../../../docs/project_evaluation/2026-06-22-rgb-vision-route-addendum.md)
[Progress Log](../../../docs/progress/LOG.md)
```

- [ ] **Step 2: Create architecture notes**

Capture the pipeline:

```text
RGB frame -> Perception -> OccupancyGrid/RiskGrid -> DWAPlanner -> SafetyStateMachine -> ControlCommand
```

Link each stage with wiki links to the baseline notes.

- [ ] **Step 3: Verify core architecture links**

Run:

```bash
rg -n "\[\[(Perception 模块|Mapping 模块|DWAPlanner|SafetyStateMachine|ControlCommand)\]\]" docs/knowledge/20-Architecture docs/knowledge/10-Decisions
```

Expected: each listed module appears at least once.

## Task 4: Create Baseline and Command Notes

**Files:**
- Create: all files under `docs/knowledge/30-Baselines/hbdnet_rt/`
- Create: `docs/knowledge/30-Baselines/legacy_vision/Legacy Pure Vision Baseline.md`
- Create: all files under `docs/knowledge/80-Commands/`

- [ ] **Step 1: Create HBD-Net-RT baseline notes**

Use `baselines/hbdnet_rt/README.md`, `baselines/hbdnet_rt/docs/module_interfaces.md`, `baselines/hbdnet_rt/docs/FILE_GUIDE.md`, `baselines/hbdnet_rt/docs/latency_budget.md`, and `baselines/hbdnet_rt/docs/engineering_notes.md` as sources.

- [ ] **Step 2: Create command notes**

Include commands exactly as repository docs describe:

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

- [ ] **Step 3: Verify command notes cite runnable paths**

Run:

```bash
rg -n "scripts/run_pipeline.py|scripts/run_dashboard.py|pytest tests" docs/knowledge/80-Commands docs/knowledge/30-Baselines/hbdnet_rt
```

Expected: command references appear in command or baseline notes.

## Task 5: Create Research, Data, Experiment, Deployment, and Glossary Notes

**Files:**
- Create: all files under `docs/knowledge/40-Research/`
- Create: all files under `docs/knowledge/50-Data/`
- Create: all files under `docs/knowledge/60-Experiments/`
- Create: all files under `docs/knowledge/70-Deployment/`
- Create: all files under `docs/knowledge/90-Glossary/`

- [ ] **Step 1: Create research notes**

Keep each research note concise. State that research outputs must pass through planner and safety-filter validation.

- [ ] **Step 2: Create data and asset notes**

Link the current annotation batch and state that large assets stay in Git LFS, not inside the vault.

- [ ] **Step 3: Create experiment and deployment notes**

Summarize the expected experiment report fields and deployment boundary from existing README files.

- [ ] **Step 4: Create glossary notes**

Each glossary note must define the term, link to at least one related architecture or baseline note, and cite source files where applicable.

## Task 6: Full Verification and Commit

**Files:**
- Read: `docs/knowledge/**/*.md`
- Modify: no existing project files

- [ ] **Step 1: Run expected-file verification**

Run the command from Task 1 again.

Expected after implementation: exit code `0` and no missing file lines.

- [ ] **Step 2: Verify no empty notes**

Run:

```bash
find docs/knowledge -name '*.md' -type f -size 0 -print
```

Expected: no output.

- [ ] **Step 3: Verify source links use relative paths**

Run:

```bash
rg -n "file://|/home/jiaming|https://github.com" docs/knowledge
```

Expected: no output.

- [ ] **Step 4: Verify no binary assets in vault**

Run:

```bash
find docs/knowledge -type f ! -name '*.md' -print
```

Expected: no output.

- [ ] **Step 5: Verify wiki links resolve by note basename**

Run:

```bash
tmp_notes="$(mktemp)"
tmp_links="$(mktemp)"
find docs/knowledge -name '*.md' -type f -printf '%f\n' | sed 's/\.md$//' | sort -u > "$tmp_notes"
rg -o "\[\[[^]]+\]\]" docs/knowledge \
  | sed 's/^.*\[\[//; s/\]\]$//; s/|.*$//; s/#.*$//' \
  | sort -u > "$tmp_links"
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

Expected: no unresolved wiki links.

- [ ] **Step 6: Review git diff**

Run:

```bash
git -c filter.lfs.process= -c filter.lfs.required=false status --short
git -c filter.lfs.process= -c filter.lfs.required=false diff -- docs/knowledge docs/superpowers/plans/2026-06-23-5td-obsidian-knowledge-base.md
```

Expected: only the new plan and new vault files are part of this implementation, aside from pre-existing unrelated user changes shown by status.

- [ ] **Step 7: Commit the knowledge base**

Run:

```bash
git -c filter.lfs.process= -c filter.lfs.required=false add docs/knowledge docs/superpowers/plans/2026-06-23-5td-obsidian-knowledge-base.md
git -c filter.lfs.process= -c filter.lfs.required=false commit -m "Add Obsidian knowledge vault"
```
