# 5TD Obsidian Knowledge Base Design

Date: 2026-06-23

## Purpose

Create an Obsidian-style knowledge base for the 5TD Tunnel UGV Navigation repository under `docs/knowledge/`.

The knowledge base should help a maintainer understand the project as a connected system rather than as a flat file tree. It should make the current RGB-only engineering route, the safety-critical navigation pipeline, the research enhancement routes, and the data/annotation loop easy to enter and cross-reference.

## Background

The repository is a monorepo for tunnel UGV navigation. Its current practical track is the RGB-only HBD-Net-RT baseline under `baselines/hbdnet_rt/`. The enhanced research route lives under `research/` and covers RGB improvements, LiDAR-RGB Transformer fusion, RL navigation, diffusion trajectory proposals, and VLM supervision.

Important existing source material:

- `README.md`
- `docs/progress/LOG.md`
- `docs/project_evaluation/`
- `baselines/hbdnet_rt/README.md`
- `baselines/hbdnet_rt/docs/`
- `data/annotation_batches/rgb_keyframes_2026-06-22/`
- `research/*/README.md`
- `src/tunnel_nav/README.md`

The vault should summarize and connect these sources. It should not duplicate long source documents.

## Chosen Approach

Use a MOC-driven vault: short notes plus source-file links.

Compared with mirroring the repository directory tree, this better fits Obsidian because navigation starts from concepts and workflows. Compared with a research-first vault, this better matches the current project state because the active baseline and safety pipeline are more concrete than most research skeletons.

## Vault Location

Create the vault at:

```text
docs/knowledge/
```

The directory can be opened directly as an Obsidian vault.

## Directory Structure

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

## First-Version Scope

The first version should create enough notes for useful project navigation without pretending that unfinished research tracks are complete.

Include these core notes:

- `Home.md`
- MOC notes for project overview, RGB-only engineering route, safety/risk, data/annotation, research route, experiments/acceptance, and legacy references.
- Decision notes for the dual-route technical strategy and current priority.
- Architecture notes for the end-to-end data flow, safety-constrained navigation, costmap/risk-grid model, and baseline promotion into `src/tunnel_nav`.
- Baseline notes for HBD-Net-RT, perception, mapping, DWA, safety state machine, control command, latency budget, test matrix, and entry scripts.
- Research notes for RGB vision, Transformer fusion, RL navigation, diffusion planner, and VLM supervisor, kept concise and linked back to the source README files.
- Data notes for the RGB annotation/training loop and the current `rgb_keyframes_2026-06-22` annotation batch.
- Command notes for HBD-Net-RT quick start, common scripts, and git/LFS environment caveats.
- Glossary notes for hard boundary, trench keep-out, safety filter, semantic risk costmap, BEV, DWA, and MOC.

## Note Style

Each note should be short and link-rich:

- Start with minimal YAML frontmatter.
- Use `[[Wiki Links]]` for related vault notes.
- Use relative markdown links to source files when citing repository material.
- Prefer summaries, interfaces, and decisions over copied prose.
- Mark uncertain project facts as open questions instead of inventing detail.

## Frontmatter Convention

Use this minimal schema:

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

For `source`, use a list of relative source paths when a note is derived from existing files. For notes that are pure indexes, use an empty list.

## Tag Convention

Use tags only when they add filtering value:

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

## Content Boundaries

The knowledge base should not:

- Embed large assets, videos, model checkpoints, zip files, databases, or raw images.
- Copy long sections from existing evaluation reports or baseline docs.
- Treat `baselines/vision_obstacle_avoidance_legacy/` as the active engineering route.
- Present RL, diffusion, VLM, or Transformer modules as direct motor-control paths.
- Rewrite existing implementation docs in full.

## Safety-Critical Emphasis

The first version should make these safety constraints visible:

- Research modules may propose risk maps, semantic costmaps, candidate trajectories, or supervisory signals, but their outputs must pass planner and safety-filter validation.
- Hard-boundary and trench keep-out zones are non-crossable constraints.
- STOP and TAKEOVER states force zero speed and braking.
- RGB-only is the near-term MVP, but tunnel lighting, reflections, trench visibility, and sensor allowance remain validation questions.

## Verification Plan

After implementation:

1. Check all expected markdown files exist under `docs/knowledge/`.
2. Scan for unresolved placeholders such as `TODO`, `TBD`, or empty source sections.
3. Confirm internal wiki links point to notes that exist in the vault.
4. Confirm source-file markdown links are relative and point to existing repository files.
5. Confirm no large binary assets were copied into the vault.

## Open Questions

The first version should keep these as explicit open questions:

- What are the real post-civil tunnel lighting and reflection conditions?
- Is the right-side trench edge reliably visible with RGB-only perception?
- What minimum trench safety distance is required?
- Is a right-side ToF, LiDAR, or distance sensor allowed as an independent safety channel?
- What is the final ROS 2 and real chassis integration boundary?

## Implementation Notes

Generate the vault as markdown-only project documentation. Do not alter source code, data assets, legacy assets, model weights, videos, databases, or existing project documents except for any small intentional link updates requested later.
