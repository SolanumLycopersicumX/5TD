# 5TD Directory Reorganization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize the current 5TD workspace into the approved dual-route repository layout and include all project assets in GitHub using Git LFS for large binaries.

**Architecture:** Preserve RGB-only engineering code as the active baseline, preserve the older pure-vision project with its data/model/media assets, and add research/shared skeletons for RGB vision, LiDAR-RGB Transformer fusion, RL, diffusion, and VLM work. Use Git LFS for zip/model/video/database artifacts that would otherwise exceed GitHub limits.

**Tech Stack:** Git, Git LFS, Markdown, YAML, Python project layout, HBD-Net-RT pytest baseline.

---

### Task 1: Enable LFS for Full Asset Upload

**Files:**
- Create: `.gitattributes`

- [ ] Track archives, model weights, videos, databases, dataset arrays, and generated binary artifacts with Git LFS.
- [ ] Verify `git lfs version` works using `/tmp/git-lfs-3.7.1/git-lfs`.

### Task 2: Move All Existing Project Assets

**Files:**
- Move: `vision_obstacle_avoidance.zip` to `archive/original/vision_obstacle_avoidance.zip`
- Move: extracted `hbdnet_rt/` to `baselines/hbdnet_rt/`
- Move: all remaining extracted project assets to `baselines/vision_obstacle_avoidance_legacy/`

- [ ] Move HBD-Net-RT without changing its internal layout.
- [ ] Move older pure-vision source, docs, tools, data, datasets, models, videos, app, hidden metadata, logs, and caches into the legacy baseline tree.
- [ ] Rename nested old `.git` metadata to `original_metadata/git/` so it can be tracked as ordinary files instead of being treated as a nested repository.

### Task 3: Add Repository Skeleton

**Files:**
- Create: `README.md`
- Create: `data/README.md`
- Create: `experiments/README.md`
- Create: `research/**/README.md`
- Create: `src/tunnel_nav/README.md`
- Create: `configs/**/*.yaml`
- Create: `deployment/README.md`
- Create: `tools/README.md`
- Create: `docs/architecture/README.md`
- Create: `docs/legacy/old_code_notes.md`

- [ ] Document RGB-only as the near-term MVP route.
- [ ] Document LiDAR-RGB/Transformer as the enhancement route.
- [ ] Add conservative config skeletons without implementation code.

### Task 4: Verify and Commit

**Files:**
- Modify: `docs/progress/LOG.md`

- [ ] Append a progress entry explaining the full-asset upload change.
- [ ] Run `git status --short`.
- [ ] Run `git diff --check` and `git diff --cached --check`.
- [ ] Run HBD-Net-RT tests with `PYTHONPATH=src pytest tests -q` if dependencies are available.
- [ ] Commit with message `Reorganize project with full asset tracking`.
