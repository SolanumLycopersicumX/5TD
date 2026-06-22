# 5TD Progress Log

This log records project-level progress for repository structure, baseline preservation, research work, verification, and deployment milestones.

## 2026-06-22

Scope:

- Initialized `/home/tomato/5TD` as a local Git repository on branch `main`.
- Connected `origin` to `https://github.com/SolanumLycopersicumX/5TD.git` without embedding credentials.
- Reviewed the project evaluation report and the extracted previous code layout.
- Identified two previous-code baselines: `hbdnet_rt` and the older `vision_obstacle_avoidance` stack.
- Drafted the repository-structure design for preserving old code while adding RL, diffusion, and VLM research tracks.

Decisions:

- Use a monorepo with separate `baselines/`, `research/`, `src/tunnel_nav/`, `configs/`, `data/`, `experiments/`, `deployment/`, and `tools/` areas.
- Keep HBD-Net-RT as the primary runnable engineering baseline.
- Keep the older pure-vision stack as a legacy baseline and comparison reference.
- Keep large archives, raw datasets, model weights, generated experiment runs, and credentials out of Git.
- Maintain this file as the project progress log.

Verification:

- Confirmed local Git branch is `main`.
- Confirmed `origin` remote does not contain the provided GitHub token.
- Confirmed the workspace had no commits before the first documentation commit and the previous code was still unmodified.

Next Actions:

- Review and approve the repository-structure spec.
- Add `.gitignore`, root `README.md`, and directory README files.
- Move the project evaluation report and previous-code baselines into the approved structure.
- Run baseline tests after moving files and record results here.


## 2026-06-22 - LiDAR-RGB-Transformer Plan Update

Scope:

- Received the updated project plan `tunnel_ugv_lidar_rgb_transformer_navigation_plan.md`.
- Moved the updated plan and the earlier RL-navigation evaluation into `docs/project_evaluation/`.
- Updated the repository-structure spec to treat LiDAR-RGB fusion and Transformer-assisted navigation as the current primary technical direction.

Decisions:

- Treat `docs/project_evaluation/tunnel_ugv_lidar_rgb_transformer_navigation_plan.md` as the current primary project basis.
- Add `research/transformer_fusion/` as the first-class new research track.
- Reserve RL and diffusion for safety-filtered trajectory or policy proposals after the LiDAR costmap and RGB semantic baseline exist.
- Keep VLM/open-vocabulary work focused on annotation support, risk explanation, and low-frequency supervision rather than direct control.

Verification:

- Confirmed the updated Markdown report is 1070 lines and explicitly recommends LiDAR + RGB + Transformer + safety filter.
- Confirmed the Git remote remains credential-free.

Next Actions:

- Add `.gitignore` rules for zip archives, rosbags, raw sensor data, checkpoints, and experiment runs.
- Add a root README reflecting the LiDAR-RGB-Transformer direction.
- Move previous-code baselines into `baselines/` and run baseline tests after import/path adjustments.


## 2026-06-22 - RGB Pure-Vision Route Clarification

Scope:

- Recorded project-owner feedback that the original RGB pure-vision route can continue.
- Added `docs/project_evaluation/2026-06-22-rgb-vision-route-addendum.md` to preserve the rationale and engineering impact.
- Updated the repository-structure spec from a fusion-first framing to a dual-route plan: RGB-only engineering baseline plus LiDAR-RGB/Transformer enhanced research.

Decisions:

- Keep HBD-Net-RT as an active RGB-only engineering baseline.
- Add `research/rgb_vision/` for pure-RGB experiments beyond the preserved baseline.
- Keep LiDAR-RGB Transformer fusion as an enhanced research and safety-upgrade path rather than the only primary path.
- Continue to require safety validation for trench-margin detection, low light, reflections, water stains, repeating textures, and lens contamination.

Verification:

- Confirmed the project-owner feedback has been captured in a dedicated addendum.
- Confirmed the structure spec now lists both `research/rgb_vision/` and `research/transformer_fusion/`.

Next Actions:

- When reorganizing the repository, keep the RGB-only baseline runnable before adding fusion research skeletons.
- Add README guidance that pure RGB is the near-term MVP route and LiDAR-RGB fusion is the enhancement path.


## 2026-06-22 - Full Asset Tracking Reorganization

Scope:

- User requested that project files not be excluded and that all files be uploaded to GitHub.
- Downloaded Git LFS v3.7.1 to `/tmp` because system `git-lfs` was unavailable and sudo installation required a password.
- Enabled local Git LFS tracking for archives, model weights, videos, databases, rosbags, numpy arrays, pickles, tarballs, and related binary assets.
- Moved the original archive into `archive/original/`.
- Moved HBD-Net-RT into `baselines/hbdnet_rt/` as the active RGB-only baseline.
- Moved the older pure-vision project and all remaining imported assets into `baselines/vision_obstacle_avoidance_legacy/`.
- Renamed the old nested `.git` directory to `baselines/vision_obstacle_avoidance_legacy/original_metadata/git/` so it can be uploaded as ordinary files rather than treated as a nested repository.
- Added root README, research README files, config skeletons, and shared project-area README files.

Decisions:

- Use Git LFS instead of `.gitignore` exclusions for large project assets.
- Preserve all imported project assets unless a credential scan identifies secret material.
- Keep root-level `README.md` focused on the RGB-only MVP route plus LiDAR-RGB/Transformer enhancement path.

Verification:

- Git LFS binary verified as `git-lfs/3.7.1`.
- Git LFS release archive SHA-256 matched the official release hash.
- Targeted scan found no `github_pat_`, GitHub short tokens, or private-key blocks in project files.
- Full HBD-Net-RT test command `PYTHONPATH=src pytest tests -q` starts but cannot collect torch-dependent tests because this environment does not have `torch` installed.
- Non-torch HBD-Net-RT subset passed: `35 passed in 0.39s` for config, DWA, safety-state-machine, and scenario tests.

Next Actions:

- Stage all files, verify LFS pointers for large assets, commit, then attempt authenticated push to GitHub.

## 2026-06-22 - GitHub Upload Attempt

Scope:

- Verified local `main` is clean after the full asset-tracking reorganization commit.
- Verified `origin` is `https://github.com/SolanumLycopersicumX/5TD.git`.
- Attempted to push `main` and all Git LFS assets to GitHub.

Decisions:

- Did not use the GitHub token previously pasted in chat because it is exposed credential material and should be revoked or rotated.
- Kept Git LFS as the upload mechanism for large binary assets instead of excluding files.

Verification:

- `git lfs ls-files` lists the original archive, database, video samples, demo videos, and SAM model checkpoint.
- Large assets are stored in the Git index as LFS pointer files, including `archive/original/vision_obstacle_avoidance.zip` and `baselines/vision_obstacle_avoidance_legacy/models/sam_vit_b_01ec64.pth`.
- `git push -u origin main` was blocked by missing GitHub HTTPS credentials: the local machine has no credential helper configured.
- The local machine also has no GitHub SSH private key under `/home/tomato/.ssh`, and `gh` is not installed.

Next Actions:

- Configure GitHub authentication securely on this machine, then run the LFS-enabled push command again.

## 2026-06-22 - Push Protection Redaction

Scope:

- GitHub Push Protection blocked the branch push because a legacy hidden Claude settings file contained GitHub token text.
- Kept the settings file in the repository, but replaced the token value with a placeholder.
- Rebuilt local unpublished history on top of the remote initial commit so the blocked secret is not present in pushed commits.

Verification:

- Re-ran hidden-file secret scanning with `rg --hidden --no-ignore` excluding `.git/`.
- Confirmed large assets remain tracked through Git LFS.

Next Actions:

- Push the cleaned history and verify GitHub receives both normal Git objects and LFS assets.
