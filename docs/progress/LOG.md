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

## 2026-06-22 - GitHub Upload Completed

Scope:

- Successfully pushed the cleaned `main` branch to `https://github.com/SolanumLycopersicumX/5TD.git`.
- Uploaded all tracked Git LFS assets, including the original archive, model checkpoint, videos, and database files.
- Preserved the legacy settings file with token values redacted instead of excluding it from Git.

Verification:

- `git push -u origin main` completed successfully after rebuilding local unpublished history on top of the remote initial commit.
- GitHub accepted the push after the hidden settings token values were replaced with placeholders.
- `git lfs ls-files` still lists 10 LFS-managed assets.

Next Actions:

- Start the next engineering phase: run the RGB-only baseline environment and prepare the first perception/navigation demo.

## 2026-06-24 - Annotation, First Training, and Safety Review

Scope:

- Validated the first non-demo Labelme annotation set from `data/annotation_batches/rgb_keyframes_2026-06-22/images`.
- Confirmed the demo-video frames are not useful for training and kept them out of the current dataset.
- Installed a user-local `git-lfs` binary so normal `git lfs`, `git push`, and `git pull` workflows can run without sudo.
- Pushed the completed first annotation JSONs to GitHub in commit `b85fcab`.
- Extracted a second batch of non-demo JPEG keyframes at original 720x1280 resolution into `data/annotation_batches/rgb_keyframes_2026-06-24_more_keyframes/`.
- Added the first passable-road segmentation tooling under `tools/passable_segmentation/`.
- Prepared the first derived binary `ego_passable` dataset in `data/derived/passable_ego_2026-06-24/`.
- Trained the first augmented small U-Net model in `runs/passable_ego/first_augmented/`.
- Generated visual overlays for the labeled set and the newly extracted frames.

Decisions:

- Use the 41 completed non-demo annotations as the initial training set while more data is manually labeled.
- Use augmentation for the small initial dataset, including blur, water-stain simulation, brightness changes, shadowing, and horizontal flips.
- Keep `demo_video_*` frames out of training unless a future review finds a specific usable subset.
- Treat the right drainage channel as the existing `ditch` class, not `right_barrier`.
- For the next model, train `ego_passable` and `ditch` together; the safe passable mask should be `ego_passable AND NOT ditch`.
- Update this progress log immediately after each completed work item going forward.

Verification:

- First non-demo annotation set: 41 images, 41 JSON labels, no image-dimension mismatch, no out-of-bounds polygon points.
- Label counts in the first set: `ego_passable=41`, `ditch=40`, `left_barrier=197`, `tunnel_wall=41`, `debris=18`, `construction_vehicle=7`, `worker=3`, `right_barrier=0`.
- First derived dataset: 41 total images, 34 train, 7 validation, 0 empty `ego_passable` masks.
- First model final validation metrics: IoU about 0.988 and Dice about 0.994 on the small held-out validation split.
- Unit tests for the first passable-segmentation tools passed with `conda run -n lerobot python -m unittest discover -s tests -p 'test_passable*.py'`.
- Second keyframe batch contains 41 JPEG images at 720x1280 resolution.

Caveats:

- The first model is not deployment-ready. Visual review showed ceiling false positives, and the user identified a safety-critical false positive where right-side drainage channel regions were predicted as passable road.
- The validation metrics are over-optimistic because the dataset is very small and visually similar across frames.

Next Actions:

- Finish and train the dual-output `ego_passable + ditch` model.
- Generate overlays for the dual-output model and inspect whether predicted passable regions are removed from the drainage channel.
- Continue manual Labelme annotation on `data/annotation_batches/rgb_keyframes_2026-06-24_more_keyframes/images`, especially marking the drainage channel as `ditch`.
- Add another log entry immediately after the dual-output training run is complete.

## 2026-06-24 - Dual Passable and Ditch Training

Scope:

- Added dual-output segmentation support for `ego_passable` and `ditch`.
- Updated dataset preparation so repeated `--label` arguments create per-class mask folders and 4-column manifests.
- Added `tools/passable_segmentation/train_passable_ditch.py` for two-head training, safety-aware metrics, and validation overlays.
- Reused the small U-Net backbone with `out_channels=2`.
- Added loss terms that penalize predicted passable pixels on true ditch pixels and direct overlap between predicted passable and predicted ditch.
- Prepared `data/derived/passable_ditch_2026-06-24/`.
- Trained the dual-output model into `runs/passable_ego/passable_ditch_augmented/`.

Decisions:

- Use the model's safe passable output as predicted `ego_passable` minus predicted `ditch`.
- Keep `ditch` as a separate semantic output instead of folding it into background, because the right drainage channel is a safety-critical negative class.
- Continue using the existing 41 annotated non-demo frames until the second manual annotation batch is ready.

Verification:

- Unit tests passed: `conda run -n lerobot python -m unittest discover -s tests -p 'test_passable*.py'` reported 8 tests OK.
- Dual derived dataset: 41 total images, 34 train, 7 validation.
- Empty masks: `ego_passable=[]`; `ditch=["11d80f_0001"]`.
- Training ran on CUDA for 90 epochs.
- Best checkpoint: `runs/passable_ego/passable_ditch_augmented/best_model.pt`.
- Final validation from the best checkpoint: `passable_iou=0.9499`, `ditch_iou=0.7992`, `safe_iou=0.9500`, `ditch_as_passable_rate=0.0447`, `passable_ditch_overlap_rate` approximately 0.
- Validation overlays were written to `runs/passable_ego/passable_ditch_augmented/overlays_val/`.

Caveats:

- The dataset is still only 41 annotated frames, so these validation metrics should be treated as a smoke test rather than proof of robustness.
- The first annotation set has one frame without `ditch`, and the visual distribution is still narrow.

Next Actions:

- Generate dual-output overlays on the newly extracted non-demo frames.
- Visually inspect whether the right drainage channel is removed from the safe passable region.
- Keep annotating the second batch, marking drainage channels as `ditch`.

## 2026-06-24 - Dual Model Visualization Review

Scope:

- Added `tools/passable_segmentation/visualize_passable_ditch.py` for repeatable dual-output visualization.
- Generated labeled-set overlays in `runs/passable_ego/passable_ditch_augmented/overlays_all_labeled/`.
- Generated new-keyframe overlays in `runs/passable_ego/passable_ditch_augmented/overlays_more_keyframes/`.
- Performed a quick visual spot check on representative labeled and newly extracted frames.

Verification:

- Unit tests passed again after adding the dual-output overlay test: 9 tests OK.
- `overlays_all_labeled/` contains 41 JPEG overlays.
- `overlays_more_keyframes/` contains 41 JPEG overlays.
- New-keyframe overlays are 1280x384 two-panel images.
- Labeled-set overlays are 1920x384 three-panel images.
- Spot checks showed the right drainage channel predicted as red `ditch` and removed from green safe passable area on inspected samples.

Caveats:

- A ceiling false positive remains visible in at least one newly extracted frame.
- More annotations are needed for ceiling/high-wall negatives and varied drainage-channel appearances before this can be trusted beyond a demo-level smoke test.

Next Actions:

- Continue annotating the second keyframe batch.
- During annotation, keep all drainage channels as `ditch` and avoid labeling ceiling or high-wall regions as `ego_passable`.
- Retrain after the second batch is complete.

## 2026-06-24 - Training Code Style Cleanup

Scope:

- Applied the company code-writing guidance from `代码规范.md` to the current maintained training and annotation-helper code.
- Limited the cleanup to `tools/passable_segmentation/`, `tools/extract_more_keyframes.py`, and the matching passable-segmentation tests.
- Did not modify HBD-Net-RT or the legacy OpenCV/SAM code in this pass.
- Removed `argparse` from the current training utilities.
- Replaced training config dataclasses with top-of-file tunable parameters and small `build_train_config()` helpers.
- Kept PyTorch `Dataset` and `nn.Module` classes because they hold required framework state.
- Added one-line module/function/class docstrings where they clarify public entry points.
- Standardized script logs with `[OK]`, `[DATA]`, and `[TRAIN]` prefixes.
- Changed `prepare_dataset.py` defaults to the current dual-label `ego_passable + ditch` workflow, with the old single-label default left as a comment switch.

Verification:

- Syntax check passed: `python -m py_compile tools/passable_segmentation/*.py tools/extract_more_keyframes.py tests/test_passable_segmentation_tools.py`.
- Unit tests passed: `conda run -n lerobot python -m unittest discover -s tests -p 'test_passable*.py'` reported 10 tests OK.
- Current training code grep check found no `argparse`, `@dataclass`, `class TrainConfig`, or `class PassableDitchConfig`.
- Default data preparation entry ran successfully and generated the dual-label dataset summary for 41 images, 34 train, 7 validation.
- Default binary visualization entry wrote 41 overlays.
- Default dual-output visualization entry wrote 41 overlays.

Caveats:

- This was a style and maintainability cleanup, not a retraining run.
- Existing checkpoints still load because checkpoint configs were already saved as dictionaries.

Next Actions:

- Continue using top-of-file tunable parameters for training script changes.
- After the second annotation batch is complete, regenerate the dual-label dataset and rerun `train_passable_ditch.py`.
