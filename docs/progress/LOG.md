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

## 2026-06-24 - GitHub Sync and Knowledge Vault Download

Scope:

- Fetched remote `main` from `https://github.com/SolanumLycopersicumX/5TD.git`.
- Downloaded and integrated the remote knowledge-base updates into the local workspace.
- Committed the current local training progress, including current segmentation tools, tests, style-cleanup work, extracted keyframes, derived datasets, run summaries, overlays, and model checkpoints.
- Restored the missing GitHub CLI binary at the path referenced by the local Git credential helper.
- Pushed the rebased local progress commit to GitHub.

Verification:

- Remote `origin/main` had four incoming commits before sync: Obsidian knowledge-base design, knowledge vault, Chinese localization, and UGV navigation research report.
- Local `docs/knowledge/` exists after sync and contains 46 files.
- Local progress commit after rebase: `fb61328 Add passable segmentation training progress`.
- GitHub CLI authentication was restored through the existing system keyring for account `SolanumLycopersicumX`.
- `git push origin main` uploaded 4 Git LFS model checkpoint objects and advanced remote `main` from `c10afe3` to `fb61328`.

Next Actions:

- Keep using `git fetch` before future upload work to detect teammate changes early.
- If `/tmp` is cleaned again, use `/home/tomato/.local/bin/gh` or reinstall GitHub CLI before pushing.

## 2026-06-24 - Third-Round Artifact Correction Label

Scope:

- Added `surface_artifact_passable` to both RGB keyframe Labelme label lists.
- Updated annotation rules for third-round correction of small rocks, shallow pits, stains, cracks, and other drivable surface texture artifacts.
- Clarified that drivable surface artifacts must remain inside `ego_passable` and must not be labeled as `ditch`.
- Clarified that only large avoidable objects should be labeled as `debris`, and only real deep channels or drainage grooves should be labeled as `ditch`.

Verification:

- Confirmed the two active annotation batches have matching `labels.txt` files.
- Confirmed `surface_artifact_passable` appears in both label lists and in the annotation guidance.

Next Actions:

- Use Labelme to mark current model false-positive regions with `surface_artifact_passable`.
- After correction annotations are available, update the dataset preparation and third-round training loss to penalize `ditch` predictions on these regions.

## 2026-06-24 - Third-Round Artifact-Corrected Training

Scope:

- Trained a third-round passable-road model using the corrected 41 Labelme images and the new `surface_artifact_passable` label.
- Updated `prepare_dataset.py` defaults to generate a three-target derived dataset: `ego_passable`, `ditch`, and `surface_artifact_passable`.
- Added `train_passable_ditch_artifact.py`, which keeps the model output as two classes (`ego_passable`, `ditch`) while using the artifact mask only as a training constraint.
- Added a true-ditch safety loss so real drainage channels must stay non-passable and be predicted as `ditch`.
- Switched the useful third-round run to fine-tune from the previous v2 checkpoint instead of training from scratch. The scratch v3 run collapsed the ditch head and was not considered usable.

Results:

- Dataset: `data/derived/passable_ditch_artifact_2026-06-24`.
- Final checkpoint: `runs/passable_ego/passable_ditch_artifact_v3_finetune/best_model.pt`.
- Summary: `runs/passable_ego/passable_ditch_artifact_v3_finetune/summary.json`.
- All 41 labeled overlays: `runs/passable_ego/passable_ditch_artifact_v3_finetune/overlays_all_labeled/`.
- Validation metrics for the fine-tuned model:
  - `passable_iou`: 0.9727.
  - `safe_iou`: 0.9730.
  - `ditch_iou`: 0.5227.
  - `ditch_as_passable_rate`: 0.00024.
  - `artifact_ditch_false_positive_rate`: 0.00564.
  - `artifact_passable_false_negative_rate`: 0.03107.
- Compared with v2 on the same artifact validation labels, the artifact passable-gap rate improved from 0.2261 to 0.0311, and true ditch marked as passable improved from 0.0447 to 0.00024.

Verification:

- Unit tests passed: `conda run -n lerobot python -m unittest discover -s tests -p 'test_passable*.py'` reported 13 tests OK.
- Confirmed the corrected dataset has 41 images, with 34 training and 7 validation samples.
- Confirmed train split label coverage: 34 `ego_passable`, 33 `ditch`, 4 `surface_artifact_passable`.
- Confirmed validation split label coverage: 7 `ego_passable`, 7 `ditch`, 6 `surface_artifact_passable`.
- Visually inspected validation overlays including `test_video_0002`, `test_video_0004`, and `test_video_0005`.

Caveats:

- The fine-tuned model is safer around real drainage channels, but `ditch_iou` is lower than v2, which means the predicted ditch shape is less complete.
- A few isolated red speckles still remain on road texture in some overlays. This should be handled next with more focused labels and/or small connected-component filtering in inference.

Next Actions:

- Review `overlays_all_labeled` before deciding whether this checkpoint is good enough for a demo.
- Add a post-processing option to remove tiny isolated ditch components if the remaining red speckles affect driving behavior.
- Continue adding new annotated frames when more data is available.

## 2026-06-24 - Ditch Speckle Post-Processing and V4 Direction

Scope:

- Added post-processing for two-head passable-road predictions to remove small isolated `ditch` connected components.
- Kept this as inference/visualization post-processing only; it does not change training masks or saved checkpoints.
- Set the default visualization threshold to `MIN_DITCH_COMPONENT_AREA = 500` pixels at the resized `384x640` model resolution.
- Regenerated filtered overlays for the corrected 41 labeled images and the 41 additional extracted keyframes.
- Clarified annotation guidance for v4: far-left curbs, wall-base edges, isolation blocks, and crash blocks should remain `left_barrier`, not `ditch`.

Outputs:

- Filtered 41 labeled overlays: `runs/passable_ego/passable_ditch_artifact_v3_finetune/overlays_all_labeled_filtered/`.
- Filtered additional-keyframe overlays: `runs/passable_ego/passable_ditch_artifact_v3_finetune/overlays_more_keyframes_filtered/`.
- Updated visualization default: `tools/passable_segmentation/visualize_passable_ditch.py`.

Verification:

- Added regression tests for small connected-component filtering and probability post-processing.
- Unit tests passed: `conda run -n lerobot python -m unittest discover -s tests -p 'test_passable*.py'` reported 15 tests OK.
- Syntax check passed: `conda run -n lerobot python -m py_compile tools/passable_segmentation/*.py tests/test_passable_segmentation_tools.py`.
- Visually inspected filtered `test_video_0004_overlay.jpg`; small isolated red components were removed while long right-side drainage-channel predictions remained.

Caveats:

- The filter removes only small isolated `ditch` blobs. It will not fix large connected false positives, such as a far-left wall or curb being consistently predicted as a long ditch segment.
- V4 should add `left_barrier` as an explicit model output class, using the existing `left_barrier` annotations, so the network can separate left-side hard boundaries from true drainage channels.

Next Actions:

- For v4, prepare a dataset with `ego_passable`, `ditch`, `left_barrier`, and auxiliary `surface_artifact_passable`.
- Train a three-output model: passable road, drainage channel, and left hard boundary.
- Use `left_barrier` for left curb and crash-block correction; only use `ditch` for real trenches or deep drainage channels.

## 2026-06-24 - GitHub Upload Before V4

Scope:

- Uploaded the third-round artifact-corrected training progress to GitHub before starting v4 work.
- Included corrected Labelme JSONs, v3/v3-finetune checkpoints, derived v3 datasets, filtered overlays, training scripts, tests, and progress notes.
- Kept model checkpoints under Git LFS.

Verification:

- `git fetch origin` completed before commit.
- Syntax check passed: `conda run -n lerobot python -m py_compile tools/passable_segmentation/*.py tools/extract_more_keyframes.py tests/test_passable_segmentation_tools.py`.
- Unit tests passed: `conda run -n lerobot python -m unittest discover -s tests -p 'test_passable*.py'` reported 15 tests OK.
- Commit pushed to GitHub: `30fcbaa Add artifact-corrected passable segmentation progress`.
- `git push origin main` uploaded 4 LFS checkpoint objects and advanced `origin/main` from `8cdd6d5` to `30fcbaa`.

Next Actions:

- Start v4 training locally after the upload.

## 2026-06-24 - V4 Left-Barrier Training Experiment

Scope:

- Added `train_passable_ditch_left_barrier.py` for a three-output model: `ego_passable`, `ditch`, and `left_barrier`.
- Prepared v4 derived datasets from the corrected 41 images.
- First v4 dataset used `ego_passable`, `ditch`, `left_barrier`, and auxiliary `surface_artifact_passable`.
- First v4 run reduced `left_barrier_as_ditch_rate`, but over-predicted `left_barrier` across ceiling/wall regions.
- Added `tunnel_wall` as an auxiliary negative target for a second v4 wall-aux run.

Outputs:

- First v4 run: `runs/passable_ego/passable_ditch_left_barrier_v4/`.
- Wall-aux v4 dataset: `data/derived/passable_ditch_left_barrier_wall_aux_2026-06-24/`.
- Wall-aux v4 run: `runs/passable_ego/passable_ditch_left_barrier_v4_wall_aux/`.
- Wall-aux all-labeled overlays: `runs/passable_ego/passable_ditch_left_barrier_v4_wall_aux/overlays_all_labeled/`.

Results:

- First v4 run final validation:
  - `safe_iou`: 0.9046.
  - `ditch_iou`: 0.8057.
  - `left_barrier_iou`: 0.0320.
  - `left_barrier_as_ditch_rate`: 0.0067.
  - Problem: predicted `left_barrier` over large ceiling/wall regions.
- Wall-aux v4 run final validation:
  - `safe_iou`: 0.8115.
  - `ditch_iou`: 0.6437.
  - `left_barrier_iou`: 0.0722.
  - `left_barrier_as_ditch_rate`: 0.0497.
  - `wall_left_barrier_false_positive_rate`: 0.0344.
  - Problem: many true `ditch` pixels also activated the `left_barrier` head.
- With a simple ditch-priority post-processing check, effective `ditch_as_left_barrier_rate` dropped from 0.962 to about 0.033, and effective `left_barrier_iou` rose to about 0.139. This was measured but not yet made the default inference behavior.

Verification:

- Added regression tests for left-boundary/ditch confusion and wall-as-left-boundary loss.
- Unit tests passed: `conda run -n lerobot python -m unittest discover -s tests -p 'test_passable*.py'` reported 18 tests OK.
- Syntax check passed: `conda run -n lerobot python -m py_compile tools/passable_segmentation/*.py tests/test_passable_segmentation_tools.py`.
- Confirmed wall-aux dataset has 41 images, 34 train, 7 validation.
- Confirmed `left_barrier` and `tunnel_wall` are non-empty in every train and validation image.

Caveats:

- V4 is not ready to replace v3-finetune. It is an experiment showing that `left_barrier` can reduce left-boundary-as-ditch confusion, but the new class still needs better separation from both `ditch` and `tunnel_wall`.
- The current v3-finetune model remains the safer model for passable-road and ditch behavior.

Next Actions:

- Make ditch-priority post-processing explicit for v4 inference and overlays.
- Consider a staged or separate left-boundary head so passable/ditch performance from v3-finetune is not degraded.
- Add more focused labels or samples where left curb/crash-block, wall-base, and true ditch are simultaneously visible.

## 2026-06-24 - Staged Boundary-Wall Fusion Experiment

Scope:

- Kept the stable v3-finetune model as the main `ego_passable`/`ditch` model.
- Added an auxiliary two-output model for `left_barrier` and `tunnel_wall` only.
- Added rule-based fusion:
  - `ditch` has priority over `left_barrier`.
  - `tunnel_wall` always removes passability.
  - `left_barrier` is kept as a boundary cue and does not remove passability.
- Excluded `test_video*` from the new boundary-wall training dataset because it represents demo footage, not useful training data.
- Added inference post-processing for small connected components:
  - remove tiny isolated `ditch`, `left_barrier`, and `tunnel_wall` predictions.
  - fill small enclosed `ego_passable` holes when they are not `ditch` or `tunnel_wall`.

Outputs:

- Clean boundary-wall dataset: `data/derived/passable_ditch_left_barrier_wall_aux_no_testvideo_2026-06-24/`.
- Boundary-wall auxiliary run: `runs/passable_ego/boundary_wall_aux_v2_no_testvideo/`.
- Fused valid-label overlays: `runs/passable_ego/fused_passable_boundary_v2_no_testvideo/overlays_valid_labeled/`.
- Fused additional-keyframe overlays: `runs/passable_ego/fused_passable_boundary_v2_no_testvideo/overlays_more_keyframes/`.
- New scripts:
  - `tools/passable_segmentation/train_boundary_wall.py`.
  - `tools/passable_segmentation/visualize_fused_passable_boundary.py`.

Results:

- Boundary-wall auxiliary validation on real-video prefix `b0c37d`:
  - `left_barrier_iou`: 0.4963.
  - `tunnel_wall_iou`: 0.7308.
  - `wall_as_left_barrier_rate`: 0.00033.
  - `left_wall_overlap_rate`: near zero.
- Fused evaluation on 34 valid labeled images:
  - `safe_passable_iou`: 0.9583.
  - `ditch_iou`: 0.4455.
  - `left_barrier_iou`: 0.4738.
  - `tunnel_wall_iou`: 0.7978.
- Visual checks:
  - Right-side drainage-channel predictions stayed red and kept priority.
  - The previous v4-style `left_barrier` full-image spread was not observed in inspected samples.
  - Small road holes in `safe_passable` were filled after the fusion post-processing, while ditch and wall remained protected.

Verification:

- Unit tests passed: `conda run -n lerobot python -m unittest discover -s tests -p 'test_passable*.py'` reported 23 tests OK.
- Syntax check passed: `conda run -n lerobot python -m py_compile tools/passable_segmentation/*.py tests/test_passable_segmentation_tools.py`.
- Generated 34 valid-label fused overlays and 41 additional-keyframe fused overlays.

Caveats:

- `ditch_iou` is still weaker than desired on the 34-image fused evaluation, so more right-side drainage-channel labels are still needed.
- The staged structure is better than v4 for avoiding task interference, but it is still data-limited.
- The current best deployment candidate remains v3-finetune plus the staged fusion/post-processing layer, not the three-output v4 model.

Next Actions:

- Review `overlays_valid_labeled` and `overlays_more_keyframes` before deciding whether to continue tuning thresholds or collect more labels.
- Add more non-demo frames where right-side ditch, left curb/barrier, tunnel wall, and small road artifacts appear together.
- Keep `test_video*` excluded from future training and model-selection datasets unless it is intentionally being used only as a visual demo.

## 2026-06-24 - Fused Overlay Post-Processing Fixes for f000030/f000090/f000210

Scope:

- Investigated the reported large false blobs in `f000030` and `f000090`, plus the large missed passable-road region in `f000210`.
- Confirmed these were fusion/post-processing issues, not a reason to retrain:
  - floating `ego_passable` islands caused ceiling/edge green blobs.
  - medium isolated `ditch` blobs caused false red patches on passable road.
  - the passable-road hole filling threshold was too small for the large `f000210` road gap.
- Updated fused inference post-processing:
  - keep only passable components connected to the bottom road region, with a largest-component fallback.
  - raise the default small-ditch filter to 2000 pixels at `384x640`.
  - raise the default passable-hole fill limit to 10000 pixels, still protecting `ditch` and `tunnel_wall`.

Outputs:

- Refreshed additional-keyframe overlays: `runs/passable_ego/fused_passable_boundary_v2_no_testvideo/overlays_more_keyframes/`.
- Refreshed valid-label overlays: `runs/passable_ego/fused_passable_boundary_v2_no_testvideo/overlays_valid_labeled/`.
- Contact sheets for manual review:
  - `runs/passable_ego/fused_passable_boundary_v2_no_testvideo/debug_contact_sheets/f000030_contact.jpg`.
  - `runs/passable_ego/fused_passable_boundary_v2_no_testvideo/debug_contact_sheets/f000090_contact.jpg`.
  - `runs/passable_ego/fused_passable_boundary_v2_no_testvideo/debug_contact_sheets/f000210_contact.jpg`.

Results:

- Visual review confirmed:
  - `f000090` top green false blob was removed.
  - `f000030` and `f000090` demo-frame red road blobs were removed.
  - `f000210` large passable-road gap was filled back to green.
  - right-side main drainage-channel predictions remained red.
- Fused evaluation on 34 valid labeled images after the fix:
  - `safe_passable_iou`: 0.9605.
  - `ditch_iou`: 0.4699.
  - `left_barrier_iou`: 0.4976.
  - `tunnel_wall_iou`: 0.7978.

Verification:

- Added regression tests for bottom-connected passable filtering, default medium ditch-blob filtering, and default larger passable-hole filling.
- Unit tests passed: `conda run -n lerobot python -m unittest discover -s tests -p 'test_passable*.py'` reported 26 tests OK.
- Syntax check passed: `conda run -n lerobot python -m py_compile tools/passable_segmentation/*.py tests/test_passable_segmentation_tools.py`.

Caveats:

- This is still post-processing around a small-data model. It fixes the observed large blobs/gaps, but more real annotated frames are still needed for stronger raw model behavior.
- `test_video*` remains excluded from training/model selection and should be treated as demo/inference-only material.

## 2026-06-24 - GitHub Upload of Staged Fusion Work

Scope:

- Uploaded the v4 experiments, boundary-wall auxiliary model, staged fusion visualizations, post-processing fixes, tests, derived datasets, overlays, and checkpoints to GitHub.
- Kept model checkpoints under Git LFS.
- Pushed directly to `origin/main`, matching the previous project sync workflow.

Verification:

- `git fetch origin` completed before committing.
- Unit tests passed before upload: `conda run -n lerobot python -m unittest discover -s tests -p 'test_passable*.py'` reported 26 tests OK.
- Syntax check passed before upload: `conda run -n lerobot python -m py_compile tools/passable_segmentation/*.py tests/test_passable_segmentation_tools.py`.
- First `git push origin main` uploaded LFS objects but failed ordinary Git object transfer with HTTP 408.
- Retried `git push origin main`; retry succeeded and advanced GitHub from `30fcbaa` to `64ba2fc`.

Commit:

- `64ba2fc Add staged boundary fusion experiments`.

Next Actions:

- Start designing the pipeline that converts the fused `safe_passable` mask into a motion trajectory and driver command interface.

## 2026-06-24 - RS232 Navigation Bridge Plan and ChatGPT Summary

Scope:

- Recorded the implementation plan for converting fused passable-road masks into conservative motion commands.
- Recorded the RS232 driver decision for the next vehicle-control bridge.
- Summarized why the old baseline should remain reference-only and not become a runtime dependency.
- Created a Chinese-facing project summary suitable for uploading to ChatGPT or another external review context.

Files:

- `docs/superpowers/plans/2026-06-24-rs232-navigation-bridge.md`
- `docs/progress/2026-06-24-rs232-navigation-bridge-summary.md`

Key Decisions:

- New runtime code should live under `src/tunnel_nav`.
- The first implementation should run offline from saved masks and output command JSON plus overlays.
- RS232 support should start as a dry-run adapter only.
- Live serial writes should remain disabled until angular sign, emergency stop, and low-speed behavior are physically validated.

## 2026-06-24 - BEV / DWA External Plan Comparison

Scope:

- Reviewed `tunnel_ugv_plan_A_BEV_DWA_trajectory.md`.
- Compared the external BEV / Risk Grid / DWA proposal with the current local RS232 navigation bridge plan.
- Recorded a feasibility comparison and recommended merged roadmap.

Result:

- The external plan is the stronger medium-term architecture for real navigation safety.
- The current local plan remains the better immediate Stage 0 because it is testable now, requires no camera calibration, and keeps RS232 in dry-run mode.
- Recommended sequence: offline mask-to-command bridge first, then calibrated BEV / Risk Grid, then DWA and safety state machine, then live RS232 bring-up.

Document:

- `docs/progress/2026-06-24-bev-dwa-plan-comparison.md`.

## 2026-06-24 - Active Plan Switched to BEV / DWA Offline Bridge

Scope:

- Accepted the user's request to follow ChatGPT plan A0/A1 more directly.
- Replaced the image-space-first implementation plan with a simplified BEV / Risk Grid / DWA offline prototype plan.
- Kept the RS232 adapter as dry-run only.
- Marked the earlier RS232/image-space bridge plan as superseded, but retained it as reference.

Active Plan:

- `docs/superpowers/plans/2026-06-24-bev-dwa-rs232-navigation-bridge.md`

Key Constraints:

- First BEV is pseudo-BEV, not calibrated IPM.
- DWA runs offline over generated risk grids.
- Output remains command JSON, overlay visualization, and RS232 dry-run JSON.
- Live serial writes remain blocked until calibration, emergency stop, angular sign, and low-speed behavior are validated.

## 2026-06-24 - Offline BEV / DWA / RS232 Dry-Run Bridge Implementation

Scope:

- Added a new `src/tunnel_nav` runtime path for the first offline navigation bridge.
- Added core data structures for masks, pseudo-BEV grids, DWA trajectories, safety-filtered motion commands, and conservative navigation configuration.
- Added fused mask export from the existing segmentation fusion script.
- Added pseudo-BEV occupancy/risk grid generation from `safe_passable`, `ditch`, `tunnel_wall`, and `left_barrier` masks.
- Added a conservative low-speed DWA planner over the risk grid.
- Added a safety filter with `S0_NORMAL`, `S1_CAUTIOUS`, `S2_SLOWDOWN`, and `S3_STOP` command behavior.
- Added an RS232 dry-run adapter that converts physical velocity commands to Modbus register values without opening serial.
- Added an offline CLI that writes command JSON, RS232 dry-run JSON, and BEV/DWA overlay images.
- Updated `configs/robot/vehicle.yaml` with differential-drive, RS232 dry-run, and pseudo-BEV/DWA defaults.

Key Files:

- `src/tunnel_nav/motion.py`
- `src/tunnel_nav/bev.py`
- `src/tunnel_nav/dwa.py`
- `src/tunnel_nav/safety.py`
- `src/tunnel_nav/rs232.py`
- `tools/navigation_bridge/run_offline_bev_dwa_bridge.py`
- `tests/test_bev_dwa_navigation_bridge.py`

Verification:

- `conda run -n lerobot python -m unittest discover -s tests -p 'test_bev_dwa_navigation_bridge.py' -v`
- `conda run -n lerobot python -m unittest discover -s tests -p 'test_passable_segmentation_tools.py' -v`
- `conda run -n lerobot python -m py_compile src/tunnel_nav/*.py tools/navigation_bridge/run_offline_bev_dwa_bridge.py tools/passable_segmentation/visualize_fused_passable_boundary.py tests/test_bev_dwa_navigation_bridge.py`

Caveats:

- The current BEV is pseudo-BEV and is not meter-accurate calibrated IPM.
- This bridge is for offline command/trajectory validation only.
- Live RS232 remains disabled until camera calibration, emergency stop, angular sign, and low-speed real-vehicle behavior are validated.

## 2026-06-24 - LAN File Server Address Fix

Scope:

- Fixed the LAN file server address selection after `192.18.0.1` / `198.18.0.1` appeared as misleading share URLs.
- Restricted displayed LAN candidates to RFC1918 private network ranges: `10.0.0.0/8`, `172.16.0.0/12`, and `192.168.0.0/16`.
- Added Linux default-route parsing through `ip -4 route show default` so the server prefers the actual Wi-Fi LAN source address instead of virtual tunnel or bridge addresses.
- Restarted the local file server on port `8000`.

Result:

- Current local share URL: `http://192.168.110.16:8000/`.
- The server log now shows only `LAN: http://192.168.110.16:8000/`.

Verification:

- `conda run -n lerobot python -m unittest discover -s tests -p 'test_lan_file_server.py' -v`
- `conda run -n lerobot python -m py_compile tools/lan_file_server.py tests/test_lan_file_server.py`
- Direct helper check returned `['192.168.110.16']`.
- `curl -sSf http://127.0.0.1:8000/` returned the upload page HTML.
