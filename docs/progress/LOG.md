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

## 2026-06-24 - Warthog Chassis CAD Archive Received

Scope:

- Received `Warthog-02M-Pro 4WD Chassis (2).zip` under `/home/tomato/5TD`.
- Inspected the archive contents and confirmed it contains CAD files, not a ready-to-run Gazebo model.
- Extracted the CAD files to `assets/cad/warthog_02m_pro_4wd_chassis/`.

Files:

- `Warthog-02M-Pro 4WD Chassis.IGS`
- `Warthog-02M-Pro 4WD Chassis.STEP`
- `Warthog-02M-Pro 4WD Chassis.x_t`

Findings:

- The STEP file was exported by SolidWorks 2024 as STEP AP214.
- The CAD unit is millimeters, so Gazebo/URDF conversion must scale meshes to meters.
- The current machine does not have FreeCAD, Blender, assimp, gmsh, or Python CAD libraries installed for direct CAD-to-mesh conversion.
- The archive has no URDF, Xacro, SDF, STL, DAE, or OBJ files.

Next Actions:

- Convert STEP/IGS/x_t to STL or DAE using a CAD-capable tool before Gazebo use.
- Build a URDF/SDF wrapper with collision geometry, wheel joints, differential-drive plugin settings, and camera pose.
- Add Git LFS rules for `.STEP`, `.IGS`, and `.x_t` before uploading extracted CAD files to GitHub.

## 2026-06-25 - CAD Conversion Tool Installation

Scope:

- Installed system CAD and mesh conversion tools for the Warthog chassis model.
- Installed `gmsh`, `assimp-utils`, `python3-gmsh`, and `python3-meshio` through `apt`.
- Installed FreeCAD through `snap` after resuming the interrupted download.

Installed Tools:

- FreeCAD snap: `1.1-g0108fd4b`, command line reports `FreeCAD 1.1.1 Revision: 44227 +647`.
- Gmsh: `4.12.1`.
- Assimp command-line tools: `5.3`.
- Python Gmsh for system Python: `4.12.1`.

Verification:

- `snap list freecad kf6-core24 lxqt-support-core24` confirms FreeCAD and its snap dependencies are installed.
- `which freecad freecad.cmd gmsh assimp meshio` confirms all required commands are available.
- `freecad.cmd --version` runs successfully.
- `/usr/bin/python3` can import `gmsh` and reports version `4.12.1`.
- FreeCAD successfully read `assets/cad/warthog_02m_pro_4wd_chassis/Warthog-02M-Pro 4WD Chassis.STEP`.
- FreeCAD STEP import reported 18 solids and a model bounding box of about `1308.556 x 1608.0 x 999.246 mm`.

Notes:

- Use `/usr/bin/python3` for `python3-gmsh`; the conda Python environments do not automatically see Ubuntu system Python packages.
- The next step is to export a visual mesh, likely STL first, then build a simplified URDF/SDF wrapper for Gazebo.

## 2026-06-25 - Warthog CAD STL and Gazebo Model Export

Scope:

- Exported the Warthog STEP chassis assembly to an STL visual mesh using FreeCAD.
- Normalized the mesh by centering it in X/Y, placing the lowest CAD point near ground, and rotating the CAD long axis into Gazebo X.
- Generated a Gazebo Sim SDF model wrapper with simplified collision geometry and four wheel joints.
- Generated a local URDF wrapper for ROS/RViz-style inspection.
- Added Git LFS rules for CAD and mesh formats: STL, DAE, OBJ, STEP/STP, IGS/IGES, and x_t.

Generated Files:

- `tools/cad/export_warthog_stl.py`
- `sim/gazebo/models/warthog_02m_pro_4wd_chassis/meshes/chassis_visual.stl`
- `sim/gazebo/models/warthog_02m_pro_4wd_chassis/meshes/chassis_visual.metadata.json`
- `sim/gazebo/models/warthog_02m_pro_4wd_chassis/model.sdf`
- `sim/gazebo/models/warthog_02m_pro_4wd_chassis/model.config`
- `sim/gazebo/models/warthog_02m_pro_4wd_chassis/README.md`
- `sim/urdf/warthog_02m_pro_4wd_chassis.urdf`

Export Result:

- STL size: about 7.0 MB.
- STL units: millimeters.
- SDF/URDF mesh scale: `0.001 0.001 0.001`.
- Normalized mesh bounding box: about `1608.0 x 1206.5 x 999.0 mm`.
- Mesh facets: `145223`.
- Assimp-readable vertices/faces: `342929` vertices and `145223` faces.

Verification:

- `freecad.cmd` successfully exported the mesh from STEP.
- `assimp info` successfully imported the generated STL.
- Python XML parsing succeeded for `model.sdf`, `model.config`, and the URDF.
- `python3 -m py_compile tools/cad/export_warthog_stl.py` succeeded.
- `git check-attr` confirms the new STL and source CAD files match Git LFS rules.

Caveats:

- Wheel radius, wheel separation, collision boxes, inertia, and mass values are initial simulation placeholders.
- The SDF targets Gazebo Sim and uses the `gz-sim-diff-drive-system` plugin.
- The model has not yet been launched in Gazebo on this machine.

## 2026-06-25 - Warthog Wheel Placeholder Alignment Fix

Scope:
- Investigated the Gazebo/RViz visual mismatch where the black simplified front wheel cylinders did not overlap the gray CAD chassis frame.

Root Cause:
- The gray chassis is the exported CAD STL, while the black wheels are simplified SDF/URDF cylinder placeholders. Their earlier positions were estimated and sat ahead of/outside the CAD wheel mounting plates.

Fix:
- Aligned wheel placeholder centers to CAD mounting plate solids: front x `0.343 m`, rear x `-0.498 m`, y `+/-0.500 m`, z `0.285 m`.
- Updated SDF and URDF wheel radius to `0.285 m`, wheel width to `0.220 m`, and SDF DiffDrive wheel separation to `1.000 m`.

Verification:
- Added and ran `tests/test_warthog_gazebo_model.py` to lock SDF/URDF wheel pose and geometry consistency.

Caveat:
- These are still simplified simulation placeholders, not final measured dynamics parameters. Measure the physical chassis before controller tuning.

## 2026-06-25 - Warthog Duplicate Wheel Visual Removal

Scope:
- Investigated the remaining visual mismatch after wheel center alignment.

Root Cause:
- The exported CAD STL already contains detailed wheel geometry, while the SDF/URDF wheel links still rendered black simplified cylinder visuals on top of the CAD wheels.

Fix:
- Removed wheel link visual cylinder elements from SDF and URDF.
- Kept wheel cylinder collision geometry, wheel joints, and DiffDrive parameters for simulation.

Verification:
- Extended `tests/test_warthog_gazebo_model.py` to require wheel links to keep collision elements but not render duplicate visual elements.

## 2026-06-25 - Gazebo Flat Ground Test World

Scope:
- Investigated the Warthog model falling when Gazebo simulation starts.

Root Cause:
- The model was being inserted into an empty/default scene without a ground collision plane, so gravity pulled the chassis downward.

Fix:
- Added `sim/gazebo/worlds/warthog_flat_test.sdf` with a static ground plane, directional light, physics settings, and the Warthog model include.
- Documented the launch command with `GZ_SIM_RESOURCE_PATH`, `SDF_PATH`, and `gz sim -r`.

Verification:
- Added a regression test requiring the flat test world to include ground plane collision/visual geometry and the Warthog model include.

## 2026-06-25 - Gazebo Launch Script ROS Setup Fix

Scope:
- Fixed `sim/gazebo/run_warthog_flat_test.sh` failing at ROS setup with `AMENT_TRACE_SETUP_FILES: unbound variable`.

Root Cause:
- The script enabled Bash nounset with `set -u` before sourcing `/opt/ros/jazzy/setup.bash`; ROS setup scripts may read optional unset environment variables.

Fix:
- Temporarily disable nounset around the ROS setup source step, then re-enable it for the rest of the script.

Verification:
- Added a regression test requiring the launch script to source ROS with nounset disabled.

## 2026-06-25 - Gazebo Driver-Style Keyboard Adapter

Scope:
- Added a Gazebo simulation adapter that mirrors the main RS232 driver velocity interface without opening serial or CAN hardware.

Implementation:
- Created `src/tunnel_nav/gazebo_control.py` with `GazeboCmdVelAdapter.set_velocity(linear_mps, angular_radps)` and driver-style helper methods: `forward`, `backward`, `turn_left`, `turn_right`, `arc_left`, `arc_right`, and `stop`.
- Created `tools/sim/gazebo_driver_keyboard.py`, a W/A/S/D keyboard controller that publishes Gazebo `/cmd_vel` directly through the adapter.

Verification:
- Added `tests/test_gazebo_cmd_vel_adapter.py` for command payload construction and helper-method mappings.

## 2026-06-25 - Gazebo Keyboard Deadman Stop

Scope:
- Investigated the simulation keyboard control continuing to move after `K` or `Space` and the vehicle sliding while a forward command remained active.

Root Cause:
- `tools/sim/gazebo_driver_keyboard.py` latched the last nonzero command. If the terminal did not capture the stop key, or no key was pressed after `W`, it kept publishing the previous forward command.

Fix:
- Changed the keyboard controller to deadman behavior: any idle cycle or unrecognized key publishes zero velocity. Movement now requires holding or key-repeat of the requested command key.

Verification:
- Added a regression test requiring idle keyboard input to produce `(0.0, 0.0)`.


## 2026-06-25 - Gazebo Wheel Friction Direction Tuning

Scope:
- Investigated continued motion after stop and right-turn/side-slip behavior when commanding straight forward motion in Gazebo.

Evidence:
- A stale `gazebo_driver_keyboard.py` process was still running after the keyboard script was edited, so the running process did not have the new deadman-stop behavior.
- Live odometry showed `linear.x` remained about `0.10 m/s` until an explicit zero `/cmd_vel` was published. After killing the stale keyboard process and publishing zero velocity, odometry twist returned to zero.
- A short `linear.x=0.04, angular.z=0.0` test showed the physical model pose still had small yaw and lateral drift, so straight-line instability was also present in the Gazebo contact model.

Fix:
- Added wheel `fdir1` rolling-friction direction to all four wheel collision ODE friction blocks in the Warthog SDF.
- Added conservative DiffDrive velocity and acceleration limits to reduce abrupt wheel commands while testing.

Verification:
- Added regression coverage requiring wheel collision friction direction and DiffDrive limits.
- Ran `python3 -m unittest tests/test_warthog_gazebo_model.py -v`.
- Ran `python3 -m unittest tests/test_gazebo_cmd_vel_adapter.py -v`.
- Ran `python3 -m py_compile src/tunnel_nav/gazebo_control.py tools/sim/gazebo_driver_keyboard.py`.
- Ran `gz sdf -k sim/gazebo/worlds/warthog_flat_test.sdf`; Gazebo reported `Valid.`

Operational Note:
- Existing Gazebo sessions must be restarted to load the updated SDF model.


## 2026-06-25 - Gazebo Wheel Axis Direction Fix

Scope:
- Followed up on straight-drive testing after wheel friction tuning revealed positive `linear.x` still moved the model in the wrong direction.

Root Cause:
- The SDF wheel links used a positive `+1.570796` roll and wheel joint axis `0 1 0`, while Gazebo's DiffDrive examples use wheel links rolled `-1.570796` with the wheel axis expressed as the link-frame `0 0 1`. The previous setup could move, but it did not match the DiffDrive convention.

Fix:
- Changed all four Gazebo SDF wheel link poses to `-1.57079632679 0 0`.
- Changed all four Gazebo SDF wheel joint axes to `0 0 1`.
- Left the URDF joint axes unchanged because URDF axis semantics are parent-frame based in the current wrapper.

Verification:
- Added regression coverage for the Gazebo SDF wheel pose and axis convention.
- Ran `python3 -m unittest tests/test_warthog_gazebo_model.py -v`.
- Ran `python3 -m unittest tests/test_gazebo_cmd_vel_adapter.py -v`.
- Ran `gz sdf -k sim/gazebo/worlds/warthog_flat_test.sdf`; Gazebo reported `Valid.`
- Ran a headless Gazebo motion test with `linear.x=0.04, angular.z=0.0` for 1.5 s. Result: `dx=0.061540`, `dy=0.000000`, `dyaw=0.000000`, and stop command cleared odometry twist.


## 2026-06-25 - Gazebo Keyboard Input Buffer Fix

Scope:
- Investigated keyboard control where `W` could drive straight, but releasing `W`, pressing `K`/`Space`, or pressing `Q` did not reliably stop or exit.

Root Cause:
- The keyboard controller read only one terminal character per loop. Holding `W` can generate repeated buffered `w` characters faster than the loop consumes them, so release/stop/quit keys may sit behind stale motion keys.

Fix:
- Added per-loop input draining in `tools/sim/gazebo_driver_keyboard.py`.
- Added control-key selection so `Q` takes priority over all buffered motion, `Space`/`K` take priority over buffered motion, and motion uses only the latest buffered motion key.

Verification:
- Added regression coverage for buffered `W` plus stop/quit keys.
- Ran `python3 -m unittest tests/test_gazebo_cmd_vel_adapter.py -v`.
- Ran `python3 -m unittest tests/test_warthog_gazebo_model.py -v`.
- Ran `python3 -m py_compile tools/sim/gazebo_driver_keyboard.py src/tunnel_nav/gazebo_control.py`.
- Stopped the stale keyboard controller process and published a zero `/cmd_vel`; live odometry twist returned to zero.


## 2026-06-25 - New RS232 Soft-Control GUI

Scope:
- Added a new RS232 GUI instead of modifying the original CAN GUI or legacy RS232 driver files.
- Addressed manual-control safety concerns observed with the physical remote: strong initial response from small joystick input and possible reverse kick when returning to center.

Implementation:
- Created `src/tunnel_nav/manual_control.py` with deadzone, curved joystick response, and slew-rate limiting.
- Created `tools/robot/rs232_vehicle_gui.py`, a PyQt5 GUI for `/dev/ttyUSB0` RS232 / Modbus RTU control.
- The GUI imports the unmodified `1/driver_controller.py` at runtime and wraps it with low-speed manual control.
- Default manual limits are conservative: max linear `0.04 m/s`, max angular `0.12 rad/s`, deadzone `0.18`, linear acceleration `0.08 m/s^2`, angular acceleration `0.25 rad/s^2`.
- Normal joystick/key release and the Soft Stop button ramp toward zero instead of using emergency stop. The red Emergency Stop button remains available for immediate emergency-stop register writes.

Verification:
- Ran `python3 -m unittest tests/test_manual_control.py -v`.
- Ran `python3 -m unittest tests/test_rs232_keyboard_drive.py -v`.
- Ran `/usr/bin/python3 -m py_compile tools/robot/rs232_vehicle_gui.py tools/robot/rs232_keyboard_drive.py src/tunnel_nav/manual_control.py`.
- Ran `/usr/bin/python3 tools/robot/rs232_vehicle_gui.py --help`.
- Ran an offscreen PyQt smoke test instantiating the new GUI without connecting hardware.

Operational Note:
- Run with `/usr/bin/python3` because system Python has PyQt5 and pyserial available.
- Do not use the CAN GUI for the current CH340 `/dev/ttyUSB0` RS232 adapter.


## 2026-06-25 - VSCode Conda PySerial Fix

Scope:
- Fixed the RS232 GUI `Connect failed: No module named 'serial'` issue when launched from VSCode.

Root Cause:
- VSCode was launching the GUI with `/home/tomato/miniconda3/bin/python3`, which had PyQt5 but did not have `pyserial`. System Python `/usr/bin/python3` already had `pyserial 3.5`.

Fix:
- Installed `pyserial 3.5` into the active conda Python environment used by VSCode.

Verification:
- Verified `import serial` from `/home/tomato/miniconda3/bin/python3`.
- Ran an offscreen smoke test instantiating `tools.robot.rs232_vehicle_gui.MainWindow` with the conda Python.


## 2026-06-25 - RGB Camera Detection Check

Scope:
- Checked whether the vehicle's built-in RGB camera data is visible from the current laptop after RS232 control was brought up.

Findings:
- Linux currently exposes `/dev/video0` through `/dev/video3`, but all nodes map to `Integrated_Webcam_HD`, the laptop's internal webcam.
- `lsusb` shows `Microdia Integrated_Webcam_HD` and the CH340 serial converter, but no separate vehicle RGB camera USB device.
- No ROS 2 topics matching camera/image/rgb/video were present.
- Captured one frame from `/dev/video0` successfully with ffmpeg, confirming the local V4L2 capture path works for the built-in laptop webcam.
- The active Python environment does not currently have `cv2` installed.

Conclusion:
- The vehicle RGB camera is not currently visible to this laptop through USB/V4L2 or ROS. RS232 only carries chassis control, not video. A separate USB camera connection, Ethernet/IP camera stream, or ROS camera publisher from the vehicle computer is required.


## 2026-06-25 - RS232 GUI In-Place Turn Lag Tuning

Scope:
- Addressed observed lag/stiction when commanding in-place left/right turns with the RS232 GUI.

Root Cause Hypothesis:
- The GUI's angular acceleration default was conservative (`0.25 rad/s^2`), so at a 100 ms control interval the angular command increased by only about `0.025 rad/s` each step. For a heavy chassis, this can sit below static friction / controller deadband before rotation starts.

Fix:
- Added `min_turn_start_radps` to `src/tunnel_nav/manual_control.py`. It applies only to in-place turns after the turn axis leaves the deadzone; it does not create turning at joystick center and does not boost turning while driving forward/backward.
- Added a GUI `Turn start` control, default `0.06 rad/s`.
- Increased the GUI default angular acceleration to `0.60 rad/s^2` while keeping soft ramping and soft stop.

Verification:
- Added regression tests covering in-place turn start boost and confirming it does not affect centered input or driving arcs.
- Ran `python3 -m unittest tests/test_manual_control.py tests/test_rs232_keyboard_drive.py -v`.
- Ran py_compile for `tools/robot/rs232_vehicle_gui.py` and `src/tunnel_nav/manual_control.py` with the VSCode conda Python.
- Ran an offscreen PyQt smoke test confirming default `Turn start = 0.06` and `Angular accel = 0.60`.


## 2026-06-25 - Laptop Webcam Road Segmentation Smoke Test

Scope:
- Tested whether the existing passable-road segmentation models can run on a frame captured from the laptop's built-in webcam.

Findings:
- The active VSCode/base Python lacks `torch` and `cv2`, but the `lerobot` conda environment has `torch 2.9.0+cu128` and `cv2 4.12.0`.
- Captured one frame from `/dev/video0` to `runs/camera_tests/laptop_webcam/raw/laptop_webcam_001.jpg`.
- Ran the existing fused passable/boundary inference using:
  - `runs/passable_ego/passable_ditch_artifact_v3_finetune/best_model.pt`
  - `runs/passable_ego/boundary_wall_aux_v2_no_testvideo/best_model.pt`
- Wrote overlay to `runs/camera_tests/laptop_webcam/overlays/laptop_webcam_001_fused_overlay.jpg` and masks under `runs/camera_tests/laptop_webcam/masks/`.

Note:
- This validates the inference path on a local webcam frame, but laptop webcam viewpoint is not equivalent to the vehicle front camera. Use it as a software smoke test, not model-quality validation.

## 2026-06-25 - Live Laptop Webcam Road Segmentation Preview

Scope:
- Added a live OpenCV preview tool for testing passable-road segmentation from the laptop's built-in webcam while the laptop is placed on the vehicle.

Implementation:
- Created `tools/passable_segmentation/live_webcam_fused_preview.py`.
- The tool opens `/dev/video0` by default, loads the existing passable/ditch model plus boundary/wall auxiliary model, runs fused inference on live frames, and displays raw camera view next to the segmentation overlay.
- Added CLI controls for camera path/index, capture size/FPS, display width, CPU forcing, frame-skip inference, and optional save of the final overlay.

Verification:
- Added `tests/test_live_webcam_fused_preview.py` covering camera argument parsing, parser defaults, and display scaling.
- Ran `python3 -m unittest tests/test_live_webcam_fused_preview.py -v`.
- Ran `python3 -m py_compile tools/passable_segmentation/live_webcam_fused_preview.py`.
- Ran `python3 tools/passable_segmentation/live_webcam_fused_preview.py --help`.
- Verified with `/home/tomato/miniconda3/envs/lerobot/bin/python` that runtime imports work and both default checkpoint files exist.

Operational Note:
- Run with the `lerobot` conda environment because it has `torch` and `cv2`.
- This preview only visualizes perception output; it does not send any drive command to the vehicle.

## 2026-06-25 - Live Webcam Preview Display Backend Fix

Scope:
- Fixed the live laptop webcam segmentation preview crash caused by OpenCV HighGUI being unavailable in the `lerobot` environment.

Root Cause:
- The `lerobot` environment has `cv2 4.12.0` with `GUI: NONE`, so `cv2.namedWindow` / `cv2.imshow` are not implemented.
- The same environment does have `tkinter` and Pillow/ImageTk available.

Fix:
- Added `--display-backend {tk,opencv}` to `tools/passable_segmentation/live_webcam_fused_preview.py`.
- Changed the default display backend to `tk`, using Tk/Pillow for the preview window while keeping OpenCV for camera capture, resizing, text drawing, and color conversion.
- Kept `--display-backend opencv` available for environments with GUI-enabled OpenCV builds.

Verification:
- Added a regression test confirming the parser defaults to `display_backend = tk`.
- Ran `python3 -m unittest tests/test_live_webcam_fused_preview.py -v`.
- Ran `python3 -m py_compile tools/passable_segmentation/live_webcam_fused_preview.py`.
- Ran `python3 tools/passable_segmentation/live_webcam_fused_preview.py --help`.
- Verified in `/home/tomato/miniconda3/envs/lerobot/bin/python` that `tkinter`, Pillow/ImageTk, `torch`, and `cv2` import successfully.

## 2026-06-25 - Vision-Gated Autonomous Forward Drive

Scope:
- Added a conservative autonomous forward-driving prototype that connects live road segmentation to the RS232 vehicle controller.

Implementation:
- Created `src/tunnel_nav/vision_autodrive.py` with pure drive-gate logic over the fused segmentation masks.
- Created `tools/robot/vision_autodrive_forward.py` to run laptop webcam inference, evaluate a near-center drive ROI, and send low-speed straight RS232 commands only when the ROI is clear.
- Default behavior is visual/zero-speed unless `--enable-driver` is passed. With `--enable-driver`, the script sends low-speed forward commands when safe and stop commands otherwise.
- The finalizer sends stop and disables the driver on window close, `q`/Esc, Ctrl+C, or exceptions.

Default Safety Parameters:
- Forward speed: `0.015 m/s`.
- Drive ROI: center-bottom image region, x `0.35-0.65`, y `0.60-0.95`.
- Minimum safe-passable ratio: `0.65`.
- Maximum hazard ratio: `0.02` for ditch, tunnel wall, or left barrier in the drive ROI.

Verification:
- Added `tests/test_vision_autodrive.py` for clear/stop ROI decisions and ROI bounds.
- Added `tests/test_vision_autodrive_forward_script.py` for script parser defaults.
- Ran `python3 -m unittest tests/test_vision_autodrive.py tests/test_vision_autodrive_forward_script.py -v`.
- Ran `python3 -m py_compile src/tunnel_nav/vision_autodrive.py tools/robot/vision_autodrive_forward.py`.
- Ran `python3 tools/robot/vision_autodrive_forward.py --help`.
- Verified with `/home/tomato/miniconda3/envs/lerobot/bin/python` that the script imports with torch, cv2, Tk/Pillow display support, and pyserial available.

Operational Note:
- Start with the physical remote/emergency stop ready. The script is a straight, low-speed gate; it does not steer around obstacles.

## 2026-06-25 - Vision Trajectory Steering Prototype

Scope:
- Added a low-speed trajectory-following prototype that derives an image-space centerline from `safe_passable` segmentation and sends differential-drive `linear/angular` commands over the existing RS232 driver.

Implementation:
- Created `src/tunnel_nav/vision_trajectory.py` for pure centerline extraction and steering command generation.
- Created `tools/robot/vision_autodrive_trajectory.py` for live webcam inference, fused-mask trajectory drawing, and RS232 command output.
- The script draws the planned yellow centerline, the lookahead target point, and the safety ROI in the preview window.
- Default behavior is visual/zero-speed unless `--enable-driver` is explicitly passed.

Default Safety Parameters:
- Forward speed: `0.012 m/s`.
- Max angular speed: `0.08 rad/s`.
- Angular gain: `0.18`.
- Center deadband: `0.04` normalized image width.
- Minimum path points: `3`.
- It stops on hazard ROI, missing path, camera read failure, window close, `q`/Esc, Ctrl+C, or exceptions.

Verification:
- Ran `python3 -m unittest tests/test_vision_autodrive.py tests/test_vision_trajectory.py tests/test_vision_autodrive_forward_script.py tests/test_vision_autodrive_trajectory_script.py tests/test_live_webcam_fused_preview.py -v` with 19 tests passing.
- Ran `python3 -m py_compile` on the related vision/autodrive modules and scripts.
- Verified with `/home/tomato/miniconda3/envs/lerobot/bin/python` that trajectory and forward autodrive scripts import with torch, cv2, Tk/Pillow, and serial dependencies available.

Operational Note:
- This is still image-space steering without camera calibration, BEV, odometry, or DWA. Use it only for very low-speed bring-up with the physical remote/emergency stop ready.

