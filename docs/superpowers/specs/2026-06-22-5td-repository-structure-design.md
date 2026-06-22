# 5TD Repository Structure Design

Date: 2026-06-22

## Purpose

This repository should preserve the previous tunnel-UGV obstacle-avoidance code while giving the RGB-only engineering route and the LiDAR-RGB fusion, Transformer, RL, diffusion, and open-vocabulary research routes clean places to grow. The target shape is a monorepo with three explicit tracks:

- an active RGB-only engineering baseline based on the existing HBD-Net-RT and pure-vision code;
- an enhanced LiDAR-RGB fusion and Transformer research track;
- learning-augmented navigation modules that can later be promoted into the engineering system.

The updated project framing is dual-route safety-aware navigation. The RGB-only route remains viable for the post-civil tunnel stage where heavy construction dust is not expected to dominate. LiDAR, RGB fusion, Transformer modules, RL, and diffusion remain valuable as safety upgrades and research extensions, but any learned policy or generated trajectory must still pass through a costmap, planner, and safety filter.

## Current State

The current workspace has project planning documents under `docs/project_evaluation/` and previous code still in its extracted archive layout:

- `docs/project_evaluation/tunnel_ugv_lidar_rgb_transformer_navigation_plan.md`: LiDAR + RGB + Transformer fusion plan, with RL/diffusion as safety-filtered trajectory assistants.
- `docs/project_evaluation/2026-06-22-rgb-vision-route-addendum.md`: project-owner feedback that the RGB pure-vision route can continue because heavy dust is mainly a civil-construction issue.
- `docs/project_evaluation/tunnel_ugv_rl_navigation_project_evaluation.md`: earlier broader evaluation of RL/RL-assisted tunnel navigation.
- `vision_obstacle_avoidance.zip`: original large archive.
- `vision_obstacle_avoidance/home/nickwang/Projects/vision_obstacle_avoidance/`: extracted previous code with the original machine path embedded in the folder hierarchy.

Inside the extracted code there are two useful baselines:

- `hbdnet_rt/`: a modular HBD-Net-RT system with configs, tests, scripts, and documentation.
- `vision_obstacle_avoidance/`: an earlier pure-vision obstacle-avoidance stack with rule-based perception, lane/boundary logic, path planning, control placeholders, voice tools, and deployment files.

The repository has been initialized locally on branch `main` and has `origin` set to `https://github.com/SolanumLycopersicumX/5TD.git`. The remote URL intentionally does not contain a token.

## Target Structure

```text
5TD/
  README.md
  .gitignore

  docs/
    project_evaluation/
      tunnel_ugv_lidar_rgb_transformer_navigation_plan.md
      2026-06-22-rgb-vision-route-addendum.md
      tunnel_ugv_rl_navigation_project_evaluation.md
    architecture/
      system_overview.md
      safety_constraints.md
      roadmap.md
    progress/
      LOG.md
    legacy/
      old_code_notes.md
    superpowers/
      specs/
        2026-06-22-5td-repository-structure-design.md

  baselines/
    hbdnet_rt/
      configs/
      docs/
      scripts/
      src/
      tests/
      requirements.txt
      README.md
    vision_obstacle_avoidance_legacy/
      src/
      docs/
      tools/
      deployment/
      requirements.txt
      README.md

  research/
    rgb_vision/
      segmentation/
      hard_boundary/
      temporal_consistency/
      train.py
      evaluate.py
    transformer_fusion/
      lidar_bev/
      rgb_semantics/
      cross_attention/
      temporal_context/
      train.py
      evaluate.py
    rl_navigation/
      envs/
      policies/
      rewards/
      train.py
      evaluate.py
    diffusion_planner/
      models/
      trajectory_sampler/
      train.py
      infer.py
    vlm_supervisor/
      prompts/
      perception_queries/
      supervisor.py

  src/
    tunnel_nav/
      common/
      sensors/
      calibration/
      perception/
      mapping/
      planning/
      safety/
      control/
      evaluation/
      ros2_interfaces/

  configs/
    robot/
      vehicle.yaml
      sensors.yaml
    calibration/
      lidar_camera_extrinsics.yaml
      imu_lidar.yaml
    navigation/
      rgb_only.yaml
      fusion.yaml
      perception.yaml
      planner.yaml
      safety.yaml
    experiments/
      rl_default.yaml
      diffusion_default.yaml

  data/
    README.md
    annotations/
    samples/
    rosbags/
    raw/
    processed/

  experiments/
    README.md
    reports/
    runs/

  scripts/
    run_baseline.py
    run_research_eval.py
    prepare_data.py

  deployment/
    docker/
    edge_device/
    systemd/

  tools/
    annotation/
    calibration/
    visualization/
```

## Directory Responsibilities

`baselines/hbdnet_rt/` is the primary runnable RGB-only engineering baseline. It keeps its existing internal layout because it already has clear boundaries for perception, mapping, planning, safety, control, configs, scripts, tests, and docs. This route remains active because the target tunnel stage is expected to be post-civil construction, where heavy dust is not the main operating condition.

`baselines/vision_obstacle_avoidance_legacy/` preserves the older pure-vision implementation. It should be usable for reference and comparison, and utilities may be reused, but new engineering work should generally happen in HBD-Net-RT or the future `src/tunnel_nav/` package.

`research/rgb_vision/` holds pure-RGB perception experiments that go beyond the preserved baseline: segmentation, hard-boundary detection, temporal consistency, and RGB-only evaluation under actual post-civil tunnel conditions.

`research/transformer_fusion/` holds the LiDAR-RGB fusion research track: LiDAR BEV/voxel features, RGB semantic/object features, cross-attention fusion, temporal context modeling, and ablation evaluation against LiDAR-only and RGB-only baselines. This is an enhanced research line, not a replacement for the RGB-only baseline.

`research/rl_navigation/` holds reinforcement-learning environments, reward definitions, policies, training entrypoints, and evaluation scripts. It should depend on shared interfaces or adapters rather than importing directly from legacy baseline internals.

`research/diffusion_planner/` holds trajectory generation experiments. Diffusion output is treated as trajectory proposals that must still pass safety checks before becoming control commands.

`research/vlm_supervisor/` holds large-model or open-vocabulary supervision experiments, mainly for offline annotation assistance, scene-risk explanation, and low-frequency supervisory decisions. This layer should not own low-level real-time control.

`src/tunnel_nav/` is the future integrated engineering package. Code should move here only after an interface is stable enough to be reused outside a single experiment. Its long-term responsibilities include sensor adapters, LiDAR-camera calibration, perception, mapping, planning, safety, control, evaluation, and ROS 2 interfaces.

`configs/` stores shared robot, sensor, navigation, and experiment parameters. Baseline-specific configs can remain inside each baseline until they are promoted to shared configuration.

`data/` contains project-level shared data notes and future shared datasets. The user requested full project-asset upload, so large raw videos, LiDAR point-cloud logs, rosbag2 recordings, sensor logs, datasets, generated datasets, and processed training outputs should be committed through Git LFS when they belong to the project.

`experiments/` contains run reports and metadata. Experiment artifacts that are part of the project record may be committed through Git LFS when needed; transient scratch output should be kept out of the repository unless explicitly requested.

`docs/progress/LOG.md` is the project progress log. It records date, changed scope, decisions, verification status, and next actions. It should be updated after repository-structure changes, baseline test runs, research experiments, and field-evaluation milestones.

`deployment/` contains production packaging and edge-device deployment material after it is separated from legacy code.

`tools/` contains reusable one-off utilities for annotation, calibration, and visualization.

## Git and Data Policy

The repository should track all project assets requested by the user: source code, configs, documentation, annotations, datasets, videos, model weights, archives, and legacy project metadata.

Large or binary project assets must be tracked with Git LFS to avoid GitHub's normal file-size limit. The repository uses `.gitattributes` for archives, model weights, videos, databases, rosbags, numpy arrays, pickles, tarballs, and similar binary artifacts.

The repository should still avoid committing credentials, GitHub tokens, API keys, private keys, or newly generated machine-local secrets. Existing legacy project metadata is preserved when it is not credential material. The leaked GitHub token from the conversation should be revoked in GitHub and replaced with a new token if authenticated push is needed.

## Migration Plan

1. Add `.gitignore`, root `README.md`, and data/experiment README placeholders.
2. Keep `docs/progress/LOG.md` updated for each major repository, baseline, and research milestone.
3. Keep the LiDAR-RGB-Transformer plan, RGB pure-vision addendum, and earlier RL navigation evaluation under `docs/project_evaluation/`.
4. Move extracted `hbdnet_rt/` into `baselines/hbdnet_rt/` while preserving its internal layout as the active RGB-only baseline.
5. Move the older `vision_obstacle_avoidance` Python package and its related helper scripts/docs into `baselines/vision_obstacle_avoidance_legacy/`.
6. Remove the embedded `home/nickwang/Projects/` directory nesting from the repository layout.
7. Move the original large zip into `archive/original/` and track it with Git LFS.
8. Preserve legacy data, datasets, models, videos, logs, caches, and metadata under `baselines/vision_obstacle_avoidance_legacy/`, using Git LFS for large binaries.
9. Create the `research/rgb_vision/` skeleton for pure-RGB experiments that extend the baseline.
10. Create the `research/transformer_fusion/` skeleton for LiDAR-RGB fusion and temporal Transformer experiments.
11. Create RL/diffusion/VLM skeletons only where the next implementation plan needs them.
12. Run baseline tests after moving files and update import paths only if tests or entrypoints require it.

## Acceptance Criteria

- The repository root has a clear README explaining the dual-route structure: active RGB-only baseline plus enhanced LiDAR-RGB/Transformer research.
- Previous code remains available under `baselines/` and is not deleted.
- The current LiDAR-RGB-Transformer project plan and RGB pure-vision addendum are tracked under `docs/project_evaluation/`.
- The RGB-only route is represented as an active baseline, not just an archived legacy path.
- New RGB vision, Transformer fusion, RL, diffusion, and VLM/open-vocabulary research have separate directories under `research/`.
- Shared future engineering code has a dedicated `src/tunnel_nav/` package with room for sensors, calibration, ROS 2 interfaces, and safety-filtered navigation.
- The project has a maintained progress log at `docs/progress/LOG.md`.
- Large data, archives, model weights, videos, databases, and other binary project assets are tracked by Git LFS.
- The Git remote remains `https://github.com/SolanumLycopersicumX/5TD.git` without embedded credentials.
- No old absolute machine path is part of the final top-level project layout.

## Non-Goals

This structure change does not implement RL training, diffusion trajectory generation, VLM supervision, real robot control, ROS integration, or sensor fusion. It only creates the repository layout needed to support those workstreams without losing the previous baseline code.

## Open Assumptions

- HBD-Net-RT is the preferred engineering baseline because it already has modular tests and clearer interfaces.
- The older pure-vision package is retained primarily for comparison, documentation, and reuse of useful utilities.
- The project will initially prioritize offline development, evaluation, and simulation before any closed-loop deployment near the right-side trench.
- The target operating stage is expected to be after civil construction, so heavy dust should not be treated as the main blocker for RGB-only perception.
- RGB-only safety still needs validation for low light, reflections, water stains, repeating tunnel textures, lens contamination, and right-side trench visibility.
- The updated report treats “radar” as LiDAR rather than mmWave radar; if the hardware is mmWave, the sensor and perception plan must be revised.
- Transformer should be used first for RGB temporal consistency or LiDAR-RGB fusion and temporal context, not for direct raw motor control.
- A right-side LiDAR/ToF or equivalent dedicated trench-distance sensor remains a safety requirement to confirm if RGB-only trench-margin estimation is not reliable enough.
