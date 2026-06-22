# 5TD Tunnel UGV Navigation

5TD is a tunnel UGV navigation project for safety-aware obstacle avoidance and local navigation in a post-civil-construction tunnel environment.

The repository keeps two practical tracks:

- **RGB-only engineering route:** `baselines/hbdnet_rt/` is the active near-term MVP baseline. It keeps the pure-vision perception, mapping, planning, and safety-state-machine work runnable and testable.
- **Enhanced research route:** `research/` contains RGB-only experiments, LiDAR-RGB Transformer fusion, RL navigation, diffusion trajectory proposals, and VLM/open-vocabulary supervision. These modules should propose perception outputs, risk maps, modes, or candidate trajectories, not bypass the safety filter.

Large assets are tracked with Git LFS so the original archive, model weights, videos, and databases can be uploaded to GitHub.

## Layout

```text
archive/original/                         Original imported archive
baselines/hbdnet_rt/                      Active RGB-only engineering baseline
baselines/vision_obstacle_avoidance_legacy/  Older pure-vision project and all imported assets
configs/                                  Shared robot, calibration, navigation, and experiment configs
data/                                     Project-level data notes and future shared datasets
deployment/                              Deployment notes and packaging area
docs/project_evaluation/                 Project evaluation and route-decision documents
docs/progress/LOG.md                     Project progress log
experiments/                             Experiment reports and run metadata
research/                                New research tracks
src/tunnel_nav/                          Future integrated engineering package
tools/                                   Shared annotation, calibration, and visualization tools
```

## Current Priority

1. Keep the RGB-only baseline runnable.
2. Evaluate RGB-only safety under actual tunnel conditions.
3. Add right-side distance sensing or LiDAR-RGB fusion if RGB-only trench-margin detection is not sufficient.
4. Add RL, diffusion, and VLM modules only behind costmap/planner/safety-filter interfaces.

## Git LFS

This repository uses Git LFS for large assets. Install Git LFS before cloning or pushing large binary changes:

```bash
git lfs install
git lfs pull
```
