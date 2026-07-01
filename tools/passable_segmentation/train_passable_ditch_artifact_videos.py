#!/usr/bin/env python3
"""Train passable-road and ditch segmentation for video multitask fine-tuning."""
from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.passable_segmentation.train_passable_ditch_artifact import (
    build_train_config,
    run_training,
)


def build_video_train_config() -> dict:
    """Build the video fine-tuning config for passable ditch artifact training."""
    config = build_train_config()
    config.update(
        {
            "dataset_dir": "data/derived/passable_boundary_obstacle_2026-06-29/passable",
            "run_dir": "runs/passable_ego/passable_ditch_artifact_videos_2026-06-29",
            "init_checkpoint": "runs/passable_ego/passable_ditch_artifact_v3_finetune/best_model.pt",
            "epochs": 50,
            "seed": 41,
        }
    )
    return config


def main() -> None:
    run_training(build_video_train_config())


if __name__ == "__main__":
    main()
