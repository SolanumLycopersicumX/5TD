#!/usr/bin/env python3
"""
测试场景分析工具。
将测试视频逐帧跑过 pipeline，输出结构化 JSON 报告。
Claude Code 可以直接读取这个 JSON 文件进行分析。
"""

import json
import sys
import time
from pathlib import Path

import cv2
import config

from camera_capture import CameraCapture
from preprocess import ImagePreprocessor
from lane_or_freespace_detector import FreeSpaceDetector
from obstacle_detector import ObstacleDetector
from decision_maker import DecisionMaker
from logger import RuntimeLogger
from utils import Decision, FreeSpaceState, ObstacleState


def analyze_video(video_path: str):
    """逐帧分析视频，生成 JSON 报告。"""

    camera = CameraCapture(camera_index=video_path)
    if not camera.is_opened():
        print(f"错误：无法打开 {video_path}")
        return

    preprocessor = ImagePreprocessor()
    free_detector = FreeSpaceDetector()
    obs_detector = ObstacleDetector()
    decider = DecisionMaker()

    frames_data = []
    warnings = []
    frame_count = 0
    total_start = time.time()

    print(f"分析中: {video_path} ...")

    while True:
        frame, ts = camera.read()
        if frame is None:
            break
        frame_count += 1
        t0 = time.time()

        # 跑完整 pipeline
        preprocessed = preprocessor.process(frame)
        free_state = free_detector.detect(
            preprocessed.roi_frame,
            preprocessed.edges,
            preprocessed.binary_mask,
        )
        obs_state = obs_detector.detect(
            preprocessed.roi_frame,
            preprocessed.edges,
            preprocessed.binary_mask,
        )
        latency_ms = (time.time() - t0) * 1000
        decision = decider.decide(free_state, obs_state, latency_ms, 15.0)

        # 收集每帧数据
        frame_info = {
            "frame": frame_count,
            "latency_ms": round(latency_ms, 1),
            "decision": decision.command,
            "reason": decision.reason,
            "danger_level": round(obs_state.danger_level, 2),
            "blocked": {
                "left": obs_state.blocked_left,
                "center": obs_state.blocked_center,
                "right": obs_state.blocked_right,
            },
            "free_scores": {
                "left": round(free_state.left_free_score, 2),
                "center": round(free_state.center_free_score, 2),
                "right": round(free_state.right_free_score, 2),
            },
            "center_offset": round(free_state.center_offset, 3),
            "obstacle_count": len(obs_state.obstacle_boxes),
            "obstacle_max_area": int(obs_state.largest_obstacle_area),
        }
        frames_data.append(frame_info)

        # 收集异常帧
        if obs_state.danger_level > 0.6:
            warnings.append({
                "type": "high_danger",
                "frame": frame_count,
                "danger": round(obs_state.danger_level, 2),
                "closest_zone": obs_state.closest_obstacle_zone,
            })

        if not free_state.is_valid:
            warnings.append({
                "type": "invalid_freespace",
                "frame": frame_count,
                "confidence": round(free_state.confidence, 2),
            })

        if decision.command in ("STOP", "SLOW_DOWN"):
            warnings.append({
                "type": "speed_change",
                "frame": frame_count,
                "command": decision.command,
                "reason": decision.reason,
            })

    # 统计摘要
    total_time = time.time() - total_start
    commands = {}
    for f in frames_data:
        cmd = f["decision"]
        commands[cmd] = commands.get(cmd, 0) + 1

    report = {
        "video": video_path,
        "total_frames": frame_count,
        "total_time_sec": round(total_time, 1),
        "avg_fps": round(frame_count / max(total_time, 0.01), 1),
        "avg_latency_ms": round(
            sum(f["latency_ms"] for f in frames_data) / max(frame_count, 1), 1
        ),
        "command_distribution": commands,
        "danger_frames": sum(
            1 for f in frames_data if f["danger_level"] > 0.5
        ),
        "invalid_freespace_frames": sum(
            1 for f in frames_data
            if f["free_scores"]["center"] < 0.2
        ),
        "warnings": warnings,
        "frame_details": frames_data,
    }

    # 写 JSON
    output_path = Path(video_path).stem + "_analysis.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"报告已生成: {output_path}")
    print(f"  {frame_count} 帧, {len(warnings)} 个警告")
    print(f"  指令分布: {commands}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python analyze_test.py <视频文件>")
        sys.exit(1)
    analyze_video(sys.argv[1])
