#!/usr/bin/env python3
"""
全视频采样分析工具。
对长视频按指定帧率采样，逐帧跑 vision pipeline，生成完整时间线 JSON 报告。
"""

import json
import sys
import time
from pathlib import Path

import cv2
import config

from preprocess import ImagePreprocessor
from lane_or_freespace_detector import FreeSpaceDetector
from obstacle_detector import ObstacleDetector
from decision_maker import DecisionMaker
from utils import Decision, FreeSpaceState, ObstacleState


def analyze_video_sampled(video_path: str, sample_fps: float = 1.0):
    """
    采样分析视频。

    Args:
        video_path: 视频文件路径
        sample_fps: 采样帧率（每秒取几帧），默认1fps
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误：无法打开 {video_path}")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / max(video_fps, 1)

    frame_skip = max(1, int(video_fps / sample_fps))
    expected_samples = total_frames // frame_skip

    print(f"视频: {Path(video_path).name}")
    print(f"  {total_frames} 帧 @ {video_fps:.1f}fps, 时长 {duration:.1f}s")
    print(f"  采样率: 每 {frame_skip} 帧取1帧 → 预计 {expected_samples} 个样本 @ ~{sample_fps}fps")
    print(f"  采样间隔: {frame_skip/video_fps:.1f}s\n")

    preprocessor = ImagePreprocessor()
    free_detector = FreeSpaceDetector()
    obs_detector = ObstacleDetector()
    decider = DecisionMaker()

    samples = []
    warnings = []
    total_start = time.time()
    frame_idx = 0
    sample_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        # 跳帧采样
        if frame_idx % frame_skip != 0:
            frame_idx += 1
            continue

        sample_idx += 1
        t0 = time.time()
        video_time = frame_idx / video_fps


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

        # 结构化数据
        sample = {
            "idx": sample_idx,
            "frame": frame_idx,
            "time_sec": round(video_time, 1),
            "time_str": f"{int(video_time//60):02d}:{video_time%60:04.1f}",
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
        samples.append(sample)

        # 警告事件
        if obs_state.danger_level > 0.6:
            warnings.append({
                "type": "high_danger",
                "time_sec": round(video_time, 1),
                "time_str": sample["time_str"],
                "frame": frame_idx,
                "danger": round(obs_state.danger_level, 2),
                "closest_zone": obs_state.closest_obstacle_zone,
            })

        if decision.command in ("STOP", "SLOW_DOWN"):
            warnings.append({
                "type": "speed_change",
                "time_sec": round(video_time, 1),
                "time_str": sample["time_str"],
                "frame": frame_idx,
                "command": decision.command,
                "reason": decision.reason,
            })

        # 进度
        if sample_idx % 50 == 0 or sample_idx == 1:
            elapsed = time.time() - total_start
            eta = elapsed / sample_idx * expected_samples - elapsed
            ts = sample["time_str"]
            print(f"  [{ts}] 样本#{sample_idx}/{expected_samples} "
                  f"| 已用{elapsed:.0f}s | 预计剩余{eta:.0f}s "
                  f"| 决策:{decision.command} 危险:{obs_state.danger_level:.2f}")

        frame_idx += 1

    cap.release()
    total_time = time.time() - total_start

    # ── 统计摘要 ──
    cmds = {}
    for s in samples:
        cmd = s["decision"]
        cmds[cmd] = cmds.get(cmd, 0) + 1

    # 按时间段聚合
    window_sec = 10
    time_windows = {}
    for s in samples:
        w = int(s["time_sec"] // window_sec)
        if w not in time_windows:
            time_windows[w] = {
                "start_sec": w * window_sec,
                "start_str": f"{int(w*window_sec//60):02d}:{w*window_sec%60:02d}",
                "samples": 0,
                "avg_danger": 0.0,
                "max_danger": 0.0,
                "decisions": {},
                "avg_obstacles": 0.0,
            }
        tw = time_windows[w]
        tw["samples"] += 1
        tw["avg_danger"] += s["danger_level"]
        tw["max_danger"] = max(tw["max_danger"], s["danger_level"])
        tw["avg_obstacles"] += s["obstacle_count"]
        tw["decisions"][s["decision"]] = tw["decisions"].get(s["decision"], 0) + 1

    for tw in time_windows.values():
        tw["avg_danger"] = round(tw["avg_danger"] / max(tw["samples"], 1), 2)
        tw["avg_obstacles"] = round(tw["avg_obstacles"] / max(tw["samples"], 1), 1)

    # 危险高峰段（danger > 0.3 的时间段）
    danger_zones = []
    zone_start = None
    for s in samples:
        if s["danger_level"] > 0.3 and zone_start is None:
            zone_start = s
        elif s["danger_level"] <= 0.3 and zone_start is not None:
            danger_zones.append({
                "start": zone_start["time_str"],
                "end": s["time_str"],
                "start_sec": zone_start["time_sec"],
                "end_sec": s["time_sec"],
                "duration_sec": round(s["time_sec"] - zone_start["time_sec"], 1),
                "max_danger_in_zone": max(
                    x["danger_level"] for x in samples
                    if zone_start["time_sec"] <= x["time_sec"] <= s["time_sec"]
                ),
            })
            zone_start = None
    if zone_start:
        danger_zones.append({
            "start": zone_start["time_str"],
            "end": samples[-1]["time_str"],
            "start_sec": zone_start["time_sec"],
            "end_sec": samples[-1]["time_sec"],
            "duration_sec": round(samples[-1]["time_sec"] - zone_start["time_sec"], 1),
            "max_danger_in_zone": max(
                x["danger_level"] for x in samples
                if x["time_sec"] >= zone_start["time_sec"]
            ),
        })

    # ── 构建报告 ──
    report = {
        "video": video_path,
        "video_duration_sec": round(duration, 1),
        "video_fps": video_fps,
        "total_frames": total_frames,
        "sample_fps": sample_fps,
        "frame_skip": frame_skip,
        "total_samples": len(samples),
        "analysis_time_sec": round(total_time, 1),
        "avg_pipeline_latency_ms": round(
            sum(s["latency_ms"] for s in samples) / max(len(samples), 1), 1
        ),
        "command_distribution": cmds,
        "total_danger_zones": len(danger_zones),
        "danger_zones": danger_zones,
        "time_windows_10sec": [time_windows[w] for w in sorted(time_windows)],
        "samples": samples,
        "warnings": warnings,
    }

    # 写 JSON
    output_path = Path(video_path).stem + "_full_analysis.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"✅ 报告: {output_path}")
    print(f"   全视频 {duration:.0f}s → {len(samples)} 个样本 "
          f"({sample_fps}fps 采样), {len(warnings)} 个警告")
    print(f"   耗时: {total_time:.0f}s (pipeline 平均 {report['avg_pipeline_latency_ms']:.0f}ms/帧)")
    print(f"   指令分布: {cmds}")
    print(f"   危险区域: {len(danger_zones)} 段")

    # 快速摘要
    if danger_zones:
        print(f"\n⚠️  危险区域概览:")
        for dz in danger_zones:
            bar = '█' * max(1, int(dz['max_danger_in_zone'] * 20))
            print(f"   {dz['start']} → {dz['end']} "
                  f"({dz['duration_sec']:.0f}s) 峰值危险:{dz['max_danger_in_zone']:.2f} |{bar}|")

    return report


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python analyze_video_sampled.py <视频文件> [采样fps(默认1)]")
        sys.exit(1)
    video = sys.argv[1]
    fps = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    analyze_video_sampled(video, fps)
