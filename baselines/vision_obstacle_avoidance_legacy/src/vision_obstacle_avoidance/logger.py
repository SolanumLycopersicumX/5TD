"""
运行日志模块。
记录每帧的 FPS、延迟、感知状态、决策结果，支持 CSV 导出和视频保存。
"""

import os
import csv
import time
from datetime import datetime

import cv2
import config
from utils import (FreeSpaceState, ObstacleState, Decision,
    LaneBoundaryState, DebrisState, PathPlan)


class RuntimeLogger:
    """
    运行时日志记录器。

    输出:
      - logs/run_YYYYMMDD_HHMMSS.csv  每帧数据结构化日志
      - videos/run_YYYYMMDD_HHMMSS.avi  可选标注视频
    """

    def __init__(self):
        self._csv_file = None
        self._csv_writer = None
        self._video_writer = None
        self._output_dir = ""
        self._frame_count = 0
        self._session_start = time.time()
        self._closed = False

    # ---- 公开接口 ----

    def start(self):
        """开始记录。创建日志文件和视频输出。"""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._output_dir = ts

        if config.SAVE_LOG:
            os.makedirs(config.LOG_DIR, exist_ok=True)
            csv_path = os.path.join(config.LOG_DIR, f"run_{ts}.csv")
            self._csv_file = open(csv_path, "w", newline="", encoding="utf-8")
            self._csv_writer = csv.writer(self._csv_file)
            self._csv_writer.writerow([
                "frame", "timestamp", "fps", "latency_ms",
                "command", "speed", "steering", "reason", "confidence",
                "drive_state",
                "has_obstacle", "danger_level", "blocked_left",
                "blocked_center", "blocked_right", "closest_zone",
                "left_free", "center_free", "right_free",
                "center_offset", "free_valid",
                # Phase 1 新增字段
                "ditch_px", "left_barrier_px", "right_barrier_px",
                "lane_valid", "lane_width_m",
                "has_debris", "has_large_debris",
                "path_passable", "path_clearance_cm",
            ])
            print(f"[Logger] CSV: {csv_path}")

        if config.SAVE_VIDEO:
            os.makedirs(config.VIDEO_DIR, exist_ok=True)
            # Linux 兼容: MJPG 比 XVID 更通用，无需额外编码器
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            video_path = os.path.join(config.VIDEO_DIR, f"run_{ts}.avi")
            self._video_writer = cv2.VideoWriter(
                video_path, fourcc, 15.0, (config.FRAME_WIDTH, config.FRAME_HEIGHT)
            )
            print(f"[Logger] Video: {video_path}")

    def update(self, frame_count: int, fps: float, latency_ms: float,
               free_state: FreeSpaceState, obstacle_state: ObstacleState,
               decision: Decision, annotated_frame=None,
               lane_state=None, debris_state=None, path_plan=None):
        if self._closed:
            return
        self._frame_count = frame_count

        # CSV（空值保护）
        if self._csv_writer and decision is not None:
            try:
                fs = free_state or FreeSpaceState()
                obs = obstacle_state or ObstacleState()
                self._csv_writer.writerow([
                    frame_count,
                    f"{time.time():.3f}",
                    f"{fps:.1f}",
                    f"{latency_ms:.1f}",
                    decision.command,
                    f"{decision.speed:.2f}",
                    f"{decision.steering:.2f}",
                    decision.reason,
                    f"{decision.confidence:.2f}",
                    decision.drive_state,
                    int(obs.has_obstacle),
                    f"{obs.danger_level:.2f}",
                    int(obs.blocked_left),
                    int(obs.blocked_center),
                    int(obs.blocked_right),
                    obs.closest_obstacle_zone,
                    f"{fs.left_free_score:.3f}",
                    f"{fs.center_free_score:.3f}",
                    f"{fs.right_free_score:.3f}",
                    f"{fs.center_offset:.3f}",
                    int(fs.is_valid),
                    # Phase 1 新增
                    lane_state.ditch_px if lane_state else "",
                    lane_state.left_barrier_px if lane_state else "",
                    lane_state.right_barrier_px if lane_state else "",
                    int(lane_state.is_valid) if lane_state else "",
                    f"{lane_state.lane_width_m:.2f}" if lane_state else "",
                    int(debris_state.has_debris) if debris_state else "",
                    int(debris_state.has_large_debris) if debris_state else "",
                    int(path_plan.passable) if path_plan else "",
                    f"{path_plan.clearance_cm:.1f}" if path_plan else "",
                ])
            except (OSError, IOError) as e:
                print(f"[Logger] CSV 写入失败: {e}")

        # 视频
        if self._video_writer and annotated_frame is not None:
            try:
                if annotated_frame.shape[1] != config.FRAME_WIDTH or \
                   annotated_frame.shape[0] != config.FRAME_HEIGHT:
                    annotated_frame = cv2.resize(
                        annotated_frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT)
                    )
                self._video_writer.write(annotated_frame)
            except cv2.error as e:
                print(f"[Logger] 视频写入失败: {e}")

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self._csv_file:
            try:
                self._csv_file.close()
                print(f"[Logger] CSV 已保存 ({self._frame_count} 行)")
            except (OSError, IOError) as e:
                print(f"[Logger] CSV 关闭失败: {e}")
        if self._video_writer:
            self._video_writer.release()
            print(f"[Logger] 视频已保存 ({self._frame_count} 帧)")
        elapsed = time.time() - self._session_start
        print(f"[Logger] 会话时长: {elapsed:.1f}s, 平均 FPS: {self._frame_count / max(elapsed, 1e-5):.1f}")
