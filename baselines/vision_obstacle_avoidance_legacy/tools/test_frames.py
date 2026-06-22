"""
快速测试脚本：用现有传统CV管线处理实地视频帧，看效果。
"""
import cv2
import os
import sys
import numpy as np

# 确保能找到项目模块
sys.path.insert(0, '/home/nickwang/Projects/vision_obstacle_avoidance/vision_obstacle_avoidance')

import config
from preprocess import ImagePreprocessor
from lane_boundary_detector import LaneBoundaryDetector
from lane_or_freespace_detector import FreeSpaceDetector
from obstacle_detector import ObstacleDetector
from debris_detector import DebrisDetector

frames_dir = '/home/nickwang/Projects/vision_obstacle_avoidance/data/frames/samples'
out_dir = '/home/nickwang/Projects/vision_obstacle_avoidance/data/frames/processed'
os.makedirs(out_dir, exist_ok=True)

# 初始化各模块
preprocessor = ImagePreprocessor()
lane_detector = LaneBoundaryDetector()
free_detector = FreeSpaceDetector()
obstacle_detector = ObstacleDetector()
debris_detector = DebrisDetector()

files = sorted(os.listdir(frames_dir))

for i, fname in enumerate(files):
    if i >= 5:  # 先测5张
        break

    path = os.path.join(frames_dir, fname)
    frame = cv2.imread(path)
    if frame is None:
        continue

    # 预处理
    preprocessed = preprocessor.process(frame)

    # 车道边界检测
    lane_state = lane_detector.detect(
        preprocessed.roi_frame, preprocessed.edges, preprocessed.enhanced, None
    )

    # 可行驶区域
    free_state = free_detector.detect(
        preprocessed.roi_frame, preprocessed.edges, preprocessed.binary_mask, lane_state
    )

    # 障碍检测
    obs_state = obstacle_detector.detect(
        preprocessed.roi_frame, preprocessed.edges, preprocessed.binary_mask
    )

    # 碎石检测
    debris_state = debris_detector.detect(
        preprocessed.gray, preprocessed.edges, lane_state, None
    )

    # 输出分析
    print(f"\n=== {fname} ===")
    print(f"  车道边界: valid={lane_state.is_valid}, "
          f"ditch={lane_state.ditch_px}, "
          f"left_barrier={lane_state.left_barrier_px}, "
          f"right_barrier={lane_state.right_barrier_px}, "
          f"lane_width={lane_state.lane_width_m:.2f}m")
    print(f"  可行驶区域: valid={free_state.is_valid}, "
          f"L={free_state.left_free_score:.2f} "
          f"C={free_state.center_free_score:.2f} "
          f"R={free_state.right_free_score:.2f}")
    print(f"  障碍: has_obs={obs_state.has_obstacle}, "
          f"danger={obs_state.danger_level:.2f}, "
          f"blocked: L={obs_state.blocked_left} C={obs_state.blocked_center} R={obs_state.blocked_right}")
    print(f"  碎石: has_debris={debris_state.has_debris}, "
          f"large={debris_state.has_large_debris}")

    # 保存调试画面
    h, w = frame.shape[:2]
    # 缩小到 1/4 拼 4 个图
    def resize(img, tw=320, th=240):
        if img is None:
            return 128 * (np.zeros((th, tw, 3), dtype=np.uint8) + 1).astype(np.uint8)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return cv2.resize(img, (tw, th))

    r1 = np.hstack([
        resize(frame, 320, 480),
        resize(preprocessed.edges, 320, 480),
    ])
    r2 = np.hstack([
        resize(lane_state.debug_image, 320, 480),
        resize(free_state.debug_image if free_state.debug_image is not None else None, 320, 480),
    ])
    combined = np.vstack([r1, r2])
    cv2.putText(combined, f"Lane: valid={lane_state.is_valid} w={lane_state.lane_width_m:.2f}m",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    cv2.putText(combined, f"Free: L={free_state.left_free_score:.2f} C={free_state.center_free_score:.2f} R={free_state.right_free_score:.2f}",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    cv2.putText(combined, f"Obs: {obs_state.has_obstacle} danger={obs_state.danger_level:.2f}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255) if obs_state.has_obstacle else (0,255,0), 1)

    out_path = os.path.join(out_dir, fname)
    cv2.imwrite(out_path, combined)

print(f"\n处理完成，结果保存在 {out_dir}/")
