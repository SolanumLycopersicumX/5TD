#!/usr/bin/env python3
"""
批量标注脚本 — 对提取的帧运行CV检测，叠加彩色标注。

颜色方案:
  绿色实线   — 隔离沟 (ditch)
  黄色实线   — 隔离带 (barrier)
  蓝色虚线   — 车道中心线
  红色框     — 障碍物 (obstacle)
  橙色框     — 碎石 (debris)
  青色半透明 — 可行驶区域 (free space)
  品红框     — 大块碎石 (large_debris, >15cm)

输出: data/annotations/labeled/  带标注叠加图
      data/annotations/labeled_report.json  标注统计
"""

import sys, os, cv2, json, time, argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import config
from preprocess import ImagePreprocessor
from lane_boundary_detector import LaneBoundaryDetector
from obstacle_detector import ObstacleDetector
from debris_detector import DebrisDetector
from calibration import GroundCalibrator
from lane_or_freespace_detector import FreeSpaceDetector


# ══════════════════════════════════════════════════════════════════════
# 标注绘制
# ══════════════════════════════════════════════════════════════════════

# 颜色定义 (BGR)
COLORS = {
    'ditch':          (0, 255, 0),    # 绿色 — 隔离沟
    'barrier_left':   (0, 255, 255),  # 黄色 — 左隔离带
    'barrier_right':  (0, 255, 255),  # 黄色 — 右隔离带
    'lane_center':    (255, 0, 0),    # 蓝色 — 车道中心线
    'obstacle':       (0, 0, 255),    # 红色 — 障碍物
    'debris':         (0, 140, 255),  # 橙色 — 碎石
    'large_debris':   (255, 0, 255),  # 品红 — 大块碎石(>15cm)
    'danger_zone':    (0, 0, 200),    # 深红 — 危险区
    'roi_boundary':   (200, 200, 200),# 灰色 — ROI边界
    'free_space_zone':(255, 255, 0),  # 青色 — 自由空间分区
}

# 图例
LEGEND = [
    ("隔离沟 Ditch",        COLORS['ditch']),
    ("隔离带 Barrier",       COLORS['barrier_left']),
    ("车道中心",             COLORS['lane_center']),
    ("障碍物 Obstacle",      COLORS['obstacle']),
    ("碎石 Debris",          COLORS['debris']),
    ("大块碎石(>15cm)",      COLORS['large_debris']),
    ("危险区",               COLORS['danger_zone']),
]


def draw_annotations(frame: np.ndarray,
                     preprocessed,
                     lane_state,
                     free_state,
                     obs_state,
                     debris_state,
                     roi_offset: tuple) -> np.ndarray:
    """
    在原始帧上叠加所有彩色标注。
    返回标注后的图像。
    """
    canvas = frame.copy()
    h, w = canvas.shape[:2]
    roi_x, roi_y = roi_offset
    roi_h = preprocessed.roi_frame.shape[0] if preprocessed.roi_frame is not None else 0
    roi_w = preprocessed.roi_frame.shape[1] if preprocessed.roi_frame is not None else 0

    # ── ROI边界（灰色虚线） ──
    cv2.rectangle(canvas, (roi_x, roi_y),
                  (roi_x + roi_w, roi_y + roi_h),
                  COLORS['roi_boundary'], 1)

    # ── 可行驶区域分区（半透明覆盖） ──
    if free_state is not None and roi_h > 0 and roi_w > 0:
        overlay = canvas.copy()
        alpha = 0.15

        if lane_state and lane_state.is_valid:
            lb = lane_state.left_barrier_px or 0
            db = lane_state.ditch_px or roi_w
            lw = db - lb
            z1_e = lb + lw // 3
            z2_s = db - lw // 3
            zones_roi = [
                (lb, 0, z1_e, roi_h, (255, 200, 100)),
                (z1_e, 0, z2_s, roi_h, (100, 255, 100)),
                (z2_s, 0, db, roi_h, (100, 200, 255)),
            ]
        else:
            lx = int(roi_w * config.ZONE_LEFT_RATIO)
            rx = int(roi_w * config.ZONE_RIGHT_RATIO)
            zones_roi = [
                (0, 0, lx, roi_h, (255, 200, 100)),
                (lx, 0, rx, roi_h, (100, 255, 100)),
                (rx, 0, roi_w, roi_h, (100, 200, 255)),
            ]

        for zx1, zy1, zx2, zy2, color in zones_roi:
            cv2.rectangle(overlay,
                         (roi_x + zx1, roi_y + zy1),
                         (roi_x + zx2, roi_y + zy2),
                         color, -1)
        canvas = cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0)

        # 分区标签 + 评分
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i, (zx1, zy1, zx2, zy2, _) in enumerate(zones_roi):
            label = ['L', 'C', 'R'][i]
            scores = [free_state.left_free_score,
                      free_state.center_free_score,
                      free_state.right_free_score]
            cx = roi_x + (zx1 + zx2) // 2 - 15
            cv2.putText(canvas, f"{label}:{scores[i]:.2f}",
                       (cx, roi_y + 20), font, 0.4, (255,255,255), 1)

    # ── 隔离沟（绿色粗线 + 标签） ──
    if lane_state and lane_state.ditch_px is not None:
        dx = roi_x + lane_state.ditch_px
        cv2.line(canvas, (dx, roi_y), (dx, roi_y + roi_h),
                COLORS['ditch'], 3)
        # 标签
        label_bg(canvas, "DITCH", dx + 5, roi_y + 25,
                COLORS['ditch'], font_scale=0.5)

    # ── 隔离带（黄色线 + 标签） ──
    for name, px, color_key in [
        ("L-BAR", lane_state.left_barrier_px if lane_state else None, 'barrier_left'),
        ("R-BAR", lane_state.right_barrier_px if lane_state else None, 'barrier_right'),
    ]:
        if px is not None:
            bx = roi_x + px
            cv2.line(canvas, (bx, roi_y), (bx, roi_y + roi_h),
                    COLORS[color_key], 2)
            label_bg(canvas, name, bx + 3, roi_y + 50,
                    COLORS[color_key], font_scale=0.4)

    # ── 车道中心线（蓝色虚线） ──
    if (lane_state and lane_state.left_barrier_px is not None
            and lane_state.ditch_px is not None):
        center = roi_x + (lane_state.left_barrier_px + lane_state.ditch_px) // 2
        for yy in range(roi_y, roi_y + roi_h, 12):
            cv2.line(canvas, (center, yy), (center, yy + 6),
                    COLORS['lane_center'], 1)

    # ── 危险区线（深红色） ──
    danger_y = roi_y + int(roi_h * config.DANGER_ZONE_TOP_RATIO)
    cv2.line(canvas, (roi_x, danger_y), (roi_x + roi_w, danger_y),
            COLORS['danger_zone'], 1)
    cv2.putText(canvas, "DANGER", (roi_x + 3, danger_y - 5),
               cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLORS['danger_zone'], 1)

    # ── 障碍物（红色框 + 危险等级） ──
    if obs_state:
        for (x, y, bw, bh) in obs_state.obstacle_boxes:
            ax, ay = roi_x + x, roi_y + y
            cv2.rectangle(canvas, (ax, ay), (ax + bw, ay + bh),
                         COLORS['obstacle'], 2)
            danger = obs_state.danger_level
            label_bg(canvas, f"OBS:{danger:.1f}", ax, ay - 8,
                    COLORS['obstacle'], font_scale=0.35)

    # ── 碎石（橙色框 / 品红框） ──
    if debris_state:
        for (x, y, bw, bh, sz_cm) in debris_state.debris_boxes:
            ax, ay = roi_x + x, roi_y + y
            color = COLORS['large_debris'] if sz_cm >= config.DEBRIS_LARGE_CM else COLORS['debris']
            cv2.rectangle(canvas, (ax, ay), (ax + bw, ay + bh), color, 2)
            label_bg(canvas, f"{sz_cm:.0f}cm", ax, ay - 5,
                    color, font_scale=0.3)

    # ── 右上角状态栏 ──
    draw_status_bar(canvas, lane_state, free_state, obs_state, debris_state)

    # ── 底部图例 ──
    draw_legend(canvas, h)

    return canvas


def label_bg(img, text, x, y, color, font_scale=0.4, thickness=1):
    """带半透明背景的文字标签。"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    # 背景
    cv2.rectangle(img, (x - 1, y - th - 2), (x + tw + 2, y + 2),
                 (40, 40, 40), -1)
    cv2.rectangle(img, (x - 1, y - th - 2), (x + tw + 2, y + 2),
                 color, 1)
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness)


def draw_status_bar(img, lane_state, free_state, obs_state, debris_state):
    """右上角状态信息。"""
    h, w = img.shape[:2]
    lines = []
    if lane_state:
        lines.append(f"Lane: {'OK' if lane_state.is_valid else '--'}")
        if lane_state.ditch_px is not None:
            lines.append(f"Ditch@{lane_state.ditch_px}")
        if lane_state.lane_width_m > 0:
            lines.append(f"Width:{lane_state.lane_width_m:.1f}m")
    if free_state:
        lines.append(f"Free: L{free_state.left_free_score:.2f} "
                     f"C{free_state.center_free_score:.2f} "
                     f"R{free_state.right_free_score:.2f}")
    if obs_state:
        lines.append(f"Obs:{obs_state.danger_level:.2f} "
                     f"({'L' if obs_state.blocked_left else '-'}"
                     f"{'C' if obs_state.blocked_center else '-'}"
                     f"{'R' if obs_state.blocked_right else '-'})")
    if debris_state:
        lines.append(f"Debris:{'LARGE' if debris_state.has_large_debris else 'small' if debris_state.has_debris else 'none'}")

    y = 20
    for line in lines:
        label_bg(img, line, w - 200, y, (255, 255, 255), font_scale=0.35)
        y += 18


def draw_legend(img, h):
    """底部图例条。"""
    x_start = 10
    y = h - 25
    font = cv2.FONT_HERSHEY_SIMPLEX
    for name, color in LEGEND:
        cv2.line(img, (x_start, y), (x_start + 20, y), color, 2)
        cv2.putText(img, name, (x_start + 25, y + 4), font, 0.3, color, 1)
        x_start += 130
        if x_start > img.shape[1] - 130:
            break


# ══════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="批量标注隧道视频帧")
    parser.add_argument("--input", default="../data/annotations/images/",
                       help="输入帧目录")
    parser.add_argument("--output", default="../data/annotations/labeled/",
                       help="输出标注图目录")
    parser.add_argument("--max-frames", type=int, default=0,
                       help="最大处理帧数(0=全部)")
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input)
    output_dir = os.path.abspath(args.output)
    os.makedirs(output_dir, exist_ok=True)

    # ── 初始化检测器 ──
    print("初始化检测器...")
    preprocessor = ImagePreprocessor()
    lane_detector = LaneBoundaryDetector()
    free_detector = FreeSpaceDetector()
    obs_detector = ObstacleDetector()
    debris_detector = DebrisDetector()
    calibrator = GroundCalibrator()

    # ── 获取所有帧 ──
    image_files = sorted([
        f for f in os.listdir(input_dir)
        if f.endswith(('.jpg', '.jpeg', '.png'))
    ])
    if args.max_frames > 0:
        image_files = image_files[:args.max_frames]

    print(f"处理 {len(image_files)} 帧 → {output_dir}/")

    report = defaultdict(lambda: defaultdict(int))
    stats = {
        'total': len(image_files),
        'lane_detected': 0,
        'obstacles_found': 0,
        'debris_found': 0,
        'large_debris_found': 0,
    }

    t0 = time.time()
    for i, fname in enumerate(image_files):
        path = os.path.join(input_dir, fname)
        frame = cv2.imread(path)
        if frame is None:
            continue

        h, w = frame.shape[:2]

        # ── 运行检测 ──
        preprocessed = preprocessor.process(frame)
        if preprocessed.roi_frame is None:
            continue

        roi_x = int(w * config.ROI_LEFT_RATIO)
        roi_y = int(h * config.ROI_TOP_RATIO)

        lane_state = lane_detector.detect(
            preprocessed.roi_frame, preprocessed.edges,
            preprocessed.enhanced, calibrator)

        free_state = free_detector.detect(
            preprocessed.roi_frame, preprocessed.edges,
            preprocessed.binary_mask, lane_state)

        obs_state = obs_detector.detect(
            preprocessed.roi_frame, preprocessed.edges,
            preprocessed.binary_mask)

        debris_state = debris_detector.detect(
            preprocessed.gray, preprocessed.edges,
            lane_state, calibrator)

        # ── 绘制标注 ──
        labeled = draw_annotations(
            frame, preprocessed, lane_state,
            free_state, obs_state, debris_state,
            (roi_x, roi_y))

        # ── 保存 ──
        out_path = os.path.join(output_dir, fname)
        cv2.imwrite(out_path, labeled, [cv2.IMWRITE_JPEG_QUALITY, 90])

        # ── 统计 ──
        vid = fname.split('_')[0]
        report[vid]['frames'] += 1
        if lane_state.is_valid:
            report[vid]['lane_ok'] += 1
            stats['lane_detected'] += 1
        if obs_state.has_obstacle:
            report[vid]['obstacles'] += len(obs_state.obstacle_boxes)
            stats['obstacles_found'] += len(obs_state.obstacle_boxes)
        if debris_state.has_debris:
            report[vid]['debris'] += len(debris_state.debris_boxes)
            stats['debris_found'] += len(debris_state.debris_boxes)
        if debris_state.has_large_debris:
            report[vid]['large_debris'] += 1
            stats['large_debris_found'] += 1

        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            fps = (i + 1) / elapsed
            print(f"  进度: {i+1}/{len(image_files)} ({fps:.1f}帧/秒)")

    elapsed = time.time() - t0
    print(f"\n✅ 完成! {len(image_files)}帧, 耗时{elapsed:.1f}秒 ({len(image_files)/elapsed:.1f}帧/秒)")

    # ── 报告 ──
    print("\n" + "=" * 60)
    print("  标注统计报告")
    print("=" * 60)
    print(f"  总帧数:        {stats['total']}")
    print(f"  车道识别:      {stats['lane_detected']} ({stats['lane_detected']/max(1,stats['total'])*100:.0f}%)")
    print(f"  障碍物标注:    {stats['obstacles_found']} 个")
    print(f"  碎石标注:      {stats['debris_found']} 个")
    print(f"  大块碎石(>15cm): {stats['large_debris_found']} 个")
    print(f"\n  按视频:")
    for vid in sorted(report.keys()):
        r = report[vid]
        print(f"    {vid}: {r['frames']}帧, "
              f"车道{r.get('lane_ok',0)}, "
              f"障碍{r.get('obstacles',0)}, "
              f"碎石{r.get('debris',0)}")

    # 保存JSON报告
    report_path = os.path.join(os.path.dirname(output_dir), 'labeled_report.json')
    with open(report_path, 'w') as f:
        json.dump({'stats': stats, 'per_video': dict(report)}, f, indent=2, ensure_ascii=False)
    print(f"\n  报告: {report_path}")

    print(f"\n  标注图: {output_dir}")
    print(f"  共 {len(image_files)} 张，可直接浏览")


if __name__ == "__main__":
    main()
