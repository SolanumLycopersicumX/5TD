#!/usr/bin/env python3
"""
隧道障碍标注数据提取管线 v1.0

智能帧采样 + CV预标注 + COCO/LabelMe双格式输出

策略：
  1. 自适应采样 — 基于感知哈希去冗余，保留特征多样的帧
  2. CV预标注 — 运行现有检测器，生成初步标签供人工精修
  3. 双格式输出 — COCO JSON (训练用) + LabelMe JSON (人工精修用)

使用：
  python extract_annotation_frames.py --videos ../*.mp4 --out ../data/annotations/
"""

import sys, os, json, cv2, hashlib, time, argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Optional

# ── 项目路径 ──────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import config
from preprocess import ImagePreprocessor
from lane_boundary_detector import LaneBoundaryDetector
from obstacle_detector import ObstacleDetector
from debris_detector import DebrisDetector
from calibration import GroundCalibrator


# ══════════════════════════════════════════════════════════════════════
# 智能帧采样
# ══════════════════════════════════════════════════════════════════════

def perceptual_hash(img: np.ndarray, size: int = 16) -> str:
    """感知哈希 — 用于检测帧间相似度，跳过重复帧。"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    avg = resized.mean()
    return ''.join(['1' if p > avg else '0' for p in resized.flatten()])

def hamming_distance(h1: str, h2: str) -> int:
    """两个感知哈希的汉明距离。"""
    return sum(c1 != c2 for c1, c2 in zip(h1, h2))

def smart_sample_frames(video_path: str,
                        fps_sample: int = 5,
                        min_hash_diff: int = 8,
                        max_frames: int = 200) -> list:
    """
    智能帧采样：均匀降采样 + 感知哈希去冗余。

    返回: [(frame_idx, frame_bgr, timestamp_sec), ...]
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    step = max(1, int(video_fps / fps_sample))

    sampled = []
    last_hash = None

    for idx in range(0, total_frames, step):
        if len(sampled) >= max_frames:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        # 感知哈希去冗余
        phash = perceptual_hash(frame)
        if last_hash is not None:
            diff = hamming_distance(phash, last_hash)
            if diff < min_hash_diff:
                continue  # 帧太相似，跳过

        sampled.append((idx, frame, idx / video_fps))
        last_hash = phash

    cap.release()
    return sampled


# ══════════════════════════════════════════════════════════════════════
# CV 预标注引擎
# ══════════════════════════════════════════════════════════════════════

@dataclass
class Annotation:
    """单条标注"""
    bbox: list              # [x, y, w, h] 像素坐标
    category: str           # 类别名
    confidence: float = 0.0 # CV检测置信度（预标注）
    attributes: dict = field(default_factory=dict)  # 额外属性


class PreAnnotator:
    """
    CV预标注引擎。
    运行现有检测管线，为每帧生成预标注。
    """

    def __init__(self):
        self.preprocessor = ImagePreprocessor()
        self.lane_detector = LaneBoundaryDetector()
        self.obstacle_detector = ObstacleDetector()
        self.debris_detector = DebrisDetector()
        self.calibrator = GroundCalibrator()

    def annotate(self, frame: np.ndarray) -> dict:
        """
        对单帧执行CV管线，返回标注结果。

        返回:
          {
            'lane': LaneBoundaryState or None,
            'obstacles': [Annotation, ...],
            'debris': [Annotation, ...],
            'preprocessed': PreprocessResult,
          }
        """
        preprocessed = self.preprocessor.process(frame)
        if preprocessed.roi_frame is None:
            return {'lane': None, 'obstacles': [], 'debris': [], 'preprocessed': preprocessed}

        # 车道边界
        lane_state = self.lane_detector.detect(
            preprocessed.roi_frame, preprocessed.edges,
            preprocessed.enhanced, self.calibrator)

        # 障碍检测
        obs_state = self.obstacle_detector.detect(
            preprocessed.roi_frame, preprocessed.edges,
            preprocessed.binary_mask)

        # 碎石检测
        debris_state = self.debris_detector.detect(
            preprocessed.gray, preprocessed.edges,
            lane_state, self.calibrator)

        # ── 组装标注 ──────────────────────────────────────────────
        annotations = {'lane': None, 'obstacles': [], 'debris': []}

        # ROI偏移量（标注坐标需转回原图）
        roi_x = int(frame.shape[1] * config.ROI_LEFT_RATIO)
        roi_y = int(frame.shape[0] * config.ROI_TOP_RATIO)

        # 车道结构（用于参考，不作为障碍标注）
        if lane_state.is_valid:
            roi_h = preprocessed.roi_frame.shape[0]
            lane_info = {
                'ditch_px': lane_state.ditch_px,
                'left_barrier_px': lane_state.left_barrier_px,
                'right_barrier_px': lane_state.right_barrier_px,
                'lane_width_m': lane_state.lane_width_m,
                'confidence': lane_state.confidence,
            }
            annotations['lane'] = lane_info

        # 障碍物
        for (x, y, bw, bh) in obs_state.obstacle_boxes:
            annotations['obstacles'].append(Annotation(
                bbox=[x + roi_x, y + roi_y, bw, bh],
                category='obstacle',
                confidence=obs_state.danger_level,
            ))

        # 碎石
        for (x, y, bw, bh, sz_cm) in debris_state.debris_boxes:
            cat = 'large_debris' if sz_cm >= config.DEBRIS_LARGE_CM else 'debris'
            annotations['debris'].append(Annotation(
                bbox=[x + roi_x, y + roi_y, bw, bh],
                category=cat,
                confidence=min(1.0, sz_cm / 30.0),
                attributes={'size_cm': round(sz_cm, 1)},
            ))

        annotations['preprocessed'] = preprocessed
        return annotations


# ══════════════════════════════════════════════════════════════════════
# COCO 格式输出
# ══════════════════════════════════════════════════════════════════════

COCO_CATEGORIES = [
    {"id": 1, "name": "obstacle",      "supercategory": "static"},
    {"id": 2, "name": "debris",        "supercategory": "static"},
    {"id": 3, "name": "large_debris",  "supercategory": "static"},
    {"id": 4, "name": "ditch",         "supercategory": "lane_structure"},
    {"id": 5, "name": "barrier_left",  "supercategory": "lane_structure"},
    {"id": 6, "name": "barrier_right", "supercategory": "lane_structure"},
    {"id": 7, "name": "worker",        "supercategory": "dynamic"},
    {"id": 8, "name": "equipment",     "supercategory": "dynamic"},
]

CAT_NAME_TO_ID = {c["name"]: c["id"] for c in COCO_CATEGORIES}


def build_coco_dataset(all_frame_data: list, output_dir: str,
                       video_names: list) -> dict:
    """
    构建 COCO 格式数据集。

    all_frame_data: [(video_name, frame_idx, frame_bgr, annotations_dict), ...]
    """
    coco = {
        "info": {
            "description": "Tunnel Obstacle Detection Dataset",
            "version": "1.0",
            "year": 2026,
            "date_created": time.strftime("%Y-%m-%d"),
        },
        "licenses": [{"id": 1, "name": "Internal Use"}],
        "images": [],
        "annotations": [],
        "categories": COCO_CATEGORIES,
    }

    img_dir = os.path.join(output_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    ann_id = 1

    for data_idx, (vid_name, frame_idx, frame, annotations) in enumerate(all_frame_data):
        img_id = data_idx + 1
        h, w = frame.shape[:2]

        # 保存图像
        img_filename = f"{vid_name}_f{frame_idx:06d}.jpg"
        img_path = os.path.join(img_dir, img_filename)
        cv2.imwrite(img_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 92])

        coco["images"].append({
            "id": img_id,
            "file_name": img_filename,
            "width": w,
            "height": h,
            "video": vid_name,
            "frame_idx": frame_idx,
        })

        # ── 障碍物标注 ──────────────────────────────────────────
        for obs in annotations.get('obstacles', []):
            x, y, bw, bh = obs.bbox
            cat_id = CAT_NAME_TO_ID.get(obs.category, 1)
            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cat_id,
                "bbox": [int(x), int(y), int(bw), int(bh)],
                "area": int(bw * bh),
                "iscrowd": 0,
                "confidence": round(obs.confidence, 3),  # CV预标注置信度
                "source": "cv_prelabel",
            })
            ann_id += 1

        # ── 碎石标注 ────────────────────────────────────────────
        for deb in annotations.get('debris', []):
            x, y, bw, bh = deb.bbox
            cat_id = CAT_NAME_TO_ID.get(deb.category, 2)
            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cat_id,
                "bbox": [int(x), int(y), int(bw), int(bh)],
                "area": int(bw * bh),
                "iscrowd": 0,
                "confidence": round(deb.confidence, 3),
                "attributes": deb.attributes,
                "source": "cv_prelabel",
            })
            ann_id += 1

        # ── 车道结构标注（用于辅助参考，不作为检测目标）────────
        lane_info = annotations.get('lane', {})
        if lane_info and lane_info.get('ditch_px') is not None:
            roi_h = frame.shape[0]
            roi_y = int(frame.shape[0] * config.ROI_TOP_RATIO)
            # 隔离沟标记为竖直线（近似bbox）
            ditch_x = lane_info['ditch_px'] + int(frame.shape[1] * config.ROI_LEFT_RATIO)
            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": CAT_NAME_TO_ID["ditch"],
                "bbox": [ditch_x - 5, roi_y, 10, int(roi_h * 0.4)],
                "area": 10 * int(roi_h * 0.4),
                "iscrowd": 0,
                "confidence": round(lane_info.get('confidence', 0), 3),
                "source": "cv_prelabel",
            })
            ann_id += 1

    return coco


# ══════════════════════════════════════════════════════════════════════
# LabelMe 格式输出
# ══════════════════════════════════════════════════════════════════════

def build_labelme_json(frame: np.ndarray, annotations: dict,
                      img_filename: str, img_dir: str) -> dict:
    """
    为单帧生成 LabelMe 兼容的 JSON。
    """
    h, w = frame.shape[:2]

    shapes = []
    for obs in annotations.get('obstacles', []):
        x, y, bw, bh = obs.bbox
        shapes.append({
            "label": obs.category,
            "points": [[x, y], [x + bw, y + bh]],
            "group_id": None,
            "shape_type": "rectangle",
            "flags": {"cv_confidence": round(obs.confidence, 3)},
        })

    for deb in annotations.get('debris', []):
        x, y, bw, bh = deb.bbox
        shapes.append({
            "label": deb.category,
            "points": [[x, y], [x + bw, y + bh]],
            "group_id": None,
            "shape_type": "rectangle",
            "flags": {"cv_confidence": round(deb.confidence, 3),
                      "size_cm": deb.attributes.get('size_cm', 0)},
        })

    return {
        "version": "5.0.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": os.path.join("images", img_filename),
        "imageData": None,  # 不嵌入base64，减小JSON体积
        "imageHeight": h,
        "imageWidth": w,
    }


# ══════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="隧道障碍标注数据提取管线")
    parser.add_argument("--videos", nargs="+", required=True,
                       help="输入视频文件路径")
    parser.add_argument("--out", default="../data/annotations/",
                       help="输出目录")
    parser.add_argument("--fps", type=int, default=5,
                       help="采样帧率 (默认5fps)")
    parser.add_argument("--hash-diff", type=int, default=8,
                       help="感知哈希最小差异 (默认8，越大保留越多)")
    parser.add_argument("--max-per-video", type=int, default=200,
                       help="每视频最大帧数")
    parser.add_argument("--no-prelabel", action="store_true",
                       help="跳过CV预标注，只提取帧")
    args = parser.parse_args()

    output_dir = os.path.abspath(args.out)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("  隧道障碍标注数据提取管线 v1.0")
    print(f"  采样: {args.fps}fps, 哈希差异≥{args.hash_diff}")
    print(f"  输出: {output_dir}")
    print("=" * 60)

    all_data = []
    video_stats = {}

    # ── 阶段1: 智能帧采样 ──────────────────────────────────────
    print("\n[阶段1] 智能帧采样...")
    preannotator = None if args.no_prelabel else PreAnnotator()

    for video_path in args.videos:
        if not os.path.exists(video_path):
            print(f"  ⚠ 跳过(不存在): {video_path}")
            continue

        vid_name = os.path.splitext(os.path.basename(video_path))[0][:6]
        print(f"  📹 {vid_name}...", end=" ", flush=True)

        frames = smart_sample_frames(
            video_path,
            fps_sample=args.fps,
            min_hash_diff=args.hash_diff,
            max_frames=args.max_per_video,
        )
        print(f"提取 {len(frames)} 帧", end="")

        if not args.no_prelabel:
            print(", 预标注中...", end=" ", flush=True)
            annotated_count = 0
            for fidx, frame, ts in frames:
                annotations = preannotator.annotate(frame)
                all_data.append((vid_name, fidx, frame, annotations))

                n_obs = len(annotations.get('obstacles', []))
                n_deb = len(annotations.get('debris', []))
                if n_obs + n_deb > 0:
                    annotated_count += 1

            lane_ok = annotations.get('lane') is not None
            print(f"障碍={annotated_count} 车道={'✅' if lane_ok else '❌'}")
            video_stats[vid_name] = {
                'frames': len(frames),
                'with_obstacles': annotated_count,
            }
        else:
            for fidx, frame, ts in frames:
                all_data.append((vid_name, fidx, frame, {}))
            print()
            video_stats[vid_name] = {'frames': len(frames)}

    print(f"\n  总计: {len(all_data)} 帧")

    # ── 阶段2: COCO格式输出 ────────────────────────────────────
    print("\n[阶段2] 生成COCO数据集...")
    coco = build_coco_dataset(all_data, output_dir,
                             video_names=list(video_stats.keys()))
    coco_path = os.path.join(output_dir, "annotations_coco.json")
    with open(coco_path, 'w', encoding='utf-8') as f:
        json.dump(coco, f, indent=2, ensure_ascii=False)

    n_anns = len(coco["annotations"])
    n_imgs = len(coco["images"])
    print(f"  ✅ COCO: {n_imgs}张图, {n_anns}条标注 → {coco_path}")

    # ── 阶段3: LabelMe格式输出 ─────────────────────────────────
    print("\n[阶段3] 生成LabelMe标注...")
    labelme_dir = os.path.join(output_dir, "labelme")
    img_dir = os.path.join(output_dir, "images")
    os.makedirs(labelme_dir, exist_ok=True)

    for vid_name, frame_idx, frame, annotations in all_data:
        img_filename = f"{vid_name}_f{frame_idx:06d}.jpg"
        labelme_json = build_labelme_json(frame, annotations,
                                          img_filename, img_dir)
        json_path = os.path.join(labelme_dir,
                                f"{vid_name}_f{frame_idx:06d}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(labelme_json, f, indent=2, ensure_ascii=False)

    # 也保存图片到labelme目录的images子目录（软链接即可，但LabelMe期望相对路径）
    print(f"  ✅ LabelMe: {len(all_data)} 个JSON → {labelme_dir}/")

    # ── 汇总报告 ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  提取完成 — 汇总报告")
    print("=" * 60)
    for vid, stats in video_stats.items():
        obs_str = f", {stats.get('with_obstacles', '?')}含障碍" if 'with_obstacles' in stats else ""
        print(f"  {vid}: {stats['frames']}帧{obs_str}")
    print(f"\n  总帧数:   {len(all_data)}")
    print(f"  总标注数: {n_anns}")

    # 类别分布
    cat_counts = defaultdict(int)
    for ann in coco["annotations"]:
        cat_name = next((c["name"] for c in COCO_CATEGORIES
                        if c["id"] == ann["category_id"]), "unknown")
        cat_counts[cat_name] += 1
    print(f"  类别分布:")
    for cat, count in sorted(cat_counts.items()):
        print(f"    {cat}: {count}")

    print(f"\n  输出结构:")
    print(f"    {output_dir}/")
    print(f"    ├── images/              ← 提取的帧图片")
    print(f"    ├── annotations_coco.json ← COCO格式(训练用)")
    print(f"    └── labelme/              ← LabelMe JSON(人工精修用)")

    print(f"\n  下一步:")
    print(f"    1. 人工精修: labelme {output_dir}/labelme/")
    print(f"    2. 训练: 使用 annotations_coco.json")


if __name__ == "__main__":
    main()
