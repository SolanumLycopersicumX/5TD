"""
SAM 自动标注脚本：对抽帧图片运行 SAM，生成候选 mask。
每个 mask 保存为单独的 PNG，同时生成可视化叠加图。
"""
import cv2
import numpy as np
import os
import sys
import json
import torch

# 加载 SAM
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

MODEL_PATH = '/home/nickwang/Projects/vision_obstacle_avoidance/models/sam_vit_b_01ec64.pth'
IMAGE_DIR = '/home/nickwang/Projects/vision_obstacle_avoidance/datasets/tunnel/images/train'
OUT_DIR = '/home/nickwang/Projects/vision_obstacle_avoidance/datasets/tunnel/annotations/train'

os.makedirs(OUT_DIR, exist_ok=True)

print("Loading SAM ViT-B...")
sam = sam_model_registry["vit_b"](checkpoint=MODEL_PATH)
device = "cuda" if torch.cuda.is_available() else "cpu"
sam.to(device)
print(f"Device: {device}")

mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,         # 每侧采样点数（越高越精细，越慢）
    pred_iou_thresh=0.88,       # IoU 预测阈值
    stability_score_thresh=0.95, # 稳定性阈值
    min_mask_region_area=100,    # 最小 mask 面积（像素）
)

files = sorted(os.listdir(IMAGE_DIR))  # 全部帧
# 跳过已有 overlay 的帧
processed = set()
for f in os.listdir(OUT_DIR):
    if '_overlay' in f:
        processed.add(f.replace('_overlay.jpg', '.jpg'))
files = [f for f in files if f not in processed]
print(f"处理 {len(files)} 张图片（跳过 {len(processed)} 张已完成）...")

for fname in files:
    path = os.path.join(IMAGE_DIR, fname)
    image = cv2.imread(path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]

    print(f"\n{fname}: {w}×{h}", end=" ")

    # SAM 推理
    masks = mask_generator.generate(image_rgb)
    print(f"→ {len(masks)} masks")

    if len(masks) == 0:
        continue

    # 按面积从大到小排序
    masks = sorted(masks, key=lambda m: m['area'], reverse=True)

    # 保存 mask 信息
    mask_info = []
    for i, mask_data in enumerate(masks):
        mask = mask_data['segmentation'].astype(np.uint8) * 255
        area = mask_data['area']
        bbox = [int(x) for x in mask_data['bbox']]  # [x, y, w, h]
        stability = mask_data['stability_score']

        # 保存单个 mask PNG
        mask_fname = f"{os.path.splitext(fname)[0]}_mask{i:03d}.png"
        cv2.imwrite(os.path.join(OUT_DIR, mask_fname), mask)

        mask_info.append({
            'id': i,
            'area': int(area),
            'bbox': bbox,
            'stability': round(stability, 3),
            'file': mask_fname,
        })

    # 保存 mask 元数据 JSON
    json_fname = f"{os.path.splitext(fname)[0]}_masks.json"
    with open(os.path.join(OUT_DIR, json_fname), 'w') as f:
        json.dump(mask_info, f, indent=2)

    # 生成彩色叠加图：用不同颜色标注前 20 个最大 mask
    overlay = image.copy()
    colors = [
        (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
        (0, 128, 255), (128, 0, 255), (255, 255, 128), (128, 255, 255),
        (255, 128, 128), (128, 128, 255), (64, 255, 64), (255, 64, 64),
        (64, 64, 255), (255, 255, 64), (255, 64, 255), (64, 255, 255),
    ]

    for i, mask_data in enumerate(masks[:20]):
        mask = mask_data['segmentation']
        color = colors[i % len(colors)]
        overlay[mask] = (overlay[mask] * 0.5 + np.array(color) * 0.5).astype(np.uint8)

        # 在 bbox 左上角标编号
        bx, by, bw, bh = [int(v) for v in mask_data['bbox']]
        cv2.putText(overlay, str(i), (bx + 2, by + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    cv2.imwrite(os.path.join(OUT_DIR, f'{os.path.splitext(fname)[0]}_overlay.jpg'), overlay)

    # 保存原图副本方便对照
    cv2.imwrite(os.path.join(OUT_DIR, f'{os.path.splitext(fname)[0]}_original.jpg'), image)

print(f"\n完成！结果保存在: {OUT_DIR}")
print(f"每个帧对应: original.jpg(原图) + overlay.jpg(彩色叠加) + masks.json(元数据) + mask*.png(独立mask)")
