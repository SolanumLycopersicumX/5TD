"""
重新生成 overlay 图：超大号编号，方便辨认。
"""
import cv2
import numpy as np
import os
import json
import sys

ANNO_DIR = '/home/nickwang/Projects/vision_obstacle_avoidance/datasets/tunnel/annotations/train'
IMAGE_DIR = '/home/nickwang/Projects/vision_obstacle_avoidance/datasets/tunnel/images/train'

# 20 种高对比度颜色
COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
    (0, 128, 255), (128, 0, 255), (255, 192, 0), (0, 255, 128),
    (255, 64, 64), (64, 255, 64), (64, 64, 255), (192, 192, 0),
    (192, 0, 192), (0, 192, 192), (255, 128, 192), (192, 255, 128),
]

files = sorted([f for f in os.listdir(ANNO_DIR) if f.endswith('_masks.json')])

for json_file in files:
    fname_base = json_file.replace('_masks.json', '')
    json_path = os.path.join(ANNO_DIR, json_file)
    img_path = os.path.join(IMAGE_DIR, f'{fname_base}.jpg')
    orig_path = os.path.join(ANNO_DIR, f'{fname_base}_original.jpg')

    # 读取原图
    image = cv2.imread(orig_path)
    if image is None:
        image = cv2.imread(img_path)
    if image is None:
        print(f"SKIP {fname_base}: no image")
        continue

    h, w = image.shape[:2]

    with open(json_path) as f:
        masks_info = json.load(f)

    # 新建叠加图
    overlay = image.copy()

    # 字号根据图片大小自适应
    font_scale = max(1.5, min(6.0, h / 200.0))
    thickness = max(3, int(font_scale * 2))
    circle_radius = max(20, int(h / 40))

    for item in masks_info:
        mid = item['id']
        mask_path = os.path.join(ANNO_DIR, f'{fname_base}_mask{mid:03d}.png')
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        color = COLORS[mid % len(COLORS)]

        # 半透明叠加
        overlay[mask > 0] = (overlay[mask > 0] * 0.45 + np.array(color) * 0.55).astype(np.uint8)

        # 找 mask 的质心，把编号画在质心位置
        ys, xs = np.where(mask > 0)
        if len(ys) == 0:
            continue
        cy, cx = int(np.mean(ys)), int(np.mean(xs))

        # 白色底圆 + 黑色数字
        cv2.circle(overlay, (cx, cy), circle_radius, (255, 255, 255), -1)
        cv2.circle(overlay, (cx, cy), circle_radius, (0, 0, 0), max(2, thickness // 2))

        text = str(mid)
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        tx, ty = cx - text_size[0] // 2, cy + text_size[1] // 2
        cv2.putText(overlay, text, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (0, 0, 0), thickness, cv2.LINE_AA)

    out_path = os.path.join(ANNO_DIR, f'{fname_base}_overlay.jpg')
    cv2.imwrite(out_path, overlay)
    print(f"Done: {fname_base}")

print(f"\n全部 {len(files)} 张 overlay 已重新生成！")
print(f"位置: {ANNO_DIR}/*_overlay.jpg")
