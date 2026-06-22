"""
Prompted SAM：用参考点引导分割，保证所有帧标注一致。

策略：
  用户在参考帧上定义每个类别的"示例点"（图像坐标），
  SAM 用这些点对所有帧做 prompted segmentation，输出一致的 mask。
"""
import cv2
import numpy as np
import os, json
import torch
from segment_anything import sam_model_registry, SamPredictor

MODEL_PATH = '/home/nickwang/Projects/vision_obstacle_avoidance/models/sam_vit_b_01ec64.pth'
IMAGE_DIR = '/home/nickwang/Projects/vision_obstacle_avoidance/datasets/tunnel/images/train'
ANNO_DIR = '/home/nickwang/Projects/vision_obstacle_avoidance/datasets/tunnel/annotations/train'
os.makedirs(ANNO_DIR, exist_ok=True)

# ============================================================
# 用户配置区：定义每个类别的参考点
# 格式: (x, y) — 图像中的像素坐标，表示"这个位置属于该类别"
# 720×1280 竖屏图像，原点在左上角，(0,0)=左上，(720,1280)=右下
# ============================================================

# 在同一张参考图上取点，要确保这些点落在这个类别的区域内
# 建议: 打开任意一张原图，用看图软件查看像素坐标

REFERENCE_POINTS = {
    'ego_passable': [
        # 可行驶路面 — 质心来自 f0090 mask1
        (347, 1037),
        (334, 914),
    ],
    'ditch': [
        # 中央隔离沟 — 质心来自 f0090 mask2
        (595, 805),
    ],
    'left_barrier': [
        # 左侧硬路肩 — 质心来自 f0090 mask3
        (102, 759),
    ],
    'right_barrier': [
        # 右侧隧道壁 — 质心来自 f0090 mask4
        (598, 775),
    ],
}

# 每类用 positive point (label=1)
# ============================================================

print("Loading SAM...")
sam = sam_model_registry["vit_b"](checkpoint=MODEL_PATH)
device = "cuda" if torch.cuda.is_available() else "cpu"
sam.to(device)
predictor = SamPredictor(sam)
print(f"Device: {device}")

# 获取所有帧
files = sorted(os.listdir(IMAGE_DIR))
print(f"共 {len(files)} 帧")

for fname in files:
    img_path = os.path.join(IMAGE_DIR, fname)
    image = cv2.imread(img_path)
    if image is None:
        continue
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    H, W = image.shape[:2]

    base = os.path.splitext(fname)[0]
    out_orig = os.path.join(ANNO_DIR, f'{base}_original.jpg')
    cv2.imwrite(out_orig, image)

    # 只保留已经有原始图的帧，跳过新帧（SAM 还在跑自动模式）
    # 这里我们对所有帧都做

    predictor.set_image(image_rgb)

    ego_passable_mask = np.zeros((H, W), dtype=np.uint8)
    hard_boundary_mask = np.zeros((H, W, 4), dtype=np.uint8)

    # 对每个类别跑 prompted segmentation
    results = {}
    for class_name, points in REFERENCE_POINTS.items():
        if not points:
            continue

        input_points = np.array(points)
        input_labels = np.ones(len(points))  # 1 = positive point

        masks, scores, _ = predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=False,  # 每个 prompt 只返回最好的一个 mask
        )

        # masks shape: (1, H, W)
        best_mask = masks[0].astype(np.uint8) * 255
        best_score = float(scores[0])

        print(f"  {fname[:20]} | {class_name:20s} score={best_score:.3f} | {np.count_nonzero(best_mask)} px")

        results[class_name] = {'score': best_score, 'px': int(np.count_nonzero(best_mask))}

        # 保存单个类别的 mask
        cv2.imwrite(os.path.join(ANNO_DIR, f'{base}_{class_name}.png'), best_mask)

        # 组装最终标注
        if class_name == 'ego_passable':
            ego_passable_mask[best_mask > 0] = 255
        elif class_name == 'ditch':
            hard_boundary_mask[best_mask > 0, 0] = 255
        elif class_name == 'left_barrier':
            hard_boundary_mask[best_mask > 0, 1] = 255
        elif class_name == 'right_barrier':
            hard_boundary_mask[best_mask > 0, 2] = 255
            hard_boundary_mask[best_mask > 0, 3] = 255  # 同时是 tunnel_wall

    # 保存最终标注
    cv2.imwrite(os.path.join(ANNO_DIR, f'{base}_ego_passable.png'), ego_passable_mask)
    cv2.imwrite(os.path.join(ANNO_DIR, f'{base}_hard_boundary.png'), hard_boundary_mask)

    # 保存元数据
    with open(os.path.join(ANNO_DIR, f'{base}_prompted.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # 生成可视化叠加图
    overlay = image.copy()
    ch_colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0), (255, 0, 255)]
    ch_names = ['ego_passable', 'ditch', 'left_barrier', 'right_barrier']
    for i, cn in enumerate(ch_names):
        mpath = os.path.join(ANNO_DIR, f'{base}_{cn}.png')
        m = cv2.imread(mpath, cv2.IMREAD_GRAYSCALE)
        if m is not None and np.count_nonzero(m) > 0:
            c = ch_colors[i]
            overlay[m > 0] = (overlay[m > 0] * 0.4 + np.array(c) * 0.6).astype(np.uint8)

    # 画图例
    legend = [
        ("绿=可行驶路面", (0, 255, 0)),
        ("蓝=中央隔离沟", (0, 0, 255)),
        ("红=左侧硬路肩", (255, 0, 0)),
        ("黄=右侧隧道壁", (255, 255, 0)),
    ]
    for j, (text, color) in enumerate(legend):
        y = 30 + j * 35
        cv2.rectangle(overlay, (10, y - 15), (35, y + 5), color, -1)
        cv2.putText(overlay, text, (45, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imwrite(os.path.join(ANNO_DIR, f'{base}_prompted_overlay.jpg'), overlay)

print(f"\n✅ 全部完成！结果: {ANNO_DIR}/")
print("每组文件:")
print("  _original.jpg       — 原图")
print("  _prompted_overlay.jpg — 标注叠加图（绿=路面, 蓝=沟, 红=左侧, 黄=右侧）")
print("  _ego_passable.png   — 可行驶路面 mask")
print("  _hard_boundary.png  — 硬边界 4通道 mask")
