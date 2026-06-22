"""
标注辅助工具：查看 SAM 生成的 mask，选择哪些 mask 对应哪些类别。
生成 HBDNet 训练所需的标注文件。
"""
import cv2
import os
import json
import numpy as np

ANNO_DIR = '/home/nickwang/Projects/vision_obstacle_avoidance/datasets/tunnel/annotations/train'
IMAGE_DIR = '/home/nickwang/Projects/vision_obstacle_avoidance/datasets/tunnel/images/train'

# 类别定义
CATEGORIES = {
    'ego_passable': '可行驶区域 (本车半幅车道内路面)',
    'ditch':        '中央隔离沟',
    'left_barrier': '左侧隔离带/隧道壁',
    'right_barrier':'右侧隔离带/隧道壁',
    'obstacle':     '障碍物 (工人/车辆/碎石/悬挂物)',
}


def show_frame_summary():
    """显示每个帧的 SAM mask 摘要"""
    results = []
    for f in sorted(os.listdir(ANNO_DIR)):
        if f.endswith('_masks.json'):
            with open(os.path.join(ANNO_DIR, f)) as fh:
                data = json.load(fh)
            fname = f.replace('_masks.json', '')
            results.append({
                'frame': fname,
                'n_masks': len(data),
                'areas': [m['area'] for m in data[:5]],  # top 5 areas
                'stabilities': [m['stability'] for m in data[:5]],
            })

    print(f"{'Frame':<50} {'Masks':>6} {'Top 5 Areas':>40}")
    print("-" * 100)
    for r in results:
        areas_str = ', '.join([f'{a}' for a in r['areas']])
        print(f"{r['frame']:<50} {r['n_masks']:>6} {areas_str:>40}")


def create_annotation(frame_name, selections):
    """
    根据用户选择的 mask 生成标注文件。

    selections = {
        'ego_passable': [0, 3, 5],     # mask ID 列表
        'ditch': [2],
        'left_barrier': [],
        'right_barrier': [7],
        'obstacle': [],
    }
    """
    masks_json = os.path.join(ANNO_DIR, f'{frame_name}_masks.json')
    if not os.path.exists(masks_json):
        print(f"找不到 {masks_json}")
        return

    with open(masks_json) as f:
        mask_info = json.load(f)

    # 1. 生成 ego_passable_mask (单通道 PNG)
    ego_passable = np.zeros((1280, 720), dtype=np.uint8)
    for mid in selections.get('ego_passable', []):
        mask_path = os.path.join(ANNO_DIR, f'{frame_name}_mask{mid:03d}.png')
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            ego_passable[mask > 0] = 255
    cv2.imwrite(os.path.join(ANNO_DIR, f'{frame_name}_ego_passable.png'), ego_passable)

    # 2. 生成 hard_boundary_mask (4 通道 PNG)
    boundary = np.zeros((1280, 720, 4), dtype=np.uint8)
    for ch, key in enumerate(['ditch', 'left_barrier', 'right_barrier', 'tunnel_wall']):
        for mid in selections.get(key, []):
            mask_path = os.path.join(ANNO_DIR, f'{frame_name}_mask{mid:03d}.png')
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                boundary[mask > 0, ch] = 255
    cv2.imwrite(os.path.join(ANNO_DIR, f'{frame_name}_hard_boundary.png'), boundary)

    # 3. 保存选择记录
    with open(os.path.join(ANNO_DIR, f'{frame_name}_selections.json'), 'w') as f:
        json.dump(selections, f, indent=2)

    print(f"已生成: {frame_name}_ego_passable.png + {frame_name}_hard_boundary.png")


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("用法:")
        print("  python annotate_helper.py summary       — 查看所有帧的 SAM mask 摘要")
        print("  python annotate_helper.py annotate <帧名前缀>  — 交互式标注 (TODO)")
        show_frame_summary()
    else:
        cmd = sys.argv[1]
        if cmd == 'summary':
            show_frame_summary()
        elif cmd == 'annotate':
            frame = sys.argv[2] if len(sys.argv) > 2 else None
            if frame:
                print(f"标注帧: {frame}")
                # TODO: 交互式标注
            else:
                print("请指定帧名前缀")
