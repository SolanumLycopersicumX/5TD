"""
诊断脚本：逐步骤检查车道边界检测为何漏检。
"""
import cv2
import numpy as np
import sys, os

sys.path.insert(0, '/home/nickwang/Projects/vision_obstacle_avoidance/vision_obstacle_avoidance')
import config
from preprocess import ImagePreprocessor

# 读取一帧
frame = cv2.imread('/home/nickwang/Projects/vision_obstacle_avoidance/data/frames/samples/11d80f_f0000.jpg')
h, w = frame.shape[:2]
print(f"图像尺寸: {w}×{h}")

# 预处理
preprocessor = ImagePreprocessor()
preprocessed = preprocessor.process(frame)
roi = preprocessed.roi_frame
edges = preprocessed.edges
enhanced = preprocessed.enhanced

roi_h, roi_w = edges.shape[:2]
print(f"ROI 尺寸: {roi_w}×{roi_h}")

# ========== 隔离沟检测诊断 ==========
print("\n===== 隔离沟检测 =====")
strip_top = int(roi_h * config.DITCH_STRIP_RATIO_BOTTOM)
print(f"分析区域: 下半部 y=[{strip_top}, {roi_h}] (DITCH_STRIP_RATIO_BOTTOM={config.DITCH_STRIP_RATIO_BOTTOM})")

roi_strip = roi[strip_top:, :]
hsv_strip = cv2.cvtColor(roi_strip, cv2.COLOR_BGR2HSV)
v_strip = hsv_strip[:, :, 2]
col_means = v_strip.mean(axis=0)

left = int(roi_w * config.DITCH_SEARCH_LEFT)
right = int(roi_w * config.DITCH_SEARCH_RIGHT)
print(f"搜索范围: x=[{left}, {right}] (DITCH_SEARCH_LEFT={config.DITCH_SEARCH_LEFT}, DITCH_SEARCH_RIGHT={config.DITCH_SEARCH_RIGHT})")

kernel = max(5, roi_w // 30)
smoothed = np.convolve(col_means, np.ones(kernel)/kernel, mode='same')
search_region = smoothed[left:right]
min_idx = int(np.argmin(search_region)) + left
min_val = smoothed[min_idx]
mean_val = np.median(smoothed[left:right])
darkness = mean_val - min_val

print(f"列均值范围: [{col_means.min():.0f}, {col_means.max():.0f}], 中位数: {np.median(col_means):.1f}")
print(f"搜索区内最暗列: x={min_idx}, V={min_val:.1f}, 中位数V={mean_val:.1f}")
print(f"暗度差值: {darkness:.1f} (阈值: {config.DITCH_V_DARK_THRESHOLD}) → {'✅通过' if darkness >= config.DITCH_V_DARK_THRESHOLD else '❌不通过'}")

# 边缘密度验证
col_slice = edges[strip_top:, max(0, min_idx-5):min(roi_w, min_idx+5)]
edge_density = np.count_nonzero(col_slice) / col_slice.size if col_slice.size > 0 else 0
print(f"该列边缘密度: {edge_density:.4f} (阈值: {config.DITCH_EDGE_DENSITY_MIN}) → {'✅通过' if edge_density >= config.DITCH_EDGE_DENSITY_MIN else '❌不通过'}")

# 打印前10个候选列（V值最低的）
print("\nV值最低的10列（候选）:")
indices = np.argsort(smoothed[left:right])[:10] + left
for i, idx in enumerate(indices):
    in_range = left <= idx <= right
    print(f"  #{i+1}: x={idx}, V={smoothed[idx]:.1f}, {'在搜索范围' if in_range else '超出范围'}")

# ========== 隔离带检测诊断 ==========
print("\n===== 隔离带检测 =====")
lines = cv2.HoughLinesP(edges, 1, np.pi/180,
    threshold=config.HOUGH_THRESHOLD,
    minLineLength=config.HOUGH_MIN_LINE_LEN,
    maxLineGap=config.HOUGH_MAX_LINE_GAP)
print(f"HoughLinesP: threshold={config.HOUGH_THRESHOLD}, minLen={config.HOUGH_MIN_LINE_LEN}, maxGap={config.HOUGH_MAX_LINE_GAP}")
print(f"检测到线段数: {len(lines) if lines is not None else 0}")

if lines is not None:
    angle_min, angle_max = config.LINE_ANGLE_MIN, config.LINE_ANGLE_MAX
    vertical_lines = 0
    left_candidates = 0
    right_candidates = 0
    debug_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = abs(np.arctan2(y2 - y1, x2 - x1 + 1e-9))
        is_vertical = angle_min <= angle <= angle_max
        x_bottom = x2 if y2 > y1 else x1
        is_left = x_bottom < roi_w * config.BARRIER_LEFT_MAX
        is_right = x_bottom > roi_w * config.BARRIER_RIGHT_MIN

        if is_vertical:
            vertical_lines += 1
            if is_left: left_candidates += 1
            if is_right: right_candidates += 1

        # 给匹配的线段画颜色
        if is_vertical:
            color = (0, 255, 0)  # 绿色=纵向
            if is_left: color = (255, 0, 0)  # 红色=左侧候选
            if is_right: color = (0, 0, 255)  # 蓝色=右侧候选
        else:
            color = (100, 100, 100)  # 灰色=非纵向
        cv2.line(debug_img, (x1, y1), (x2, y2), color, 1)

    print(f"纵向线段 (角度{angle_min:.1f}~{angle_max:.1f}rad): {vertical_lines}")
    print(f"左侧候选 (x<{roi_w * config.BARRIER_LEFT_MAX:.0f}): {left_candidates}")
    print(f"右侧候选 (x>{roi_w * config.BARRIER_RIGHT_MIN:.0f}): {right_candidates}")

    # 看角度分布
    angles = [abs(np.arctan2(y2-y1, x2-x1+1e-9)) for line in lines for x1,y1,x2,y2 in [line[0]]]
    if angles:
        angles = np.array(angles)
        print(f"\n线段角度分布: min={angles.min():.2f}rad, max={angles.max():.2f}rad, median={np.median(angles):.2f}rad")
        in_range = ((angles >= angle_min) & (angles <= angle_max)).mean()
        print(f"角度在[{angle_min:.2f}, {angle_max:.2f}]内的比例: {in_range:.1%}")

    cv2.imwrite('/home/nickwang/Projects/vision_obstacle_avoidance/data/frames/processed/diag_hough.jpg', debug_img)
    print(f"Hough诊断图已保存: data/frames/processed/diag_hough.jpg")

# 保存列投影图
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# 子图1: ROI原图
axes[0].imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
axes[0].axvline(x=left, color='r', linestyle='--', alpha=0.5, label=f'search L={left}')
axes[0].axvline(x=right, color='r', linestyle='--', alpha=0.5, label=f'search R={right}')
axes[0].axvline(x=min_idx, color='g', linewidth=2, label=f'min x={min_idx}')
axes[0].set_title('ROI with search region')
axes[0].legend(fontsize=8)

# 子图2: V通道列投影
axes[1].plot(smoothed, 'b-', linewidth=1, alpha=0.7, label='smoothed V')
axes[1].axvspan(left, right, alpha=0.1, color='yellow')
axes[1].axvline(x=min_idx, color='r', linestyle='--', label=f'min at x={min_idx}')
axes[1].axhline(y=mean_val, color='gray', linestyle=':', alpha=0.5)
axes[1].axhline(y=mean_val - config.DITCH_V_DARK_THRESHOLD, color='orange', linestyle=':', alpha=0.5, label=f'threshold (median-{config.DITCH_V_DARK_THRESHOLD})')
axes[1].set_title(f'V channel column projection (darkness={darkness:.1f}, threshold={config.DITCH_V_DARK_THRESHOLD})')
axes[1].legend(fontsize=8)

# 子图3: Canny边缘图
axes[2].imshow(edges, cmap='gray')
axes[2].set_title(f'Canny edges (edge density at x={min_idx}: {edge_density:.4f}, threshold={config.DITCH_EDGE_DENSITY_MIN})')

plt.tight_layout()
plt.savefig('/home/nickwang/Projects/vision_obstacle_avoidance/data/frames/processed/diag_ditch.png', dpi=100)
print(f"隔离沟诊断图已保存: data/frames/processed/diag_ditch.png")
