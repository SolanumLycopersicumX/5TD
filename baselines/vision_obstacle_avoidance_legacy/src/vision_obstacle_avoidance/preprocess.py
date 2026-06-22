"""
图像预处理模块。
对原始帧做 ROI 裁剪 → 灰度化 → CLAHE 增强 → 高斯滤波 → Canny 边缘 → 二值分割 → 形态学处理。
"""

import cv2
import numpy as np
import config
from utils import PreprocessResult


class ImagePreprocessor:
    """
    图像预处理器。
    输入：原始 BGR 图像
    输出：PreprocessResult（包含 roi_frame, gray, enhanced, edges, binary_mask, debug_images）
    """

    def __init__(self):
        if config.GAUSSIAN_BLUR_SIZE % 2 == 0:
            raise ValueError(f"GAUSSIAN_BLUR_SIZE 必须为奇数，当前值: {config.GAUSSIAN_BLUR_SIZE}")
        # CLAHE: 自适应直方图均衡化，解决工业场景暗光/反光/阴影问题
        self.clahe = cv2.createCLAHE(
            clipLimit=config.CLAHE_CLIP_LIMIT,
            tileGridSize=(config.CLAHE_TILE_SIZE, config.CLAHE_TILE_SIZE),
        )
        self.morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (config.MORPH_KERNEL_SIZE, config.MORPH_KERNEL_SIZE),
        )

    def process(self, frame: np.ndarray) -> PreprocessResult:
        result = PreprocessResult()
        debug = {}

        if frame is None or frame.size == 0:
            return result

        h, w = frame.shape[:2]
        debug["0_original"] = frame.copy()

        # ---- 1. ROI 裁剪 ----
        # 取画面下半部分（车前地面区域），裁剪天空和远景
        roi_y1 = int(h * config.ROI_TOP_RATIO)
        roi_y2 = int(h * config.ROI_BOTTOM_RATIO)
        roi_x1 = int(w * config.ROI_LEFT_RATIO)
        roi_x2 = int(w * config.ROI_RIGHT_RATIO)

        roi_frame = frame[roi_y1:roi_y2, roi_x1:roi_x2].copy()
        if roi_frame.size == 0:
            return result
        result.roi_frame = roi_frame
        roi_h, roi_w = roi_frame.shape[:2]
        debug["1_roi"] = roi_frame.copy()

        # ---- 2. 灰度化 ----
        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        result.gray = gray
        debug["2_gray"] = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # ---- 3. CLAHE 光照增强 ----
        # 不平滑光照下（隧道、反光、阴影），CLAHE 局部增强对比度
        enhanced = self.clahe.apply(gray)
        result.enhanced = enhanced
        debug["3_clahe"] = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

        # ---- 4. 高斯滤波降噪 ----
        blurred = cv2.GaussianBlur(enhanced, (config.GAUSSIAN_BLUR_SIZE, config.GAUSSIAN_BLUR_SIZE), 0)
        debug["4_blurred"] = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)

        # ---- 5. Canny 边缘检测 ----
        edges = cv2.Canny(blurred, config.CANNY_LOW, config.CANNY_HIGH)
        result.edges = edges
        debug["5_edges"] = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        # ---- 6. HSV 地面分割 ----
        # 区分地面（灰色/水泥色）和非地面物体（颜色差异较大的障碍物）
        hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
        ground_low = np.array(config.GROUND_HSV_LOW)
        ground_high = np.array(config.GROUND_HSV_HIGH)
        ground_mask = cv2.inRange(hsv, ground_low, ground_high)

        # ---- 7. 形态学操作 ----
        # 开运算去小噪点 → 闭运算填小空洞
        binary_mask = cv2.morphologyEx(ground_mask, cv2.MORPH_OPEN,
                                       self.morph_kernel, iterations=config.MORPH_OPEN_ITER)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE,
                                       self.morph_kernel, iterations=config.MORPH_CLOSE_ITER)
        result.binary_mask = binary_mask
        debug["6_mask"] = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)

        result.debug_images = debug
        return result
