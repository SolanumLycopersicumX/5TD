"""
工业场景纯视觉避障 Demo —— 主入口。
串联摄像头 → 预处理 → 可行驶区域检测 → 障碍检测 → 决策 → 控制 → 日志。

运行: python main.py
退出: 按 q / Esc 键, 或 SIGTERM / SIGINT
"""

import logging
import os
import signal
import socket
import sys
import time
import platform

import cv2
import numpy as np
import config

# 检测是否有显示器可用（工控机无头模式自动适配）
_HAS_DISPLAY = os.environ.get('DISPLAY', '') != ''

from camera_capture import CameraCapture
from preprocess import ImagePreprocessor
from lane_or_freespace_detector import FreeSpaceDetector
from obstacle_detector import ObstacleDetector
from decision_maker import DecisionMaker
from vehicle_controller import VehicleController
from logger import RuntimeLogger
from calibration import GroundCalibrator
from lane_boundary_detector import LaneBoundaryDetector
from debris_detector import DebrisDetector
from path_planner import PathPlanner
from utils import (create_decision_canvas, draw_decision_overlay,
                   Decision, FreeSpaceState, ObstacleState,
                   LaneBoundaryState, DebrisState, PathPlan,
                   DegradationStatus)


# ── 全局运行标志 ─────────────────────────────────────────────────────────
_running = True


def _handle_shutdown(signum, frame):
    """SIGTERM / SIGINT → 优雅退出。"""
    global _running
    name = signal.Signals(signum).name
    logging.warning("收到 %s，正在安全退出...", name)
    _running = False


# ── 结构化日志 ───────────────────────────────────────────────────────────

def _setup_logging():
    """
    初始化日志系统，输出到 stderr（由 systemd journald 自动捕获）。
    格式包含 ISO 8601 时间戳，方便 journalctl -o short-precise 查看。
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-7s] %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stderr,
    )
    logging.info("vision_obstacle_avoidance v1.0 启动 "
                 "(platform=%s)", platform.node())


# ── systemd watchdog ──────────────────────────────────────────────────────

# systemd 通过 NOTIFY_SOCKET 环境变量传递 socket 路径
_WATCHDOG_SOCK = os.environ.get("NOTIFY_SOCKET", "")
_WATCHDOG_USEC = int(os.environ.get("WATCHDOG_USEC", "0"))
# 在 WatchdogSec 的一半时间内喂狗，留足余量
_WATCHDOG_PING_INTERVAL = (_WATCHDOG_USEC / 2_000_000.0) if _WATCHDOG_USEC else 0
_WATCHDOG_ENABLED = bool(_WATCHDOG_SOCK and _WATCHDOG_USEC)


def _watchdog_ping():
    """
    向 systemd 发送 WATCHDOG=1。
    无外部依赖，直接使用 Unix socket 通信。
    非 systemd 环境下为空操作。
    """
    if not _WATCHDOG_ENABLED:
        return
    sock_path = _WATCHDOG_SOCK
    # systemd 抽象 socket 以 @ 开头，需要替换为 null byte
    addr = "\0" + sock_path[1:] if sock_path.startswith("@") else sock_path
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        sock.settimeout(0.5)
        sock.sendto(b"WATCHDOG=1", addr)
        sock.close()
    except OSError:
        pass  # watchdog 失败不阻塞主循环

def build_debug_display(original, preprocessed, free_state, obstacle_state,
                        decision, fps, latency_ms,
                        lane_state=None, path_plan=None):
    """
    拼接多窗口调试画面。
    布局: 上排 4 个小窗，下排 4 个中窗 + 决策面板。
    Phase 1 新增: LaneBoundary + PathPlan 面板。
    """
    h_small, w_small = 180, 320
    h_wide, w_wide = 240, 320

    def resize_to(img, target_h, target_w):
        if img is None or img.size == 0:
            return np.zeros((target_h, target_w, 3), dtype=np.uint8)
        return cv2.resize(img, (target_w, target_h))

    def gray_to_bgr(img):
        if img is None or img.size == 0:
            return np.zeros((100, 100, 3), dtype=np.uint8)
        if len(img.shape) == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img

    # ---- 上排 ----
    row1 = []
    # 原始画面
    orig = resize_to(original, h_small, w_small)
    cv2.putText(orig, "Original", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
    row1.append(orig)

    # ROI
    if preprocessed and preprocessed.debug_images:
        roi_img = gray_to_bgr(preprocessed.debug_images.get("1_roi", orig))
    else:
        roi_img = np.zeros((h_small, w_small, 3), dtype=np.uint8)
    roi_img = resize_to(roi_img, h_small, w_small)
    cv2.putText(roi_img, "ROI", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
    row1.append(roi_img)

    # 边缘
    if preprocessed and preprocessed.debug_images:
        edges_img = gray_to_bgr(preprocessed.debug_images.get("5_edges",
                                np.zeros((100,100,3), dtype=np.uint8)))
    else:
        edges_img = np.zeros((h_small, w_small, 3), dtype=np.uint8)
    edges_img = resize_to(edges_img, h_small, w_small)
    cv2.putText(edges_img, "Edges", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
    row1.append(edges_img)

    # Mask
    if preprocessed and preprocessed.debug_images:
        mask_img = gray_to_bgr(preprocessed.debug_images.get("6_mask",
                                np.zeros((100,100,3), dtype=np.uint8)))
    else:
        mask_img = np.zeros((h_small, w_small, 3), dtype=np.uint8)
    mask_img = resize_to(mask_img, h_small, w_small)
    cv2.putText(mask_img, "Binary Mask", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
    row1.append(mask_img)

    row1_display = np.hstack(row1)

    # ---- 下排 ----
    row2 = []
    # 可行驶区域
    freespace = resize_to(
        gray_to_bgr(free_state.debug_image if free_state else None),
        h_wide, w_wide)
    cv2.putText(freespace, "FreeSpace", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
    row2.append(freespace)

    # 障碍检测
    obstacle = resize_to(
        gray_to_bgr(obstacle_state.debug_image if obstacle_state else None),
        h_wide, w_wide)
    cv2.putText(obstacle, "Obstacle", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
    row2.append(obstacle)

    # 车道边界 (Phase 1 新增)
    lane_panel = resize_to(
        gray_to_bgr(lane_state.debug_image if lane_state else None),
        h_wide, w_wide)
    cv2.putText(lane_panel, "LaneBoundary", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
    row2.append(lane_panel)

    # 路径规划 (Phase 1 新增)
    path_panel = resize_to(
        gray_to_bgr(path_plan.debug_image if path_plan else None),
        h_wide, w_wide)
    cv2.putText(path_panel, "PathPlan", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
    row2.append(path_panel)

    # 决策面板
    panel = create_decision_canvas(h_wide, w_wide)
    draw_decision_overlay(panel, decision or Decision(),
                          free_state or FreeSpaceState(),
                          obstacle_state or ObstacleState(), fps, latency_ms)
    row2.append(panel)

    row2_display = np.hstack(row2)

    # ---- 拼接（统一下排行宽，补齐到上排宽度） ----
    if row1_display.shape[1] != row2_display.shape[1]:
        pad_w = row1_display.shape[1] - row2_display.shape[1]
        if pad_w > 0:
            row2_display = np.hstack([row2_display, np.zeros((row2_display.shape[0], pad_w, 3), dtype=np.uint8)])
        elif pad_w < 0:
            row1_display = np.hstack([row1_display, np.zeros((row1_display.shape[0], -pad_w, 3), dtype=np.uint8)])
    return np.vstack([row1_display, row2_display])


# ====================================================================


def main():
    # ── 日志系统 ──
    _setup_logging()

    print("=" * 60)
    print("  隧道施工场景纯视觉避障系统 v2.0")
    print("  WP2.1 — HybridNets + 多层安全防御")
    print(f"  平台: {platform.system()} {platform.release()}")
    if _WATCHDOG_ENABLED:
        print(f"  systemd watchdog: 已启用 "
              f"(间隔={_WATCHDOG_PING_INTERVAL:.1f}s)")
    print("=" * 60)

    # ── 注册信号处理器 ──
    signal.signal(signal.SIGTERM, _handle_shutdown)
    signal.signal(signal.SIGINT, _handle_shutdown)

    # ====================================================================
    # 模块初始化
    # ====================================================================

    # 第0层: 摄像头
    camera = CameraCapture()
    if not camera.is_opened():
        logging.error("摄像头不可用: %s", config.CAMERA_INDEX)
        print("[错误] 摄像头不可用。")
        return

    # 隧道出口防过曝：锁定相机曝光（配置项，默认关闭）
    if config.CAMERA_EXPOSURE_LOCK:
        camera.lock_exposure(config.CAMERA_EXPOSURE_VALUE)

    # 第1层: 预处理
    preprocessor = ImagePreprocessor()

    # Phase 2 新增预处理模块
    from exposure_detector import ExposureStateDetector, ExposureState
    from temporal_denoiser import TemporalDenoiser
    from dehazer import DarkChannelDehazer

    exposure_detector = ExposureStateDetector(
        overexposed_ratio=config.EXPOSURE_OVEREXPOSED_RATIO,
        underexposed_ratio=config.EXPOSURE_UNDEREXPOSED_RATIO,
        ae_delta_threshold=config.EXPOSURE_AE_DELTA_THRESHOLD,
    )
    temporal_denoiser = TemporalDenoiser(
        num_frames=config.TDNR_NUM_FRAMES,
        weights=config.TDNR_WEIGHTS,
        scene_change_threshold=config.TDNR_SCENE_CHANGE_THRESHOLD,
    )
    dehazer = DarkChannelDehazer(
        patch_size=config.DEHAZE_PATCH_SIZE,
        omega=config.DEHAZE_OMEGA,
        t0=config.DEHAZE_T0,
        guided_filter_radius=config.DEHAZE_GF_RADIUS,
        guided_filter_eps=config.DEHAZE_GF_EPS,
        downsample_factor=config.DEHAZE_DOWNSAMPLE,
    )

    # 第2层: DL 感知 (HBD-Net-RT → HybridNets → 传统 CV 逐级回退)
    dl_engine = None
    dl_engine_type = "none"

    # 优先: HBD-Net-RT (RepVGG-lite + 5 Head, PyTorch)
    try:
        from hbdnet_rt_engine import HBDNetRTEngine
        dl_engine = HBDNetRTEngine(
            use_gpu=config.HBDNET_USE_GPU,
            conf_threshold=config.HYBRIDNETS_CONF_THRESHOLD,
            iou_threshold=config.HYBRIDNETS_IOU_THRESHOLD,
        )
        dl_engine_type = "hbdnet_rt"
        print("[HBD-Net-RT] RepVGG-lite 模型加载成功，启用 DL 感知层 (随机权重)")
    except Exception as e:
        print(f"[HBD-Net-RT] 加载失败: {e}")

    # 回退: HybridNets ONNX (如果 HBD-Net-RT 不可用)
    if dl_engine is None:
        try:
            from hybridnets_engine import HybridNetsEngine
            dl_engine = HybridNetsEngine(
                onnx_path=config.HYBRIDNETS_ONNX_PATH,
                use_gpu=config.HYBRIDNETS_USE_GPU,
                conf_threshold=config.HYBRIDNETS_CONF_THRESHOLD,
                iou_threshold=config.HYBRIDNETS_IOU_THRESHOLD,
            )
            dl_engine_type = "hybridnets"
            print("[HybridNets] ONNX 模型加载成功，启用 DL 感知层")
        except Exception as e:
            print(f"[HybridNets] 模型加载失败: {e} — 将使用传统 CV 感知层")

    # 第2层回退: 传统 CV
    free_space_detector = FreeSpaceDetector()
    obstacle_detector = ObstacleDetector()
    calibrator = GroundCalibrator()
    lane_detector = LaneBoundaryDetector()
    debris_detector = DebrisDetector()

    # 第3层: 占用栅格
    from occupancy_grid import OccupancyGridFusion
    grid_fusion = OccupancyGridFusion(
        grid_width_m=config.OCCUPANCY_GRID_WIDTH_M,
        grid_length_m=config.OCCUPANCY_GRID_LENGTH_M,
        resolution_m=config.OCCUPANCY_GRID_RESOLUTION_M,
        safety_margin_m=config.SAFETY_MARGIN_M,
        debris_min_frames=config.OCCUPANCY_DEBRIS_MIN_FRAMES,
    )

    # 第4层: 路径规划 + 决策
    path_planner = PathPlanner()
    decision_maker = DecisionMaker()

    # 第5层: 安全降级
    from safety_degrader import SafetyDegrader, DegradationLevel
    safety_degrader = SafetyDegrader()
    safety_degrader.l1_confidence = config.SAFETY_L1_CONFIDENCE
    safety_degrader.l2_confidence = config.SAFETY_L2_CONFIDENCE
    safety_degrader.l3_confidence = config.SAFETY_L3_CONFIDENCE
    safety_degrader.l1_abnormal_frames = config.SAFETY_L1_ABNORMAL_FRAMES
    safety_degrader.l2_abnormal_frames = config.SAFETY_L2_ABNORMAL_FRAMES
    safety_degrader.l3_abnormal_frames = config.SAFETY_L3_ABNORMAL_FRAMES
    safety_degrader.l3_timeout_sec = config.SAFETY_L3_TIMEOUT_SEC
    safety_degrader.speed_limits.update({
        safety_degrader._level: config.SAFETY_SPEED_L0_KMH,
    })

    # 控制 + 日志
    vehicle_controller = VehicleController()
    rt_logger = RuntimeLogger()
    rt_logger.start()

    # ====================================================================
    # 主循环
    # ====================================================================
    frame_count = 0
    fps = 0.0
    fps_update_interval = 30
    fps_timer = time.time()
    last_watchdog_ping = time.time()

    logging.info("主循环开始 (v2.0: HybridNets + 多层安全防御)")
    print("\n[系统] 按 'q' 键退出\n")

    global _running

    try:
        while _running:
            t0 = time.time()

            # ---- watchdog ----
            if _WATCHDOG_ENABLED:
                now = t0
                if now - last_watchdog_ping >= _WATCHDOG_PING_INTERVAL:
                    _watchdog_ping()
                    last_watchdog_ping = now

            # ============================================================
            # 1. 图像获取
            # ============================================================
            frame, timestamp = camera.read()
            if frame is None:
                safety_degrader.report_camera(False)
                time.sleep(0.05)
                continue
            safety_degrader.report_camera(True)

            # ============================================================
            # 2. 预处理 (第1层: 增强 + 去雾 + 降噪)
            # ============================================================
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 2a. 曝光状态检测 (v2.2: 分区域分析)
            exp_zoned = exposure_detector.detect_zones(gray)
            exp_state = exp_zoned.global_state if exp_zoned else ExposureState.NORMAL

            # 2b. 时域降噪 (低照度时启用)
            if exp_state.value in ("underexposed",):
                frame = temporal_denoiser.process(frame)

            # 2c. 去雾 (过曝/光幕散射时跳过，效果不佳)
            if exp_state.value not in ("overexposed",):
                # 仅当对比度低时启用去雾
                if gray.std() < 60:
                    frame = dehazer.process(frame)

            # 2d. 传统预处理 (ROI + Canny + HSV)
            preprocessed = preprocessor.process(frame)

            # ============================================================
            # 3. 消失点累积 (非阻塞)
            # ============================================================
            if not calibrator.calibrated:
                vp = calibrator.estimate_vp_from_edges(
                    preprocessed.edges,
                    roi_offset_x=int(frame.shape[1] * config.ROI_LEFT_RATIO),
                    roi_offset_y=int(frame.shape[0] * config.ROI_TOP_RATIO),
                )
                if vp is not None:
                    calibrator.accumulate(vp)

            # ============================================================
            # 4. 感知层 (第2层: DL + 传统CV 双轨)
            # ============================================================
            dl_output = None
            dl_confidence = 0.0

            if dl_engine is not None:
                try:
                    dl_output = dl_engine.infer(frame)
                    dl_confidence = dl_output.confidence
                    safety_degrader.report_inference(True)
                except Exception as e:
                    logging.warning("HybridNets 推理失败: %s, 回退传统CV", e)
                    safety_degrader.report_inference(False)
                    dl_output = None

            # 传统CV (始终运行, 用于回退 + 在线对比)
            lane_state = lane_detector.detect(
                preprocessed.roi_frame,
                preprocessed.edges,
                preprocessed.enhanced,
                calibrator,
            )
            free_space_state = free_space_detector.detect(
                preprocessed.roi_frame,
                preprocessed.edges,
                preprocessed.binary_mask,
                lane_state if dl_output is None else None,
            )
            obstacle_state = obstacle_detector.detect(
                preprocessed.roi_frame,
                preprocessed.edges,
                preprocessed.binary_mask,
            )

            # 融合: 优先使用 DL 结果, 低置信度时融合传统CV
            if dl_output is not None and dl_confidence >= 0.30:
                # 使用 DL 分割结果更新可行驶区域评分
                if dl_output.drivable_mask is not None:
                    h, w = dl_output.drivable_mask.shape
                    chunk_h = h // 3
                    for i, key in enumerate(['left', 'center', 'right']):
                        y1, y2 = i * chunk_h, (i + 1) * chunk_h
                        chunk = dl_output.drivable_mask[y1:y2, :]
                        score = float(chunk.mean()) / 255.0
                        if key == 'left':
                            free_space_state.left_free_score = max(
                                free_space_state.left_free_score, score)
                        elif key == 'center':
                            free_space_state.center_free_score = max(
                                free_space_state.center_free_score, score)
                        elif key == 'right':
                            free_space_state.right_free_score = max(
                                free_space_state.right_free_score, score)
                free_space_state.confidence = max(free_space_state.confidence, dl_confidence)

            # ============================================================
            # 5. 占用栅格融合 (第3层)
            # ============================================================
            occupancy_grid = grid_fusion.fuse(
                drivable_mask=dl_output.drivable_mask if dl_output else None,
                lane_mask=dl_output.lane_mask if dl_output else None,
                bboxes=dl_output.bboxes if dl_output else [],
                calibrator=calibrator,
                lane_boundary=lane_state,
            )

            # 碎石检测
            debris_state = debris_detector.detect(
                preprocessed.gray,
                preprocessed.edges,
                lane_state,
                calibrator,
            )

            # ============================================================
            # 6. 安全降级评估 (第5层)
            # ============================================================
            degradation_level = safety_degrader.evaluate(
                dl_confidence=dl_confidence,
                exposure_state=exp_state,
                cv_lane_valid=lane_state.is_valid,
            )

            # v2.2: 远区过曝强制降级 — 隧道出口看不清远处障碍物
            if exp_zoned and exp_zoned.is_far_blind:
                lvl = degradation_level.value if hasattr(degradation_level, 'value') else 0
                if lvl < 1:
                    degradation_level = DegradationLevel.L1_CAUTION
                    if config.DEBUG_VIEW and _HAS_DISPLAY:
                        print(f"[曝光] 远区盲区 → 强制 L1_CAUTION (far_ratio={exp_zoned.far_overexposed_ratio:.2f})")

            # ============================================================
            # 7. 路径规划 (第4层, 受安全降级约束)
            # ============================================================
            path_plan = path_planner.plan(
                lane_state, debris_state,
                free_space_state, obstacle_state,
                calibrator,
            )

            # 安全间距 × 降级乘数
            path_plan.clearance_cm *= safety_degrader.clearance_multiplier

            # ============================================================
            # 8. 决策 (第4层)
            # ============================================================
            latency_ms = (time.time() - t0) * 1000

            deg_status = DegradationStatus(
                level=safety_degrader.level.value,
                level_name=safety_degrader.level.name,
                speed_limit_kmh=safety_degrader.speed_limit_kmh,
                clearance_multiplier=safety_degrader.clearance_multiplier,
                allow_detour=safety_degrader.allow_detour,
                should_stop=safety_degrader.should_stop,
                needs_takeover=safety_degrader.needs_takeover,
                summary=safety_degrader.status_summary(),
            )

            decision = decision_maker.decide(
                free_space_state, obstacle_state, latency_ms, fps,
                debris_state, path_plan, lane_state,
                degradation_status=deg_status,
            )

            # ============================================================
            # 9. 下发控制
            # ============================================================
            vehicle_controller.send(decision)

            # ============================================================
            # 10. 日志 + FPS
            # ============================================================
            rt_logger.update(frame_count + 1, fps, latency_ms,
                            free_space_state, obstacle_state, decision, frame,
                            lane_state, debris_state, path_plan)

            frame_count += 1
            if frame_count % fps_update_interval == 0:
                now = time.time()
                fps = fps_update_interval / (now - fps_timer + 1e-9)
                fps_timer = now

            # ============================================================
            # 11. 可视化调试
            # ============================================================
            if config.DEBUG_VIEW and _HAS_DISPLAY:
                display = build_debug_display(
                    frame, preprocessed, free_space_state,
                    obstacle_state, decision, fps, latency_ms,
                    lane_state, path_plan,
                )
                # 叠加安全状态
                h, w = display.shape[:2]
                status_color = {
                    0: (0, 255, 0), 1: (0, 255, 255),
                    2: (0, 165, 255), 3: (0, 0, 255), 4: (0, 0, 255),
                }.get(safety_degrader.level.value, (255, 255, 255))
                cv2.putText(display, f"SAFETY: {safety_degrader.level.name}",
                           (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                           0.5, status_color, 1)
                cv2.imshow("Vision Obstacle Avoidance v2.0", display)

            # ---- 退出 ----
            if _HAS_DISPLAY:
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    print("\n[系统] 用户按键退出")
                    break

    except Exception as e:
        print(f"\n[异常] {e}")
        import traceback
        traceback.print_exc()
    finally:
        # ---- 安全退出 ----
        print("[系统] 正在安全退出...")
        vehicle_controller.emergency_stop()
        safety_degrader.force_level(safety_degrader._level.__class__.L4_TAKEOVER)
        camera.release()
        rt_logger.close()
        cv2.destroyAllWindows() if _HAS_DISPLAY else None
        print("[系统] 退出完成。")


if __name__ == "__main__":
    main()
