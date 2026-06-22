#!/usr/bin/env python3
"""
可视化工具。
显示原图 + detection boxes + ego_passable / hard_boundary / edge masks
+ risk_grid + DWA selected trajectory。
支持: 单张图片 OR 视频逐帧 OR 纯感知输出 (不依赖摄像头)。
"""
import sys, os, argparse
import numpy as np
import cv2
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from hbdnet_rt.utils.config import load_config
from hbdnet_rt.utils.logger import get_logger
from hbdnet_rt.perception.inference import PerceptionInference
from hbdnet_rt.mapping.occupancy_grid import OccupancyGrid
from hbdnet_rt.mapping.risk_grid import RiskGrid
from hbdnet_rt.planning.dwa import DWAPlanner
from hbdnet_rt.safety.state_machine import SafetyStateMachine


# ── 颜色 ──
CLASS_COLORS = {
    0: (0, 100, 255),   # construction_vehicle → orange
    1: (0, 255, 0),     # worker → green
    2: (255, 200, 0),   # suspended_object → cyan
    3: (255, 0, 100),   # falling_debris → purple-red
}
CLASS_NAMES = {0: "vehicle", 1: "worker", 2: "suspended", 3: "debris"}


def draw_detections(img, detections):
    """在图像上绘制检测框。"""
    if detections is None:
        return img
    boxes = detections.get("boxes")
    scores = detections.get("scores")
    labels = detections.get("labels")
    if boxes is None or boxes.numel() == 0:
        return img

    out = img.copy()
    for i in range(min(boxes.shape[0], 50)):
        x1, y1, x2, y2 = boxes[i].tolist()
        score = float(scores[i]) if scores.numel() > 0 else 0.0
        label = int(labels[i].item()) if labels.numel() > 0 else 0
        color = CLASS_COLORS.get(label, (200, 200, 200))
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        text = f"{CLASS_NAMES.get(label, '?')}:{score:.2f}"
        cv2.putText(out, text, (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return out


def draw_mask_overlay(img, mask, color, alpha=0.4):
    """半透明叠加 mask。"""
    if mask is None or mask.numel() == 0:
        return img
    m = mask.squeeze().cpu().numpy()
    if m.ndim == 2:
        m = (m > 0.5).astype(np.uint8)
    elif m.ndim == 3:
        m = m.max(axis=0)  # 多类取 max
        m = (m > 0.5).astype(np.uint8)
    m_resized = cv2.resize(m, (img.shape[1], img.shape[0]))
    overlay = img.copy()
    overlay[m_resized > 0] = color
    return cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)


def draw_risk_grid(risk_grid, size=(300, 480)):
    """将 risk_grid 渲染为彩色热力图。"""
    if risk_grid is None:
        return np.zeros((size[1], size[0], 3), dtype=np.uint8)
    rg = risk_grid.squeeze()
    if hasattr(rg, 'numpy'):
        rg = rg.numpy()
    if rg.ndim == 2:
        rg = rg.T  # transpose for display
    rg = cv2.resize(rg, size, interpolation=cv2.INTER_NEAREST)
    # 灰度 → 彩色热力 (绿→黄→红)
    heat = cv2.applyColorMap((rg * 255).astype(np.uint8), cv2.COLORMAP_JET)
    cv2.putText(heat, "Risk Grid", (5, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return heat


def draw_trajectory(grid_img, trajectory, extent, grid_size=(300, 480)):
    """在 risk_grid 上绘制 DWA 选中轨迹。"""
    if trajectory is None:
        return grid_img
    traj = np.array(trajectory)
    if traj.ndim < 2:
        return grid_img

    h, w = grid_img.shape[:2]
    # grid 坐标系 → 像素
    x_min, x_max = extent.get("x_min", -2.5), extent.get("x_max", 2.5)
    y_min, y_max = extent.get("y_min", 0.0), extent.get("y_max", 8.0)

    pts = []
    for pt in traj:
        px = int((pt[0] - x_min) / (x_max - x_min) * w)
        py = int((pt[1] - y_min) / (y_max - y_min) * h)
        # grid y→screen y 翻转
        py = h - 1 - py
        px = max(0, min(w - 1, px))
        py = max(0, min(h - 1, py))
        pts.append((px, py))

    for i in range(len(pts) - 1):
        cv2.line(grid_img, pts[i], pts[i + 1], (0, 255, 255), 2)
    if pts:
        cv2.circle(grid_img, pts[-1], 4, (0, 255, 255), -1)
        cv2.circle(grid_img, pts[0], 4, (255, 255, 0), -1)
    return grid_img


def build_display(image, perception_output, risk_result, dwa_result, safety_result):
    """组装完整可视化面板。"""
    h_main = 480
    det_img = draw_detections(image, perception_output.get("detections"))
    det_img = cv2.resize(det_img, (640, h_main))

    # 三列 masks
    ego_m = perception_output.get("ego_passable_mask")
    hb_m = perception_output.get("hard_boundary_mask")
    edge_m = perception_output.get("hard_boundary_edge")

    ego_vis = draw_mask_overlay(image, ego_m, (0, 255, 0), alpha=0.5)
    hb_vis = draw_mask_overlay(image, hb_m, (255, 0, 0), alpha=0.5)
    edge_vis = draw_mask_overlay(image, edge_m, (255, 255, 0), alpha=0.5)

    ego_vis = cv2.resize(ego_vis, (320, 240))
    hb_vis = cv2.resize(hb_vis, (320, 240))
    edge_vis = cv2.resize(edge_vis, (320, 240))
    cv2.putText(ego_vis, "Ego-Passable", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
    cv2.putText(hb_vis, "Hard-Boundary", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,0), 1)
    cv2.putText(edge_vis, "Edge", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,0), 1)
    masks_row = np.hstack([ego_vis, hb_vis, edge_vis])

    # Risk Grid + Trajectory
    risk = risk_result.get("risk_grid")
    extent = risk_result.get("metadata", {})
    risk_vis = draw_risk_grid(risk)
    risk_vis = draw_trajectory(
        risk_vis, dwa_result.get("selected_trajectory"), extent)
    risk_vis = cv2.resize(risk_vis, (320, 240))

    # Info panel
    info = np.zeros((240, 320, 3), dtype=np.uint8)
    conf = perception_output.get("confidence", {})
    safety_state = safety_result.get("safety_state", "?")
    y = 20
    for key, val in [
        ("State", safety_state),
        ("Speed", f"{dwa_result.get('target_speed',0):.2f} m/s"),
        ("Steer", f"{dwa_result.get('target_steering',0):.2f} rad"),
        ("Status", dwa_result.get("planner_status", "?")),
        ("Conf", f"{conf.get('overall',0):.3f}"),
        ("MaxRisk", f"{risk_result.get('max_risk',0):.2f}"),
        ("Brake", str(safety_result.get("brake", False))),
    ]:
        cv2.putText(info, f"{key}: {val}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        y += 22

    bottom_row = np.hstack([risk_vis, info])

    # 拼接
    if det_img.shape[1] != masks_row.shape[1]: det_img = cv2.resize(det_img, (masks_row.shape[1], det_img.shape[0]))
    if bottom_row.shape[1] != masks_row.shape[1]: bottom_row = cv2.resize(bottom_row, (masks_row.shape[1], bottom_row.shape[0]))
    display = np.vstack([det_img, masks_row, bottom_row])
    return display


def process_single_image(image_path, infer, occup_grid, risk_grid_gen, planner, safety):
    """处理单张图片。"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图片: {image_path}")
        return
    img = cv2.resize(img, (640, 384))
    tensor = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0

    perc_out = infer(tensor)
    occ = occup_grid.generate(perc_out)
    risk = risk_grid_gen.generate(perc_out, occ["occupancy_grid"])
    dwa_out = planner.plan({
        "current_pose": [0, 0, 0], "current_velocity": 0.5,
        "risk_grid": risk["risk_grid"], "grid_extent": risk["metadata"],
    })
    safety_out = safety.evaluate({
        "overall_confidence": perc_out["confidence"]["overall"],
        "max_risk": risk["max_risk"],
        "boundary_distance": 5.0, "worker_distance": 10.0,
        "has_feasible_path": dwa_out["planner_status"] == "OK",
    })

    display = build_display(img, perc_out, risk, dwa_out, safety_out)
    cv2.imshow("HBD-Net-RT Visualization", display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="HBD-Net-RT 可视化工具")
    parser.add_argument("--input", "-i", help="输入图片路径")
    parser.add_argument("--output", "-o", help="输出图片路径 (可选)")
    args = parser.parse_args()

    logger = get_logger("visualize")
    cfg = load_config()
    infer = PerceptionInference(cfg)
    occup_grid = OccupancyGrid(cfg)
    risk_grid_gen = RiskGrid(cfg)
    planner = DWAPlanner(cfg)
    safety = SafetyStateMachine(cfg)

    if args.input:
        process_single_image(args.input, infer, occup_grid, risk_grid_gen, planner, safety)
        return

    # 无输入: 用随机 tensor 测试可视化管线
    logger.info("无输入图片，使用随机 tensor 测试可视化管线...")
    dummy = torch.randn(1, 3, 384, 640)
    perc_out = infer(dummy)
    occ = occup_grid.generate(perc_out)
    risk = risk_grid_gen.generate(perc_out, occ["occupancy_grid"])
    dwa_out = planner.plan({
        "current_pose": [0, 0, 0], "current_velocity": 0.5,
        "risk_grid": risk["risk_grid"], "grid_extent": risk["metadata"],
    })
    safety_out = safety.evaluate({
        "overall_confidence": perc_out["confidence"]["overall"],
        "max_risk": risk["max_risk"],
        "boundary_distance": 5.0, "worker_distance": 10.0,
        "has_feasible_path": dwa_out["planner_status"] == "OK",
    })

    # 生成一张假的"原图"
    fake_img = (dummy.squeeze(0).permute(1,2,0).numpy() * 255).astype(np.uint8)
    fake_img = cv2.cvtColor(fake_img[:, :, ::-1], cv2.COLOR_RGB2BGR)  # RGB→BGR
    fake_img = cv2.resize(fake_img, (640, 384))

    display = build_display(fake_img, perc_out, risk, dwa_out, safety_out)

    if args.output:
        cv2.imwrite(args.output, display)
        logger.info(f"已保存: {args.output}")
    else:
        cv2.imshow("HBD-Net-RT Visualization (demo)", display)
        logger.info("按任意键关闭...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    logger.info("✅ 可视化管线 OK")


if __name__ == "__main__":
    main()
