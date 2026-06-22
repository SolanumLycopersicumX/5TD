#!/usr/bin/env python3
"""
工程师调试视图。
干净清晰: 每个面板独立显示一种数据, 无半透明叠加。
按 q 退出, 1-6 数字键切换重点面板。
"""
import sys, os, time, argparse
import cv2
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from hbdnet_rt.utils.config import load_config
from hbdnet_rt.utils.logger import get_logger
from hbdnet_rt.perception.preprocessor import ImagePreprocessor
from hbdnet_rt.perception.inference import PerceptionInference
from hbdnet_rt.mapping.occupancy_grid import OccupancyGrid
from hbdnet_rt.mapping.risk_grid import RiskGrid
from hbdnet_rt.planning.dwa import DWAPlanner
from hbdnet_rt.safety.state_machine import SafetyStateMachine

CLASS_COLORS = [(0, 140, 255), (0, 255, 0), (200, 200, 0), (150, 0, 255)]
CLASS_NAMES = ["vehicle", "worker", "suspended", "debris"]


class DebugView:
    """工程师调试视图 — 干净、分面板、无半透明覆盖。"""

    def __init__(self, cfg, use_camera=False, video_path=None):
        self.cfg = cfg
        self.preprocessor = ImagePreprocessor(cfg)
        self.infer = PerceptionInference(cfg)
        self.occup_grid = OccupancyGrid(cfg)
        self.risk_gen = RiskGrid(cfg)
        self.planner = DWAPlanner(cfg)
        self.safety = SafetyStateMachine(cfg)
        self.pose = [0.0, 0.0, 0.0]

        if use_camera:
            self.cap = cv2.VideoCapture(0)
        elif video_path:
            self.cap = cv2.VideoCapture(video_path)
        else:
            self.cap = None

        self.frame_count = 0
        self.fps_timer = time.time()
        self.fps = 0.0

    def run(self):
        logger = get_logger("debug")
        logger.info("调试视图启动 (q 退出, 1-6 切换面板)")

        while True:
            t0 = time.perf_counter()

            if self.cap is not None:
                ret, frame = self.cap.read()
                if not ret: break
                frame = cv2.resize(frame, (1280, 720))
            else:
                frame = np.full((720, 1280, 3), 55, dtype=np.uint8)
                cv2.putText(frame, "DEMO — NO CAMERA", (340, 380),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (100,100,100), 3)

            tensor = self.preprocessor(frame)
            perc = self.infer(tensor)
            occ = self.occup_grid.generate(perc)
            risk = self.risk_gen.generate(perc, occ["occupancy_grid"])
            risk_arr = risk["risk_grid"].numpy() if torch.is_tensor(risk["risk_grid"]) else risk["risk_grid"]
            dwa = self.planner.plan({
                "current_pose": self.pose, "current_velocity": 0.5,
                "risk_grid": risk_arr, "grid_extent": risk["metadata"],
            })
            saf = self.safety.evaluate({
                "overall_confidence": perc["confidence"]["overall"],
                "max_risk": risk["max_risk"],
                "boundary_distance": 5.0, "worker_distance": 10.0,
                "has_feasible_path": dwa["planner_status"] == "OK",
            })

            lat = (time.perf_counter() - t0) * 1000
            self.frame_count += 1
            if self.frame_count % 30 == 0:
                now = time.time()
                self.fps = 30 / (now - self.fps_timer + 1e-9)
                self.fps_timer = now

            display = self._render(frame, perc, dwa, saf, risk, lat)
            cv2.imshow("HBD-Net-RT Debug", display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if self.cap: self.cap.release()
        cv2.destroyAllWindows()

    # ═══════════════════════════════════════════

    def _render(self, frame, perc, dwa, saf, risk, lat):
        det = perc["detections"]
        ego_m = perc["ego_passable_mask"]
        hb_m = perc["hard_boundary_mask"]
        edge_m = perc["hard_boundary_edge"]
        rg = risk["risk_grid"]
        conf = perc["confidence"]
        state = saf["safety_state"]
        speed = dwa.get("target_speed", 0)
        steer = dwa.get("target_steering_angle", dwa.get("target_steering", 0))
        traj = dwa.get("selected_trajectory")
        extent = risk.get("metadata", {})

        # ── 上排 4 面板 (320×240 每个) ──
        p1 = self._panel_original(frame)
        p2 = self._panel_detections(frame, det)
        p3 = self._panel_mask(frame, ego_m, "Ego-Passable", (0, 180, 0))
        p4 = self._panel_mask(frame, hb_m, "Hard-Boundary", (0, 0, 220))
        row1 = np.hstack([p1, p2, p3, p4])

        # ── 下排 4 面板 ──
        p5 = self._panel_mask(frame, edge_m, "Edge", (200, 180, 0))
        p6 = self._panel_risk_grid(rg, traj, extent)
        p7 = self._panel_occupancy(rg, extent)
        p8 = self._panel_stats(conf, state, speed, steer, lat, dwa, saf)
        row2 = np.hstack([p5, p6, p7, p8])

        return np.vstack([row1, row2])

    # ══════════════════ 各面板 ══════════════════

    def _panel_original(self, frame):
        p = cv2.resize(frame, (320, 240))
        self._label(p, "1.Original")
        return p

    def _panel_detections(self, frame, det):
        """检测框: 纯色细线框 + 标签, 不做半透明覆盖。"""
        p = cv2.resize(frame, (320, 240))
        boxes = det.get("boxes")
        if boxes is not None and boxes.numel() > 0:
            scores = det.get("scores")
            labels = det.get("labels")
            for i in range(boxes.shape[0]):
                s = float(scores[i]) if scores.numel() > 0 else 0
                if s < 0.3: continue
                x1, y1, x2, y2 = boxes[i].tolist()
                x1 = int(x1 * 320 / 1280); y1 = int(y1 * 240 / 720)
                x2 = int(x2 * 320 / 1280); y2 = int(y2 * 240 / 720)
                lb = int(labels[i].item()) if labels.numel() > 0 else 0
                color = CLASS_COLORS[lb % len(CLASS_COLORS)]
                cv2.rectangle(p, (x1, y1), (x2, y2), color, 1)
                cv2.putText(p, f"{CLASS_NAMES[lb]}:{s:.2f}", (x1+2, y1-4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        self._label(p, "2.Detections")
        return p

    def _panel_mask(self, frame, mask, title, color):
        """分割 mask: 用轮廓线显示, 不做大面积覆盖。"""
        p = np.full((240, 320, 3), 30, dtype=np.uint8)
        if mask is None:
            self._label(p, f"3.{title} (none)")
            return p

        m = mask.squeeze().detach().cpu().numpy()
        if m.ndim == 3:
            # 多通道: 每通道不同颜色显示轮廓
            palette = [(0,180,0), (0,0,220), (0,180,220), (180,180,180)]
            for ch in range(min(m.shape[0], 4)):
                ch_m = m[ch]
                ch_m = cv2.resize(ch_m, (320, 240))
                _, binary = cv2.threshold((ch_m * 255).astype(np.uint8), 100, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(p, contours, -1, palette[ch], 1)
        else:
            m = cv2.resize(m, (320, 240))
            # 轮廓线显示
            _, binary = cv2.threshold((m * 255).astype(np.uint8), 100, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(p, contours, -1, color, 1)
            # 轮廓内部极淡填充
            mask_fill = np.zeros_like(p)
            cv2.drawContours(mask_fill, contours, -1, color, -1)
            p = cv2.addWeighted(p, 1.0, mask_fill, 0.08, 0)

        self._label(p, f"3.{title}")
        return p

    def _panel_risk_grid(self, rg, traj, extent):
        """风险栅格: 灰度图 + 轨迹线。"""
        p = np.full((240, 320, 3), 30, dtype=np.uint8)
        if rg is not None:
            r = rg.squeeze()
            if hasattr(r, 'numpy'): r = r.numpy()
            # 灰度: 0=黑(安全), 255=白(高风险)
            r_vis = (r * 255).astype(np.uint8)
            r_vis = cv2.resize(r_vis, (320, 240), interpolation=cv2.INTER_NEAREST)
            p = cv2.cvtColor(r_vis, cv2.COLOR_GRAY2BGR)

        # 轨迹
        if traj is not None and extent:
            x_min, x_max = extent.get("x_min", -2.5), extent.get("x_max", 2.5)
            y_min, y_max = extent.get("y_min", 0), extent.get("y_max", 8.0)
            pts = []
            for pt in traj:
                px = int((pt[0] - x_min) / (x_max - x_min) * 320)
                py = 239 - int((pt[1] - y_min) / (y_max - y_min) * 240)
                pts.append((max(0, min(319, px)), max(0, min(239, py))))
            for i in range(len(pts)-1):
                cv2.line(p, pts[i], pts[i+1], (0, 255, 255), 1)
            if pts:
                cv2.circle(p, pts[0], 3, (255, 255, 0), -1)
                cv2.circle(p, pts[-1], 3, (0, 255, 255), -1)

        # 标尺
        for y_m in [2, 4, 6]:
            py = 239 - int(y_m / 8.0 * 240)
            cv2.line(p, (0, py), (10, py), (180, 180, 180), 1)
            cv2.putText(p, f"{y_m}m", (12, py+4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180,180,180), 1)

        self._label(p, "5.Risk+Traj")
        return p

    def _panel_occupancy(self, rg, extent):
        """占用栅格: 二值黑白显示。"""
        p = np.full((240, 320, 3), 30, dtype=np.uint8)
        if rg is not None:
            r = rg.squeeze()
            if hasattr(r, 'numpy'): r = r.numpy()
            # 二值化: risk>0.7 → 黑色(占用), risk<0.3 → 白色(可通), 中间灰色
            occ_vis = np.full_like(r, 128, dtype=np.uint8)
            occ_vis[r > 0.7] = 0
            occ_vis[r < 0.3] = 255
            occ_vis = cv2.resize(occ_vis, (320, 240), interpolation=cv2.INTER_NEAREST)
            p = cv2.cvtColor(occ_vis, cv2.COLOR_GRAY2BGR)

            occ_pct = int((r < 0.3).mean() * 100)
            cv2.putText(p, f"Free: {occ_pct}%", (5, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1)

        self._label(p, "6.Occupancy")
        return p

    def _panel_stats(self, conf, state, speed, steer, lat, dwa, saf):
        """数据面板: 纯文字, 无图表。"""
        p = np.full((240, 320, 3), 25, dtype=np.uint8)

        rows = [
            ("STATE", state, self._state_color(state)),
            ("Speed", f"{speed:.2f} m/s", (200, 200, 200)),
            ("Steer", f"{steer:+.3f} rad", (200, 200, 200)),
            ("DWA", dwa.get("planner_status", "?"),
             (0, 200, 0) if dwa.get("planner_status") == "OK" else (0, 0, 220)),
            ("", "", (0,0,0)),
            ("Confidence", "", (180, 180, 180)),
            ("  detection", f"{conf.get('detection', 0):.3f}", (200, 200, 200)),
            ("  passable",  f"{conf.get('passable', 0):.3f}", (200, 200, 200)),
            ("  boundary",  f"{conf.get('boundary', 0):.3f}", (200, 200, 200)),
            ("  surface",   f"{conf.get('surface_risk', 0):.3f}", (200, 200, 200)),
            ("  overall",   f"{conf.get('overall', 0):.3f}", (200, 200, 200)),
            ("", "", (0,0,0)),
            ("Risk max", f"{risk_max_from_dwa(dwa):.3f}", (200, 200, 200)),
            (f"FPS:{self.fps:.0f}", f"Lat:{lat:.0f}ms", (150, 150, 150)),
        ]

        y = 10
        for label, value, color in rows:
            text = f"{label}: {value}" if label else value
            cv2.putText(p, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.35 if label.startswith("  ") else 0.4, color, 1)
            y += 16

        # 警告标记
        warnings = []
        if saf.get("brake"): warnings.append("BRAKE")
        if state in ("S3_STOP", "S4_MANUAL_TAKEOVER"): warnings.append("STOP")
        if conf.get("overall", 1) < 0.5: warnings.append("LOW_CONF")
        if dwa.get("planner_status") == "NO_PATH": warnings.append("NO_PATH")
        for i, w in enumerate(warnings):
            cv2.putText(p, f"! {w}", (10, 225), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (0, 0, 255), 1)

        self._label(p, "7.Stats")
        return p

    def _label(self, panel, text):
        cv2.putText(panel, text, (4, 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, (200, 200, 200), 1)

    def _state_color(self, state):
        return {"S0_NORMAL": (0,200,0), "S1_CAUTIOUS": (0,220,220),
                "S2_SLOWDOWN": (0,140,255), "S3_STOP": (0,0,255),
                "S4_MANUAL_TAKEOVER": (0,0,139)}.get(state, (128,128,128))


def risk_max_from_dwa(dwa):
    cb = dwa.get("cost_breakdown", {})
    return cb.get("max_risk", 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", "-c", action="store_true")
    parser.add_argument("--video", "-v")
    args = parser.parse_args()
    cfg = load_config()
    DebugView(cfg, use_camera=args.camera, video_path=args.video).run()


if __name__ == "__main__":
    main()
