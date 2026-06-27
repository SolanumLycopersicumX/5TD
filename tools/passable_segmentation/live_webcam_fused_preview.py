#!/usr/bin/env python3
"""Live webcam preview for fused passable-road segmentation."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

DEFAULT_PASSABLE_CHECKPOINT = Path("runs/passable_ego/passable_ditch_artifact_v3_finetune/best_model.pt")
DEFAULT_BOUNDARY_CHECKPOINT = Path("runs/passable_ego/boundary_wall_aux_v2_no_testvideo/best_model.pt")


def parse_camera(value: str) -> int | str:
    """Return integer camera index for numeric input, otherwise keep the device path/URL."""
    return int(value) if value.isdigit() else value


def scale_to_display_width(width: int, height: int, max_width: int) -> tuple[int, int]:
    """Scale dimensions down to max_width while preserving aspect ratio."""
    width = int(width)
    height = int(height)
    max_width = int(max_width)
    if max_width <= 0 or width <= max_width:
        return width, height
    scale = max_width / float(width)
    return max_width, max(1, int(round(height * scale)))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", default="/dev/video0", help="camera index, V4L2 device, or video URL")
    parser.add_argument("--width", type=int, default=640, help="capture width")
    parser.add_argument("--height", type=int, default=480, help="capture height")
    parser.add_argument("--fps", type=int, default=15, help="requested capture FPS")
    parser.add_argument("--display-width", type=int, default=1280, help="maximum display window image width")
    parser.add_argument(
        "--display-backend",
        choices=("tk", "opencv"),
        default="tk",
        help="window backend; tk works with headless OpenCV builds, opencv requires HighGUI support",
    )
    parser.add_argument("--every-n", type=int, default=1, help="run model every N frames and reuse overlay between runs")
    parser.add_argument("--cpu", action="store_true", help="force CPU inference even if CUDA is available")
    parser.add_argument("--save-last", type=Path, default=None, help="optional path to save the latest overlay when exiting")
    parser.add_argument("--passable-checkpoint", type=Path, default=DEFAULT_PASSABLE_CHECKPOINT)
    parser.add_argument("--boundary-checkpoint", type=Path, default=DEFAULT_BOUNDARY_CHECKPOINT)
    return parser


def _require_runtime_modules() -> tuple[Any, Any, Any]:
    try:
        import cv2
        import numpy as np
        import torch
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing runtime dependency. Use the lerobot environment, for example:\n"
            "  /home/tomato/miniconda3/envs/lerobot/bin/python "
            "tools/passable_segmentation/live_webcam_fused_preview.py --camera /dev/video0"
        ) from exc
    return cv2, np, torch


def _load_runtime_helpers():
    from tools.passable_segmentation.train_boundary_wall import LABELS as BOUNDARY_LABELS
    from tools.passable_segmentation.train_passable_ditch import LABELS as PASSABLE_LABELS
    from tools.passable_segmentation.visualize_fused_passable_boundary import (
        fuse_passable_boundary_predictions,
        load_model,
        make_fused_overlay,
        predict_probabilities,
    )

    return {
        "BOUNDARY_LABELS": BOUNDARY_LABELS,
        "PASSABLE_LABELS": PASSABLE_LABELS,
        "fuse_passable_boundary_predictions": fuse_passable_boundary_predictions,
        "load_model": load_model,
        "make_fused_overlay": make_fused_overlay,
        "predict_probabilities": predict_probabilities,
    }


class _OpenCvDisplay:
    def __init__(self, cv2_module: Any, window_name: str) -> None:
        self._cv2 = cv2_module
        self._window_name = window_name
        self._cv2.namedWindow(self._window_name, self._cv2.WINDOW_NORMAL)

    def show(self, image_bgr: Any) -> bool:
        self._cv2.imshow(self._window_name, image_bgr)
        key = self._cv2.waitKey(1) & 0xFF
        return key not in (ord("q"), 27)

    def close(self) -> None:
        self._cv2.destroyWindow(self._window_name)


class _TkDisplay:
    def __init__(self, window_name: str) -> None:
        try:
            import tkinter as tk
            from PIL import Image, ImageTk
        except ModuleNotFoundError as exc:
            raise SystemExit(
                "Missing Tk/Pillow display dependency. Use the lerobot environment or pass "
                "--display-backend opencv if your OpenCV build supports GUI windows."
            ) from exc

        self._tk = tk
        self._Image = Image
        self._ImageTk = ImageTk
        try:
            self._root = tk.Tk()
        except tk.TclError as exc:
            raise SystemExit(
                "Could not open a Tk preview window. Run this from the desktop session with DISPLAY set, "
                "or use --display-backend opencv with a GUI-enabled OpenCV build."
            ) from exc
        self._root.title(window_name)
        self._root.protocol("WM_DELETE_WINDOW", self._request_close)
        self._root.bind("q", self._request_close)
        self._root.bind("<Escape>", self._request_close)
        self._label = tk.Label(self._root, bd=0)
        self._label.pack()
        self._photo = None
        self._closed = False

    def _request_close(self, _event: Any = None) -> None:
        self._closed = True

    def show(self, image_bgr: Any) -> bool:
        if self._closed:
            return False
        image_rgb = image_bgr[:, :, ::-1].copy()
        image = self._Image.fromarray(image_rgb)
        self._photo = self._ImageTk.PhotoImage(image=image)
        self._label.configure(image=self._photo)
        try:
            self._root.update_idletasks()
            self._root.update()
        except self._tk.TclError:
            return False
        return not self._closed

    def close(self) -> None:
        self._closed = True
        try:
            self._root.destroy()
        except self._tk.TclError:
            pass


def make_display_backend(kind: str, cv2_module: Any, window_name: str) -> Any:
    if kind == "opencv":
        return _OpenCvDisplay(cv2_module, window_name)
    if kind == "tk":
        return _TkDisplay(window_name)
    raise ValueError(f"Unknown display backend: {kind}")


def run_preview(args: argparse.Namespace) -> int:
    cv2, _np, torch = _require_runtime_modules()
    helpers = _load_runtime_helpers()

    if not args.passable_checkpoint.exists():
        raise FileNotFoundError(args.passable_checkpoint)
    if not args.boundary_checkpoint.exists():
        raise FileNotFoundError(args.boundary_checkpoint)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"[INFO] Loading models on {device}...")
    passable_model = helpers["load_model"](
        args.passable_checkpoint,
        expected_labels=helpers["PASSABLE_LABELS"],
        device=device,
    )
    boundary_model = helpers["load_model"](
        args.boundary_checkpoint,
        expected_labels=helpers["BOUNDARY_LABELS"],
        device=device,
    )

    camera = parse_camera(str(args.camera))
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera: {args.camera}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(args.width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(args.height))
    cap.set(cv2.CAP_PROP_FPS, int(args.fps))

    window_name = "Fused Passable Road Preview"
    preview_window = None
    latest_canvas = None
    frame_idx = 0
    last_time = time.monotonic()
    fps = 0.0
    every_n = max(1, int(args.every_n))

    print("[INFO] Press q or Esc in the preview window to quit.")
    try:
        preview_window = make_display_backend(args.display_backend, cv2, window_name)
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                print("[WARN] Camera frame read failed")
                time.sleep(0.05)
                continue

            run_model = latest_canvas is None or frame_idx % every_n == 0
            if run_model:
                image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                passable_probs = helpers["predict_probabilities"](passable_model, image_rgb, device)
                boundary_probs = helpers["predict_probabilities"](boundary_model, image_rgb, device)
                height, width = passable_probs.shape[1:]
                image_resized = cv2.resize(image_rgb, (width, height), interpolation=cv2.INTER_AREA)
                fused = helpers["fuse_passable_boundary_predictions"](passable_probs, boundary_probs)
                canvas_rgb = helpers["make_fused_overlay"](image_resized, fused)
                latest_canvas = cv2.cvtColor(canvas_rgb, cv2.COLOR_RGB2BGR)

            now = time.monotonic()
            dt = now - last_time
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else 1.0 / dt
            last_time = now

            display = latest_canvas.copy()
            text = f"device={device} fps={fps:.1f} camera={args.camera} q/Esc quit"
            cv2.putText(display, text, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(display, text, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)

            scaled_w, scaled_h = scale_to_display_width(display.shape[1], display.shape[0], args.display_width)
            if (scaled_w, scaled_h) != (display.shape[1], display.shape[0]):
                display = cv2.resize(display, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
            if preview_window is not None and not preview_window.show(display):
                break
            frame_idx += 1
    finally:
        if args.save_last is not None and latest_canvas is not None:
            args.save_last.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(args.save_last), latest_canvas)
            print(f"[OK] Saved latest overlay to {args.save_last}")
        cap.release()
        if preview_window is not None:
            preview_window.close()
    return 0


def main() -> None:
    args = build_arg_parser().parse_args()
    raise SystemExit(run_preview(args))


if __name__ == "__main__":
    main()
