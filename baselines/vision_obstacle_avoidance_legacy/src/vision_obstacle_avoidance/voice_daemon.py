#!/usr/bin/env python3
"""
语音输入守护进程 (Wayland 兼容版)。
后台运行，接收 SIGUSR1 信号来切换录音状态。
配合 GNOME 自定义快捷键使用。

用法:
    python voice_daemon.py &          # 后台运行
    kill -USR1 $(cat /tmp/voice_daemon.pid)  # 切换录音

GNOME 快捷键设置:
    Ctrl+Alt+Space → bash -c 'kill -USR1 $(cat /tmp/voice_daemon.pid)'
"""

import argparse
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import config
from hardware_discovery import get_best_audio_device

# ---- 配置 ----
PROJECT_DIR = Path(__file__).parent

# 临时音频文件：/dev/shm 内存盘（回退到 /tmp）
_temp_dir = Path(config.AUDIO_TEMP_DIR) if config.AUDIO_TEMP_DIR else Path("/dev/shm")
if not _temp_dir.exists() or not os.access(_temp_dir, os.W_OK):
    _temp_dir = Path("/tmp")
AUDIO_FILE = _temp_dir / ".voice_daemon_temp.wav"

PID_FILE = Path("/tmp/voice_daemon.pid")
USB_MIC_DEVICE = config.AUDIO_DEVICE or get_best_audio_device(fallback="default")
DEFAULT_MODEL = "small"

# 全局状态
recording_proc: subprocess.Popen | None = None
recording_start: float = 0.0
model = None
model_lock = threading.Lock()


# ═══════════════════════════════════════════════════════════════════════
#  通知 + 音效
# ═══════════════════════════════════════════════════════════════════════

def notify(title: str, body: str = "", urgency: str = "normal"):
    """桌面通知。"""
    try:
        subprocess.run(
            ["notify-send", "-u", urgency, title, body],
            timeout=2, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass


def beep():
    """提示音 (同时用多种方式确保听到)。"""
    try:
        subprocess.run(["paplay", "/usr/share/sounds/freedesktop/stereo/message.oga"],
                       timeout=2, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass
    # fallback
    sys.stdout.write("\a")
    sys.stdout.flush()


# ═══════════════════════════════════════════════════════════════════════
#  录音
# ═══════════════════════════════════════════════════════════════════════

def record_start():
    global recording_proc, recording_start
    if AUDIO_FILE.exists():
        AUDIO_FILE.unlink()
    recording_proc = subprocess.Popen(
        ["arecord", "-D", USB_MIC_DEVICE, "-f", "S16_LE",
         "-r", "16000", "-c", "1", "-t", "wav", str(AUDIO_FILE)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    recording_start = time.time()
    beep()
    notify("🔴 录音中...", "再次按 Ctrl+Alt+Space 停止转写", "low")


def record_stop() -> float | None:
    global recording_proc, recording_start
    if recording_proc is None:
        return None
    proc = recording_proc
    recording_proc = None
    duration = time.time() - recording_start
    proc.terminate()
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    beep()
    if not AUDIO_FILE.exists() or AUDIO_FILE.stat().st_size < 1000:
        notify("⚠️ 录音太短", "请说至少 0.5 秒", "critical")
        return None
    return duration


# ═══════════════════════════════════════════════════════════════════════
#  转写 + 剪贴板
# ═══════════════════════════════════════════════════════════════════════

def transcribe_audio() -> str:
    from faster_whisper import WhisperModel
    global model

    with model_lock:
        segments, _ = model.transcribe(
            str(AUDIO_FILE), beam_size=5, language=None,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )
        text = "".join(seg.text.strip() for seg in segments).strip()

    if AUDIO_FILE.exists():
        AUDIO_FILE.unlink()
    return text


def copy_to_clipboard(text: str) -> bool:
    import shutil
    for cmd, args in [("xclip", ["-selection", "clipboard"]), ("wl-copy", [])]:
        if shutil.which(cmd):
            try:
                subprocess.run([cmd] + args, input=text, text=True, timeout=3)
                return True
            except Exception:
                pass
    return False


def on_transcribe_done(text: str, duration: float):
    """转写完成回调 (在后台线程中执行)。"""
    if text:
        copied = copy_to_clipboard(text)
        preview = text[:80] + ("..." if len(text) > 80 else "")
        status = "✅ 已复制到剪贴板" if copied else "📋 转写完成"
        notify(status, f"({int(duration)}s) {preview}", "normal")
    else:
        notify("📝 未检测到语音", f"录制了 {duration:.0f}s", "low")


# ═══════════════════════════════════════════════════════════════════════
#  信号处理 (核心)
# ═══════════════════════════════════════════════════════════════════════

def on_toggle_signal(signum, frame):
    """收到 SIGUSR1 → 切换录音状态。"""
    global recording_proc

    if recording_proc is not None:
        # 停止 + 异步转写
        duration = record_stop()
        if duration is not None and duration >= 0.5:
            def _run():
                try:
                    notify("📝 转写中...", f"录制了 {duration:.1f}s", "low")
                    text = transcribe_audio()
                    on_transcribe_done(text, duration)
                except Exception as e:
                    notify("❌ 转写失败", str(e)[:100], "critical")
            threading.Thread(target=_run, daemon=True).start()
    else:
        record_start()


# ═══════════════════════════════════════════════════════════════════════
#  主入口
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="语音输入守护进程 (Wayland)")
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL,
                        choices=["tiny", "small", "medium", "large-v3"],
                        help=f"模型大小 (默认: {DEFAULT_MODEL})")
    args = parser.parse_args()

    # 写 PID 文件
    PID_FILE.write_text(str(os.getpid()))

    # 注册信号处理器
    signal.signal(signal.SIGUSR1, on_toggle_signal)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    signal.signal(signal.SIGINT, lambda *_: sys.exit(0))

    # 加载模型
    print(f"⏳ 加载语音模型 ({args.model})...", end="", flush=True)
    from faster_whisper import WhisperModel
    global model
    model = WhisperModel(args.model, device="cpu", compute_type="int8")
    print(" 完成 ✅")

    print(f"""
╔═══════════════════════════════════════════════════╗
║     🎤 语音守护进程 (PID: {os.getpid()})              ║
╠═══════════════════════════════════════════════════╣
║  Ctrl+Alt+Space  → 开始/停止录音                   ║
║  停止后自动转写并复制到剪贴板, Ctrl+V 粘贴         ║
║  kill {os.getpid()}  → 退出守护进程                ║
╚═══════════════════════════════════════════════════╝
""")

    notify("🎤 语音就绪", f"PID: {os.getpid()}\nCtrl+Alt+Space 开始录音", "low")

    # 等待信号 (阻塞)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        if recording_proc is not None:
            recording_proc.terminate()
            recording_proc.wait()
        if AUDIO_FILE.exists():
            AUDIO_FILE.unlink()
        if PID_FILE.exists():
            PID_FILE.unlink()
        print("\n👋 语音守护进程已退出。")


if __name__ == "__main__":
    main()
