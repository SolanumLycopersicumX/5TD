#!/usr/bin/env python3
"""
全局热键语音输入守护进程。
在后台运行，按 Ctrl+Alt+Space 开始/停止录音，
停止后自动转写并复制到剪贴板。

用法:
    python voice_hotkey.py           # 前台运行 (Ctrl+C 退出)
    python voice_hotkey.py &         # 后台运行
    python voice_hotkey.py -m medium # 使用更准确的模型

热键:
    Ctrl+Alt+Space   → 切换录音 (开始/停止)
    Ctrl+Alt+Q       → 退出守护进程
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

from pynput import keyboard
import config
from hardware_discovery import get_best_audio_device

# ---- 配置 ----
PROJECT_DIR = Path(__file__).parent

# 临时音频文件：/dev/shm 内存盘（回退到 /tmp）
_temp_dir = Path(config.AUDIO_TEMP_DIR) if config.AUDIO_TEMP_DIR else Path("/dev/shm")
if not _temp_dir.exists() or not os.access(_temp_dir, os.W_OK):
    _temp_dir = Path("/tmp")
AUDIO_FILE = _temp_dir / ".voice_hotkey_temp.wav"

USB_MIC_DEVICE = config.AUDIO_DEVICE or get_best_audio_device(fallback="default")
DEFAULT_MODEL = "small"

# 全局状态
recording_proc: subprocess.Popen | None = None
recording_start: float = 0.0
model = None
model_lock = threading.Lock()
running = True


# ═══════════════════════════════════════════════════════════════════════
#  通知
# ═══════════════════════════════════════════════════════════════════════

def notify(title: str, body: str = "", urgency: str = "normal"):
    """发送桌面通知。"""
    try:
        subprocess.run(
            ["notify-send", "-u", urgency, title, body],
            timeout=2, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass


def beep(duration_ms: int = 100):
    """播放提示音。"""
    try:
        subprocess.run(
            ["paplay", "/usr/share/sounds/freedesktop/stereo/message.oga"],
            timeout=2, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except Exception:
        # 终极回退: 终端响铃
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
    notify("🎤 录音中...", "再次按 Ctrl+Alt+Space 停止", "low")


def record_stop() -> float | None:
    """停止录音, 返回录音时长(秒), 失败返回 None。"""
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
        notify("⚠️ 录音失败", "音频太短或为空，请重试", "critical")
        return None
    return duration


# ═══════════════════════════════════════════════════════════════════════
#  转写 + 剪贴板
# ═══════════════════════════════════════════════════════════════════════

def transcribe_audio() -> str:
    """转写录音文件, 返回文本。"""
    from faster_whisper import WhisperModel
    global model

    with model_lock:
        if model is None:
            model = WhisperModel(DEFAULT_MODEL, device="cpu", compute_type="int8")

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
    """复制文本到剪贴板。"""
    import shutil
    for cmd, args in [("xclip", ["-selection", "clipboard"]), ("wl-copy", [])]:
        if shutil.which(cmd):
            try:
                subprocess.run([cmd] + args, input=text, text=True, timeout=3)
                return True
            except Exception:
                pass
    return False


# ═══════════════════════════════════════════════════════════════════════
#  热键回调 — 异步执行转写 (避免阻塞键盘监听)
# ═══════════════════════════════════════════════════════════════════════

def on_toggle_recording():
    """热键触发: 切换录音状态。"""
    global recording_proc

    if recording_proc is not None:
        # 停止录音 → 异步转写
        duration = record_stop()
        if duration is not None and duration >= 0.5:
            threading.Thread(target=_transcribe_and_copy, args=(duration,), daemon=True).start()
    else:
        # 开始录音
        record_start()


def _transcribe_and_copy(duration: float):
    """后台线程: 转写 + 复制 + 通知。"""
    notify("📝 转写中...", f"录制了 {duration:.1f} 秒", "low")
    try:
        text = transcribe_audio()
        if text:
            copied = copy_to_clipboard(text)
            preview = text[:80] + ("..." if len(text) > 80 else "")
            status = "✅ 已复制到剪贴板" if copied else "📋 转写完成 (剪贴板不可用)"
            notify(status, preview, "normal")
        else:
            notify("📝 转写结果为空", "未检测到语音内容", "low")
    except Exception as e:
        notify("❌ 转写失败", str(e)[:100], "critical")


def on_quit():
    """热键触发: 退出守护进程。"""
    global running, recording_proc
    if recording_proc is not None:
        recording_proc.terminate()
        recording_proc.wait()
    notify("🛑 语音热键已退出", "", "low")
    running = False
    # 停止 pynput 监听器
    return False


# ═══════════════════════════════════════════════════════════════════════
#  主入口
# ═══════════════════════════════════════════════════════════════════════

def main():
    global DEFAULT_MODEL

    parser = argparse.ArgumentParser(description="全局热键语音输入守护进程")
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL,
                        choices=["tiny", "small", "medium", "large-v3"],
                        help=f"模型大小 (默认: {DEFAULT_MODEL})")
    args = parser.parse_args()

    DEFAULT_MODEL = args.model

    # 预加载模型
    print(f"⏳ 加载语音模型 ({DEFAULT_MODEL})...", end="", flush=True)
    from faster_whisper import WhisperModel
    global model
    with model_lock:
        model = WhisperModel(DEFAULT_MODEL, device="cpu", compute_type="int8")
    print(" 完成 ✅")

    print("""
╔═══════════════════════════════════════════════════╗
║       🎤 语音热键守护进程                          ║
╠═══════════════════════════════════════════════════╣
║  Ctrl+Alt+Space  → 开始/停止录音，自动转写+复制    ║
║  Ctrl+Alt+Q      → 退出守护进程                    ║
║                                                   ║
║  录音停止后结果自动进剪贴板，Ctrl+V 粘贴即可       ║
╚═══════════════════════════════════════════════════╝
""")
    notify("🎤 语音热键就绪", "Ctrl+Alt+Space 开始录音\nCtrl+Alt+Q 退出", "low")

    # ---- 注册全局热键 ----
    # 使用 pynput 的全局热键功能
    hotkeys = {
        "<ctrl>+<alt>+<space>": on_toggle_recording,
        "<ctrl>+<alt>+q": on_quit,
    }

    with keyboard.GlobalHotKeys(hotkeys) as listener:
        try:
            listener.join()
        except KeyboardInterrupt:
            pass

    # 清理
    if recording_proc is not None:
        recording_proc.terminate()
        recording_proc.wait()
    if AUDIO_FILE.exists():
        AUDIO_FILE.unlink()
    print("\n👋 语音热键已退出。")


if __name__ == "__main__":
    main()
