#!/usr/bin/env python3
"""
语音录制 + 转文字工具。
使用 faster-whisper (small 模型, CPU 友好), 支持中英文混合。
默认使用 USB 麦克风 (Lenovo Services E03)。

交互式控制:
    空格  → 开始/停止录音（停止后自动转写）
    r    → 放弃当前结果, 重新录制
    q    → 退出

也支持命令行模式:
    python transcribe.py -f audio.wav   # 转录已有音频文件
    python transcribe.py -m medium      # 使用中模型 (更准但慢)
"""

import argparse
import atexit
import os
import select
import subprocess
import sys
import termios
import time
from pathlib import Path

# 国内镜像加速 HuggingFace 模型下载
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

AUDIO_FILE = Path(__file__).parent / ".dictation_temp.wav"
USB_MIC_DEVICE = "plughw:1,0"  # plug 插件自动转换采样率/声道
DEFAULT_MODEL = "small"


# ═══════════════════════════════════════════════════════════════════════
#  终端原始模式（非阻塞按键检测）
# ═══════════════════════════════════════════════════════════════════════

def enable_raw_mode():
    """启用终端原始模式, 返回原始设置。"""
    fd = sys.stdin.fileno()
    if not os.isatty(fd):
        return None
    old = termios.tcgetattr(fd)
    new = termios.tcgetattr(fd)
    new[3] = new[3] & ~(termios.ECHO | termios.ICANON)  # 关闭回显 & 规范模式
    new[6][termios.VMIN] = 0
    new[6][termios.VTIME] = 0
    termios.tcsetattr(fd, termios.TCSANOW, new)
    return old


def disable_raw_mode(old):
    """恢复终端设置。"""
    if old is None:
        return
    try:
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old)
    except Exception:
        pass


def read_key(timeout: float = 0.1) -> str | None:
    """非阻塞读取单字符按键, 无输入时返回 None。"""
    if select.select([sys.stdin], [], [], timeout)[0]:
        return sys.stdin.read(1)
    return None


# ═══════════════════════════════════════════════════════════════════════
#  录音
# ═══════════════════════════════════════════════════════════════════════

def start_recording(output_path: str, device: str = USB_MIC_DEVICE) -> subprocess.Popen:
    """启动录音子进程, 返回 Popen 对象。"""
    proc = subprocess.Popen(
        [
            "arecord", "-D", device,
            "-f", "S16_LE",
            "-r", "16000",
            "-c", "1",
            "-t", "wav",
            output_path,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return proc


def stop_recording(proc: subprocess.Popen) -> bool:
    """停止录音子进程, 返回是否成功保存了有效音频文件。"""
    proc.terminate()
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    return Path(AUDIO_FILE).exists() and Path(AUDIO_FILE).stat().st_size > 1000


# ═══════════════════════════════════════════════════════════════════════
#  转写
# ═══════════════════════════════════════════════════════════════════════

def load_model(model_size: str = DEFAULT_MODEL):
    """预加载 Whisper 模型（耗时操作, 启动时执行一次）。"""
    from faster_whisper import WhisperModel
    return WhisperModel(model_size, device="cpu", compute_type="int8")


def transcribe(audio_path: str, model, show_progress: bool = True) -> str:
    """使用已加载的模型转写音频。"""
    if show_progress:
        sys.stdout.write("\n  📝 转写中...")
        sys.stdout.flush()

    segments, info = model.transcribe(
        audio_path,
        beam_size=5,
        language=None,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )

    text_parts = []
    for seg in segments:
        text_parts.append(seg.text.strip())
        if show_progress:
            sys.stdout.write(f"\r  [{seg.start:.1f}s → {seg.end:.1f}s] {seg.text.strip()}\n")
            sys.stdout.flush()

    text = "".join(text_parts).strip()
    return text


# ═══════════════════════════════════════════════════════════════════════
#  剪贴板
# ═══════════════════════════════════════════════════════════════════════

def copy_to_clipboard(text: str) -> bool:
    """尝试复制文本到剪贴板。"""
    import shutil
    for cmd in ["xclip", "wl-copy"]:
        if shutil.which(cmd):
            try:
                subprocess.run(
                    [cmd, "-selection", "clipboard"] if cmd == "xclip" else [cmd],
                    input=text, text=True, timeout=3,
                )
                return True
            except Exception:
                pass
    return False


# ═══════════════════════════════════════════════════════════════════════
#  交互式界面
# ═══════════════════════════════════════════════════════════════════════

def draw_header():
    """打印界面头部。"""
    os.system("clear" if os.name != "nt" else "cls")
    print("╔══════════════════════════════════════════════╗")
    print("║       🎤 语音转文字 (Ctrl+C 退出)           ║")
    print("╠══════════════════════════════════════════════╣")
    print("║  [空格] 开始/停止录音  停止后自动转写       ║")
    print("║  [r]    放弃结果, 重新录制                  ║")
    print("║  [q]    退出                                ║")
    print("╚══════════════════════════════════════════════╝")
    print()


def draw_status(is_recording: bool, elapsed: float, last_result: str | None):
    """刷新状态区域。"""
    # 移动光标到状态区开头（重新打印整个状态区）
    # 先清到行尾, 再打印状态
    if is_recording:
        mins = int(elapsed) // 60
        secs = int(elapsed) % 60
        sys.stdout.write(f"\r  🔴 录音中... [{mins:02d}:{secs:02d}]  按 [空格] 停止        \n")
    elif last_result is not None:
        sys.stdout.write(f"\r  ✅ 就绪 (已录制 {elapsed:.0f}s)  按 [空格] 重录  [q] 退出\n")
        sys.stdout.write("  ─────────────────────────────────────────\n")
        sys.stdout.write(f"  📋 转写结果:\n")
        # 限制显示行数, 避免刷屏
        lines = last_result.split("\n")
        for line in lines[:6]:
            if len(line) > 70:
                line = line[:67] + "..."
            sys.stdout.write(f"     {line}\n")
        sys.stdout.write("  ─────────────────────────────────────────\n")
    else:
        sys.stdout.write(f"\r  ⏸  就绪  按 [空格] 开始录音                          \n")

    sys.stdout.flush()


def clear_status_lines(n: int = 12):
    """清除状态区行。"""
    for _ in range(n):
        sys.stdout.write("\033[K")  # 清除当前行
        sys.stdout.write("\033[F")  # 上移一行
    sys.stdout.write("\033[K")  # 清除最后一行
    sys.stdout.flush()


def interactive_loop(model):
    """交互式主循环：非阻塞按键 + 录音控制 + 实时状态。"""
    draw_header()

    recording: subprocess.Popen | None = None
    recording_start: float = 0.0
    last_result: str | None = None
    last_duration: float = 0.0
    running = True

    # 初始状态
    draw_status(is_recording=False, elapsed=0.0, last_result=None)

    while running:
        key = read_key(timeout=0.1)

        if key is not None:
            # ---- 按键处理 ----
            if key == " ":
                # 空格: 切换录音状态
                if recording is not None:
                    # 停止录音
                    duration = time.time() - recording_start
                    sys.stdout.write(f"\n  ⏳ 正在停止录音 (录制了 {duration:.1f}s)...")
                    sys.stdout.flush()
                    success = stop_recording(recording)
                    recording = None
                    if not success or duration < 0.5:
                        sys.stdout.write("\r   ⚠️  录音太短或失败, 请重试                        \n")
                        sys.stdout.flush()
                        time.sleep(1)
                        draw_header()
                        draw_status(False, 0.0, last_result)
                        continue
                    # 自动转写
                    text = transcribe(str(AUDIO_FILE), model)
                    last_result = text
                    last_duration = duration
                    if AUDIO_FILE.exists():
                        AUDIO_FILE.unlink()
                    # 复制到剪贴板
                    if text:
                        copied = copy_to_clipboard(text)
                        if copied:
                            sys.stdout.write("\r  (已复制到剪贴板 ✅)                              \n")
                            sys.stdout.flush()
                    draw_header()
                    draw_status(False, last_duration, last_result)
                else:
                    # 开始录音
                    if AUDIO_FILE.exists():
                        AUDIO_FILE.unlink()
                    sys.stdout.write("\r   🎤 即将开始...                                    \n")
                    sys.stdout.flush()
                    time.sleep(0.3)
                    draw_header()
                    recording = start_recording(str(AUDIO_FILE))
                    recording_start = time.time()
                    last_result = None
                    draw_status(True, 0.0, None)

            elif key.lower() == "r":
                # r: 放弃结果, 重新录制
                if recording is not None:
                    stop_recording(recording)
                    recording = None
                if AUDIO_FILE.exists():
                    AUDIO_FILE.unlink()
                last_result = None
                last_duration = 0.0
                draw_header()
                sys.stdout.write("   🗑️  已清除, 按 [空格] 重新录制\n")
                sys.stdout.flush()
                time.sleep(0.6)
                draw_header()
                draw_status(False, 0.0, None)

            elif key.lower() == "q":
                # q: 退出
                if recording is not None:
                    sys.stdout.write("\r   正在停止录音...")
                    sys.stdout.flush()
                    stop_recording(recording)
                    recording = None
                running = False

            elif key == "\x03":
                # Ctrl+C
                if recording is not None:
                    stop_recording(recording)
                running = False

        else:
            # ---- 无按键: 实时更新录音计时器 ----
            if recording is not None:
                elapsed = time.time() - recording_start
                mins = int(elapsed) // 60
                secs = int(elapsed) % 60
                sys.stdout.write(f"\r  🔴 录音中... [{mins:02d}:{secs:02d}]  按 [空格] 停止    \r")
                sys.stdout.flush()

    return last_result


# ═══════════════════════════════════════════════════════════════════════
#  主入口
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="语音转文字工具 (faster-whisper)\n\n"
                    "交互模式: 空格切换录音, 停止后自动转写。"
    )
    parser.add_argument("-f", "--file", help="转录已有音频文件, 跳过交互模式")
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL,
                        choices=["tiny", "small", "medium", "large-v3"],
                        help=f"模型大小 (默认: {DEFAULT_MODEL})")
    args = parser.parse_args()

    # ---- 文件模式: 直接转写已有音频 ----
    if args.file:
        if not Path(args.file).exists():
            print(f"❌ 文件不存在: {args.file}")
            sys.exit(1)
        print(f"⏳ 加载模型 ({args.model})...", end="", flush=True)
        model = load_model(args.model)
        print(" 完成")
        text = transcribe(args.file, model, show_progress=True)
        print("\n" + "=" * 50)
        print("📋 转写结果:")
        print("=" * 50)
        print(text)
        print("=" * 50)
        if text:
            copy_to_clipboard(text)
        return

    # ---- 交互模式 ----
    # 预加载模型（避免首次转写等待）
    print(f"⏳ 加载模型 ({args.model})...", end="", flush=True)
    try:
        model = load_model(args.model)
        print(" 完成 ✅")
    except Exception as e:
        print(f"\n❌ 模型加载失败: {e}")
        sys.exit(1)

    # 设置终端原始模式
    old_term = enable_raw_mode()
    if old_term is None:
        print("❌ 当前终端不支持交互模式")
        sys.exit(1)
    atexit.register(lambda: disable_raw_mode(old_term))

    print()  # 空一行

    try:
        final_text = interactive_loop(model)
    except Exception as e:
        disable_raw_mode(old_term)
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        disable_raw_mode(old_term)
        # 清理
        if AUDIO_FILE.exists():
            AUDIO_FILE.unlink()

    print("\n👋 再见!")
    if final_text:
        print(f"\n最后转录结果:\n  {final_text}")


if __name__ == "__main__":
    main()
