#!/usr/bin/env python3
"""
语音输入 GUI —— 快捷键唤出，按钮控制录音，自动加标点修正。

特性:
  - Ctrl+Alt+Space → 唤出/隐藏窗口
  - 录音按钮控制开始/结束
  - 转写后自动加标点 + 基本语义修正
  - 关闭窗口 = 隐藏（后台运行）
  - 结果自动复制到剪贴板

用法:
    python voice_gui.py
    python voice_gui.py -m medium
"""

import argparse
import os
import re
import signal
import subprocess
import sys
import threading
import time
import tkinter as tk
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
AUDIO_FILE = _temp_dir / ".voice_gui_temp.wav"

PID_FILE = Path("/tmp/voice_gui.pid")
USB_MIC_DEVICE = config.AUDIO_DEVICE or get_best_audio_device(fallback="default")
DEFAULT_MODEL = "small"

# 全局
recording_proc: subprocess.Popen | None = None
recording_start: float = 0.0
model = None
model_size = DEFAULT_MODEL


# ═══════════════════════════════════════════════════════════════════════
#  中文后处理：加标点 + 语义修正
# ═══════════════════════════════════════════════════════════════════════

# 从句连接词 —— 前面加逗号
_COMMA_BEFORE = re.compile(
    r'(但是|可是|不过|然而|所以|因此|因而|于是|然后|接着|'
    r'而且|并且|况且|另外|此外|还有|同时|同样|'
    r'否则|不然|要不|总之|毕竟|反正|其实|当然|确实|的确)'
)

# 新话题/新句子标记 —— 前面加句号
_PERIOD_BEFORE = re.compile(
    r'(我们的目标是|目标是|方案是|关键是|问题是|重点是|'
    r'具体来说|具体而言|换句话|总的来|综上所述|'
    r'第一|第二|第三|首先|其次|最后|接下来|下一步)'
)

# 分句标记 —— 前面加逗号（注意避免匹配组合词中的字）
_COMMA_AFTER_CLAUSE = re.compile(
    r'(其中包括|包括|比如|例如|譬如|也就是|即|'
    r'或者说|准确地说|也就是说|就像|'
    r'分别是|主要有|具体有|'
    r'其中|此时|这时|这种情况下|一般来说)'
)

# 问句结尾词
_QUESTION_END = re.compile(r'(吗|呢|吧|呀|啊)$')

# 常见语气词（误识别片段）
_FILLER_WORDS = re.compile(r'^(嗯|呃|啊|哦|唔|额)$')

# 连续重复字
_REPEAT_CHARS = re.compile(r'(.)\1{3,}')


def restore_punctuation(text: str, segments: list) -> str:
    """
    中文标点恢复：文本模式为主 + 时间间隔为辅。
    """
    if not text:
        return text

    # ==== 第一轮：基于文本内容插入标点 ====

    # 1. 在从句连接词前加逗号（避免重复逗号）
    def _add_comma_before(m):
        word = m.group(1)
        # 前面已是标点则跳过
        if m.start() > 0 and text[m.start()-1] in '，。！？、；：':
            return word
        return f'，{word}'
    text = _COMMA_BEFORE.sub(_add_comma_before, text)

    # 2. 在新话题/新句子标记前加句号
    text = _PERIOD_BEFORE.sub(r'。\1', text)

    # 3. 在举例/说明标记前加逗号
    text = _COMMA_AFTER_CLAUSE.sub(r'，\1', text)

    # 4. 长宾语停顿（如 "目标是仅依靠..." → "目标是，仅依靠..."）
    text = re.sub(r'(目标是|方案是|关键是|重点是)(?!，)(?=[一-鿿]{3})', r'\1，', text)

    # ==== 第二轮：基于时间间隔补充标点 ====

    if segments and len(segments) > 1:
        # 构建已处理文本的位置映射（近似）
        for i in range(len(segments) - 1):
            gap = segments[i + 1][0] - segments[i][1]
            if gap > 0.8:
                # 大间隔 → 检查此处是否已有标点，若没有则加句号
                # 用文本内容匹配定位（近似）
                seg_text = segments[i][2].strip()
                if seg_text and seg_text[-1] not in '。！？，、；：':
                    # 在 seg_text 末尾查找是否已有标点
                    try:
                        idx = text.index(seg_text)
                        end_pos = idx + len(seg_text)
                        if end_pos < len(text) and text[end_pos] not in '。！？，、；：':
                            if _QUESTION_END.search(seg_text):
                                text = text[:end_pos] + '？' + text[end_pos:]
                            elif gap > 1.0:
                                text = text[:end_pos] + '。' + text[end_pos:]
                            else:
                                text = text[:end_pos] + '，' + text[end_pos:]
                    except ValueError:
                        pass

    # ==== 第三轮：清理修正 ====

    # 确保末尾有标点
    if text and text[-1] not in '。！？，、；：':
        if _QUESTION_END.search(text[-2:] if len(text) > 1 else text):
            text += '？'
        else:
            text += '。'

    # 去掉开头多余的标点
    text = re.sub(r'^[，。！？、；：]+', '', text)

    # 修正重复标点
    text = re.sub(r'([，。！？、；：])\1+', r'\1', text)

    # 修正标点 + 标点的组合：保留后者
    text = re.sub(r'[，][。]', '。', text)
    text = re.sub(r'[。][，]', '。', text)
    text = re.sub(r'[，、]{2,}', '，', text)

    # 去掉"和"前面的顿号（"A、B、和C" → "A、B和C"）
    text = re.sub(r'、和', '和', text)

    return text


def semantic_cleanup(text: str) -> str:
    """基本语义修正：去口头禅、修正重复、常见同音错词。"""
    if not text:
        return text

    # 1. 删除口头禅
    text = re.sub(r'那个那个+', '那个', text)
    text = re.sub(r'就是说就是说+', '就是说', text)
    text = re.sub(r'然后然后然后+', '然后', text)

    # 2. 修正连续重复字（保留 2 个以内）
    text = _REPEAT_CHARS.sub(lambda m: m.group(1) * 2, text)

    # 3. 常见同音错词修正（中文语音识别常见错误）
    _HOMOPHONE_FIXES = {
        "在来": "再来", "在见": "再见", "在会": "再会",
        "象是": "像是", "好象": "好像",
        "以经": "已经", "以钱": "以前",
        "因该": "应该", "因当": "应当",
        "知到": "知道", "只到": "知道",
        "一各": "一个", "这各": "这个", "那各": "那个",
        "可已": "可以",
        "的却": "的确",
        "不关": "不管",  # 需要看上下文，保守处理
        "既然": "既然",
        "在说": "再说",
        "再那里": "在那里", "在见": "再见",
        "有什么事": "有什么",  # 可能多余
    }
    for wrong, right in _HOMOPHONE_FIXES.items():
        # 只替换独立出现的（前后有标点或空格或开头结尾）
        text = re.sub(rf'(^|[^a-zA-Z0-9一-鿿]){re.escape(wrong)}([^a-zA-Z0-9一-鿿]|$)',
                      rf'\1{right}\2', text)

    # 4. 去除多余空白
    text = re.sub(r'\s+', '', text)

    return text


def post_process(text: str, segments: list) -> str:
    """完整的后处理流水线：标点恢复 + 语义修正。"""
    text = restore_punctuation(text, segments)
    text = semantic_cleanup(text)
    return text


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


def record_stop():
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
    if not AUDIO_FILE.exists() or AUDIO_FILE.stat().st_size < 1000:
        return None
    return duration


# ═══════════════════════════════════════════════════════════════════════
#  转写（返回文本 + 片段信息用于标点恢复）
# ═══════════════════════════════════════════════════════════════════════

def transcribe_audio() -> tuple[str, list]:
    """转写，返回 (文本, 片段列表)。"""
    global model
    segments, info = model.transcribe(
        str(AUDIO_FILE), beam_size=5, language=None,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )
    seg_list = []
    text_parts = []
    for seg in segments:
        text_parts.append(seg.text.strip())
        seg_list.append((seg.start, seg.end, seg.text.strip()))

    text = "".join(text_parts).strip()
    if AUDIO_FILE.exists():
        AUDIO_FILE.unlink()
    return text, seg_list


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


# ═══════════════════════════════════════════════════════════════════════
#  GUI
# ═══════════════════════════════════════════════════════════════════════

class VoiceGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("🎤 语音输入")
        self.root.geometry("520x440")
        self.root.resizable(True, True)
        self.root.configure(bg="#1e1e1e")
        self.root.attributes("-topmost", True)

        # 状态
        self.is_recording = False
        self.last_text = ""
        self.last_raw_text = ""  # 原始转写（修正前）
        self.timer_job = None

        self._build_ui()
        self._update_status("idle")

        # 绑定 Esc 键 → 隐藏窗口
        self.root.bind("<Escape>", lambda e: self._hide_window())
        # 窗口关闭 → 隐藏而不是销毁
        self.root.protocol("WM_DELETE_WINDOW", self._hide_window)

    def _build_ui(self):
        # ---- 顶部标题 ----
        title = tk.Label(
            self.root, text="🎤 语音输入",
            font=("Sans", 18, "bold"),
            bg="#1e1e1e", fg="#e0e0e0",
        )
        title.pack(pady=(15, 5))

        hint = tk.Label(
            self.root, text="Ctrl+Alt+Space 唤出/隐藏 · Esc 关闭窗口",
            font=("Sans", 9),
            bg="#1e1e1e", fg="#666666",
        )
        hint.pack()

        # ---- 状态指示器 ----
        status_frame = tk.Frame(self.root, bg="#1e1e1e")
        status_frame.pack(pady=5)

        self.status_indicator = tk.Canvas(
            status_frame, width=30, height=30,
            bg="#1e1e1e", highlightthickness=0,
        )
        self.status_indicator.pack(side="left", padx=(0, 8))
        self._indicator_circle = self.status_indicator.create_oval(
            4, 4, 26, 26, fill="#555555", outline=""
        )

        self.status_label = tk.Label(
            status_frame, text="⏸ 就绪",
            font=("Sans", 16, "bold"),
            bg="#1e1e1e", fg="#aaaaaa",
        )
        self.status_label.pack(side="left")

        # 计时器
        self.timer_label = tk.Label(
            self.root, text="",
            font=("Sans", 13),
            bg="#1e1e1e", fg="#888888",
        )
        self.timer_label.pack(pady=2)

        # ---- 主按钮 ----
        self.action_btn = tk.Button(
            self.root,
            text="🎤 开始录音",
            font=("Sans", 18, "bold"),
            bg="#4CAF50", fg="white",
            activebackground="#45a049", activeforeground="white",
            relief="flat", bd=0, padx=30, pady=12,
            cursor="hand2",
            command=self._toggle_recording,
        )
        self.action_btn.pack(pady=15)

        # ---- 分隔线 ----
        sep = tk.Frame(self.root, height=1, bg="#444444")
        sep.pack(fill="x", padx=30, pady=5)

        # ---- 转写结果 ----
        result_header = tk.Label(
            self.root, text="📋 转写结果（已自动加标点、修正）",
            font=("Sans", 11, "bold"),
            bg="#1e1e1e", fg="#999999",
        )
        result_header.pack(anchor="w", padx=30, pady=(8, 2))

        result_frame = tk.Frame(self.root, bg="#2a2a2a",
                                highlightbackground="#444444", highlightthickness=1)
        result_frame.pack(fill="both", expand=True, padx=20, pady=(0, 8))

        self.result_text = tk.Text(
            result_frame,
            font=("Sans", 13),
            bg="#2a2a2a", fg="#e0e0e0",
            wrap="word", relief="flat", bd=0,
            padx=12, pady=10,
            height=5,
        )
        self.result_text.pack(fill="both", expand=True)
        self.result_text.insert("1.0", "等待录音...\n")
        self.result_text.config(state="disabled")

        # 原始文本对比（可折叠显示）
        self.raw_var = tk.BooleanVar(value=False)
        self.raw_check = tk.Checkbutton(
            self.root, text="显示原始转写（修正前）",
            variable=self.raw_var,
            command=self._toggle_raw_view,
            font=("Sans", 9),
            bg="#1e1e1e", fg="#777777",
            selectcolor="#1e1e1e",
            activebackground="#1e1e1e", activeforeground="#999999",
        )
        self.raw_check.pack(anchor="w", padx=30, pady=(0, 2))

        # ---- 底部按钮 ----
        bottom_frame = tk.Frame(self.root, bg="#1e1e1e")
        bottom_frame.pack(fill="x", padx=20, pady=(0, 12))

        self.copy_btn = tk.Button(
            bottom_frame,
            text="📋 复制到剪贴板",
            font=("Sans", 11),
            bg="#333333", fg="#e0e0e0",
            activebackground="#555555", activeforeground="white",
            relief="flat", bd=0, padx=12, pady=5,
            cursor="hand2",
            command=self._copy_result,
            state="disabled",
        )
        self.copy_btn.pack(side="left")

        self.clear_btn = tk.Button(
            bottom_frame,
            text="🗑 清除",
            font=("Sans", 11),
            bg="#333333", fg="#e0e0e0",
            activebackground="#555555", activeforeground="white",
            relief="flat", bd=0, padx=12, pady=5,
            cursor="hand2",
            command=self._clear_result,
        )
        self.clear_btn.pack(side="right")

    # ---- 窗口显示/隐藏 ----

    def show(self):
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()

    def _hide_window(self):
        self.root.withdraw()
        # 隐藏后短暂显示一个浮动提示
        self._show_toast("语音输入已隐藏\nCtrl+Alt+Space 再次唤出")

    def _show_toast(self, msg: str):
        """短暂弹出提示窗口。"""
        toast = tk.Toplevel(self.root)
        toast.title("")
        toast.geometry("280x60")
        toast.configure(bg="#333333")
        toast.overrideredirect(True)  # 无标题栏
        toast.attributes("-topmost", True)
        # 居中
        toast.update_idletasks()
        sw = toast.winfo_screenwidth()
        sh = toast.winfo_screenheight()
        x = (sw - 280) // 2
        y = sh - 120
        toast.geometry(f"+{x}+{y}")
        label = tk.Label(toast, text=msg, font=("Sans", 11),
                         bg="#333333", fg="#e0e0e0", justify="center")
        label.pack(expand=True)
        toast.after(2500, toast.destroy)

    def toggle_visibility(self):
        if self.root.state() == "withdrawn":
            self.show()
        else:
            self._hide_window()

    # ---- 状态更新 ----

    def _update_status(self, state: str):
        colors = {
            "idle": ("#555555", "#aaaaaa", "⏸ 就绪"),
            "recording": ("#e53935", "#e53935", "🔴 录音中"),
            "transcribing": ("#fb8c00", "#fb8c00", "📝 转写中..."),
            "done": ("#43a047", "#43a047", "✅ 转写完成"),
        }
        if state not in colors:
            return
        fill, fg, text = colors[state]
        self.status_indicator.itemconfig(self._indicator_circle, fill=fill)
        self.status_label.config(text=text, fg=fg)

    def _update_timer(self):
        if self.is_recording and recording_proc is not None:
            elapsed = time.time() - recording_start
            mins, secs = int(elapsed) // 60, int(elapsed) % 60
            self.timer_label.config(text=f"[{mins:02d}:{secs:02d}]")
            self.timer_job = self.root.after(200, self._update_timer)
        else:
            self.timer_label.config(text="")
            self.timer_job = None

    def _set_result(self, text: str):
        self.result_text.config(state="normal")
        self.result_text.delete("1.0", "end")
        self.result_text.insert("1.0", text)
        self.result_text.config(state="normal")
        self.last_text = text
        self.copy_btn.config(state="normal" if text else "disabled")

    def _toggle_raw_view(self):
        if self.raw_var.get():
            self._set_result(self.last_raw_text or self.last_text)
        else:
            self._set_result(self.last_text)

    def _clear_result(self):
        self.result_text.config(state="normal")
        self.result_text.delete("1.0", "end")
        self.result_text.insert("1.0", "等待录音...\n")
        self.result_text.config(state="normal")
        self.last_text = ""
        self.last_raw_text = ""
        self.raw_var.set(False)
        self.copy_btn.config(state="disabled")
        self._update_status("idle")

    def _copy_result(self):
        if self.last_text:
            ok = copy_to_clipboard(self.last_text)
            self.copy_btn.config(
                text="✅ 已复制!" if ok else "❌ 复制失败",
                bg="#43a047" if ok else "#e53935",
            )
            self.root.after(2000, lambda: self.copy_btn.config(
                text="📋 复制到剪贴板", bg="#333333",
            ))

    # ---- 核心操作 ----

    def _toggle_recording(self):
        if self.is_recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self):
        if self.is_recording:
            return
        self.is_recording = True
        record_start()

        self._update_status("recording")
        self._update_timer()
        self.action_btn.config(
            text="⏹ 停止录音", bg="#e53935",
            activebackground="#c62828",
        )
        self.result_text.config(state="normal")
        self.result_text.delete("1.0", "end")
        self.result_text.insert("1.0", "录音中...\n")
        self.result_text.config(state="normal")

    def _stop_recording(self):
        if not self.is_recording:
            return
        self.is_recording = False

        duration = record_stop()
        self._update_status("transcribing")
        self.action_btn.config(
            text="📝 转写中...", bg="#ff9800",
            activebackground="#f57c00", state="disabled",
        )
        dur_text = f"录制了 {int(duration or 0)}s"
        self.timer_label.config(text=dur_text)

        if duration is None or duration < 0.5:
            self._update_status("idle")
            self.action_btn.config(
                text="🎤 开始录音", bg="#4CAF50",
                activebackground="#45a049", state="normal",
            )
            self.timer_label.config(text="⚠️ 录音太短，请重试")
            self._set_result("录音太短，请说至少 0.5 秒")
            return

        def _run():
            try:
                raw_text, segments = transcribe_audio()
                # 后处理：加标点 + 语义修正
                cleaned = post_process(raw_text, segments)
                self.root.after(0, lambda: self._on_done(cleaned, raw_text))
            except Exception as e:
                self.root.after(0, lambda: self._on_error(str(e)))

        threading.Thread(target=_run, daemon=True).start()

    def _on_done(self, cleaned: str, raw: str):
        self._update_status("done")
        self.action_btn.config(
            text="🎤 开始录音", bg="#4CAF50",
            activebackground="#45a049", state="normal",
        )
        self.timer_label.config(text="")
        self.last_raw_text = raw
        self.raw_var.set(False)

        if cleaned:
            self._set_result(cleaned)
            copy_to_clipboard(cleaned)
            self.copy_btn.config(text="✅ 已复制到剪贴板", bg="#43a047")
            self.root.after(2000, lambda: self.copy_btn.config(
                text="📋 复制到剪贴板", bg="#333333",
            ))
        else:
            self._set_result("未检测到语音内容")

    def _on_error(self, error: str):
        self._update_status("idle")
        self.action_btn.config(
            text="🎤 开始录音", bg="#4CAF50",
            activebackground="#45a049", state="normal",
        )
        self._set_result(f"转写失败: {error}")


# ═══════════════════════════════════════════════════════════════════════
#  主入口
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="语音输入 GUI")
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL,
                        choices=["tiny", "small", "medium", "large-v3"],
                        help=f"模型大小 (默认: {DEFAULT_MODEL})")
    args = parser.parse_args()

    global model, model_size
    model_size = args.model

    # 写 PID
    PID_FILE.write_text(str(os.getpid()))

    # 预加载模型
    print(f"⏳ 加载语音模型 ({model_size})...", end="", flush=True)
    from faster_whisper import WhisperModel
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    print(" 完成 ✅")

    root = tk.Tk()
    app = VoiceGUI(root)

    # ---- 信号处理：SIGUSR1 → 切换窗口显示/隐藏 ----
    def on_signal(signum, frame):
        root.after(0, app.toggle_visibility)

    signal.signal(signal.SIGUSR1, on_signal)

    # ---- 窗口关闭 = 隐藏（不退出进程）- ---
    root.protocol("WM_DELETE_WINDOW", lambda: app._hide_window())

    # ---- 真正退出：SIGTERM 或 SIGINT ----
    def cleanup_and_exit(*args):
        global recording_proc
        if recording_proc is not None:
            recording_proc.terminate()
            recording_proc.wait()
        if AUDIO_FILE.exists():
            AUDIO_FILE.unlink()
        if PID_FILE.exists():
            PID_FILE.unlink()
        root.destroy()

    signal.signal(signal.SIGTERM, cleanup_and_exit)
    signal.signal(signal.SIGINT, cleanup_and_exit)

    # ---- 启动时隐藏窗口，快捷键唤出 ----
    print("启动 GUI (隐藏中，Ctrl+Alt+Space 唤出)...")
    root.withdraw()  # 初始隐藏

    root.mainloop()


if __name__ == "__main__":
    main()
