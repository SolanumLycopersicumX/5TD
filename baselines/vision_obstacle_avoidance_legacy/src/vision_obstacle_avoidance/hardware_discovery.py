"""
Linux 硬件自动发现模块。
枚举摄像头 (V4L2) 和音频输入设备 (PipeWire / PulseAudio / ALSA)，
所有方法设有 fallback，检测失败时返回 None 由 config.py 兜底。

在非 Linux 平台上所有函数直接返回 None。
"""

import logging
import platform
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

# ── 缓存：避免重复调用子进程 ──────────────────────────────────────────
_discovery_cache: dict = {}


def _is_linux() -> bool:
    return platform.system() == "Linux"


# ═══════════════════════════════════════════════════════════════════════════
#  摄像头发现
# ═══════════════════════════════════════════════════════════════════════════

def discover_cameras() -> list[dict] | None:
    """
    列举所有视频设备。

    返回: [{'index': 0, 'path': '/dev/video0', 'name': 'USB Camera'}, ...]
          检测失败返回 None
    """
    if not _is_linux():
        return None

    cache_key = "cameras"
    if cache_key in _discovery_cache:
        return _discovery_cache[cache_key]

    cameras = _discover_via_v4l2ctl()
    if cameras is None:
        cameras = _discover_via_sysfs()

    _discovery_cache[cache_key] = cameras
    return cameras


def _discover_via_v4l2ctl() -> list[dict] | None:
    """通过 v4l2-ctl 列举设备。"""
    if not shutil.which("v4l2-ctl"):
        return None

    try:
        result = subprocess.run(
            ["v4l2-ctl", "--list-devices"],
            capture_output=True, text=True, timeout=5,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None

    if result.returncode != 0:
        return None

    cameras = []
    current_name = None

    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            current_name = None
            continue

        # 设备名行（不以 /dev/ 开头）
        if not line.startswith("/dev/"):
            # 去掉末尾的冒号
            current_name = line.rstrip(":")
        elif current_name is not None:
            # /dev/videoN 行
            dev_path = line.strip()
            try:
                index = int(dev_path.replace("/dev/video", ""))
            except ValueError:
                continue
            cameras.append({
                "index": index,
                "path": dev_path,
                "name": current_name,
            })

    return cameras if cameras else None


def _discover_via_sysfs() -> list[dict] | None:
    """通过 /sys/class/video4linux 列举设备（fallback）。"""
    sysfs_dir = Path("/sys/class/video4linux")
    if not sysfs_dir.exists():
        return None

    cameras = []
    for entry in sorted(sysfs_dir.iterdir()):
        if not entry.name.startswith("video"):
            continue
        try:
            index = int(entry.name.replace("video", ""))
        except ValueError:
            continue

        # 读取设备名称
        name_file = entry / "name"
        if name_file.exists():
            try:
                name = name_file.read_text().strip()
            except OSError:
                name = entry.name
        else:
            name = entry.name

        dev_path = f"/dev/{entry.name}"
        cameras.append({
            "index": index,
            "path": dev_path,
            "name": name,
        })

    return cameras if cameras else None


# ═══════════════════════════════════════════════════════════════════════════
#  音频设备发现
# ═══════════════════════════════════════════════════════════════════════════

def resolve_audio_device() -> str | None:
    """
    自动查找最佳录音设备，返回 ALSA 设备字符串（如 "plughw:1,0"）。

    优先级: PipeWire → PulseAudio → ALSA /proc/asound
    返回 None 表示没有找到可用设备。
    """
    if not _is_linux():
        return None

    cache_key = "audio_device"
    if cache_key in _discovery_cache:
        return _discovery_cache[cache_key]

    device = None

    # 1. PipeWire (Ubuntu 24.04 默认)
    device = _discover_via_pipewire()
    if device:
        logger.info(f"Audio device (PipeWire): {device}")
        _discovery_cache[cache_key] = device
        return device

    # 2. PulseAudio
    device = _discover_via_pulseaudio()
    if device:
        logger.info(f"Audio device (PulseAudio): {device}")
        _discovery_cache[cache_key] = device
        return device

    # 3. ALSA /proc/asound
    device = _discover_via_alsa()
    if device:
        logger.info(f"Audio device (ALSA): {device}")
        _discovery_cache[cache_key] = device
        return device

    logger.warning("No audio input device found")
    _discovery_cache[cache_key] = None
    return None


def _discover_via_pipewire() -> str | None:
    """通过 pw-cli 查找录音设备。"""
    if not shutil.which("pw-cli"):
        return None

    try:
        result = subprocess.run(
            ["pw-cli", "list-objects", "Node"],
            capture_output=True, text=True, timeout=5,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None

    if result.returncode != 0:
        return None

    # 解析 pw-cli 输出，找输入音频节点
    # 格式: id N, type PipeWire:Interface:Node
    #       ...
    #       node.name = "alsa_input.usb-..."
    best_node = None
    current_node = None
    current_is_source = False

    for line in result.stdout.splitlines():
        line = line.strip()

        if line.startswith("id ") and "Node" in line:
            # 检查上一节点
            if current_node and current_is_source:
                best_node = current_node
                break  # 找到第一个即可
            current_node = None
            current_is_source = False
            continue

        if "node.name" in line:
            name = line.split("=")[-1].strip().strip('"')
            if current_node is None:
                current_node = name
            # 判断是否为输入设备
            if "input" in name.lower() or "mic" in name.lower() or "source" in name.lower():
                current_is_source = True
                current_node = name

    if best_node:
        # 用 pw-cli 的 node.name 反向查 ALSA 设备号
        # 简化: 直接查 /proc/asound 匹配
        return _find_alsa_device_by_name(best_node)

    return None


def _discover_via_pulseaudio() -> str | None:
    """通过 pactl 查找录音设备。"""
    if not shutil.which("pactl"):
        return None

    try:
        result = subprocess.run(
            ["pactl", "list", "sources", "short"],
            capture_output=True, text=True, timeout=5,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None

    if result.returncode != 0:
        return None

    # 找到第一个非 monitor 的 source
    for line in result.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        name = parts[1].strip()
        # 跳过 monitor（输出回采）
        if ".monitor" in name:
            continue
        # 如果名字包含 alsa_input，尝试提取
        if "alsa_input" in name:
            alsa_dev = _find_alsa_device_by_name(name)
            if alsa_dev:
                return alsa_dev
        # 返回 PulseAudio 设备名本身（arecord 也支持）
        return name

    return None


def _find_alsa_device_by_name(name_hint: str) -> str | None:
    """
    根据名称提示（来自 PipeWire/PulseAudio）查找对应 ALSA 设备号。
    返回 "plughw:CARD,DEV" 或 None。

    匹配策略：如果 name_hint 包含 "usb"，优先匹配 USB 声卡；
    否则返回第一个可用录音设备。
    """
    proc_path = Path("/proc/asound")
    if not proc_path.exists():
        return None

    prefer_usb = "usb" in name_hint.lower()
    fallback_device: str | None = None
    usb_device: str | None = None

    # 遍历声卡
    for card_dir in sorted(proc_path.iterdir()):
        if not card_dir.name.startswith("card"):
            continue
        try:
            card_num = int(card_dir.name.replace("card", ""))
        except ValueError:
            continue

        # 检查该声卡是否包含录音 PCM
        has_capture = False
        best_dev_num = None
        for pcm_dev in sorted(card_dir.iterdir()):
            if not pcm_dev.name.startswith("pcm"):
                continue
            dev_name = pcm_dev.name  # pcm0c, pcm0p, etc.
            # 'c' suffix = capture (录音)
            if not dev_name.endswith("c"):
                continue
            try:
                dev_num = int(dev_name[3:-1])  # pcm0c → 0
            except ValueError:
                continue
            has_capture = True
            if best_dev_num is None or dev_num < best_dev_num:
                best_dev_num = dev_num

        if not has_capture or best_dev_num is None:
            continue

        device_str = f"plughw:{card_num},{best_dev_num}"

        # USB 声卡特征: /proc/asound/cardX/usbid 文件存在
        is_card_usb = (card_dir / "usbid").exists()

        if is_card_usb:
            usb_device = usb_device or device_str
        elif fallback_device is None:
            fallback_device = device_str

    # 返回策略: USB 优先 > name_hint 含 usb 时强制 USB > 首个可用
    if prefer_usb and usb_device:
        logger.info(f"Matched USB audio device: {usb_device} (hint: {name_hint})")
        return usb_device
    if usb_device:
        logger.info(f"Preferring USB audio device: {usb_device}")
        return usb_device

    return fallback_device


def _discover_via_alsa() -> str | None:
    """直接通过 /proc/asound 查找录音设备（最终 fallback）。"""
    return _find_alsa_device_by_name("")


# ═══════════════════════════════════════════════════════════════════════════
#  便捷函数
# ═══════════════════════════════════════════════════════════════════════════

def get_best_audio_device(fallback: str = "default") -> str:
    """
    获取最佳录音设备，自动检测失败时返回 fallback。

    用法:
        device = get_best_audio_device()
        subprocess.run(["arecord", "-D", device, ...])
    """
    detected = resolve_audio_device()
    if detected:
        return detected
    logger.info(f"Using fallback audio device: {fallback}")
    return fallback


def get_best_camera_index(fallback: int = 0) -> int:
    """
    获取最佳摄像头索引，自动检测失败时返回 fallback。

    用法:
        index = get_best_camera_index()
        cap = cv2.VideoCapture(index)
    """
    cameras = discover_cameras()
    if cameras:
        logger.info(f"Found {len(cameras)} camera(s), using {cameras[0]['path']}")
        return cameras[0]["index"]
    logger.info(f"Using fallback camera index: {fallback}")
    return fallback
