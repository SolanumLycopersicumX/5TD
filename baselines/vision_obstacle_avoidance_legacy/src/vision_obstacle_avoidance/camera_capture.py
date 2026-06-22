"""
摄像头采集模块。
使用 OpenCV VideoCapture，支持工业相机断线自动重连（指数退避）。
预留工业相机 SDK 替换接口。
"""

import platform
import time

import cv2
import numpy as np
import config


class CameraCapture:
    """
    摄像头采集类。

    断线重连:
      读取失败时自动尝试重连，使用指数退避避免死循环和 CPU 空转。
      超过最大重试次数后放弃，返回 (None, None) 由决策层触发 STOP。

    替换为工业相机 SDK 的方法：
      1. 改写 __init__ 中的初始化逻辑（如海康 SDK 的 MV_CC_CreateHandle）
      2. 改写 read() 中的取流逻辑
      3. 保持 read() 返回值格式不变： (frame, timestamp) 或 (None, None)
    """

    def __init__(self, camera_index=None, width=None, height=None):
        self.camera_index = camera_index if camera_index is not None else config.CAMERA_INDEX
        self.width = width or config.FRAME_WIDTH
        self.height = height or config.FRAME_HEIGHT

        self.cap = None
        self._is_video_file = isinstance(self.camera_index, str)
        self._backend = cv2.CAP_V4L2 if (platform.system() == "Linux"
                                          and not self._is_video_file) else cv2.CAP_ANY

        # ── 断线重连状态 ──
        self._reconnect_enabled = (config.CAMERA_RECONNECT_ENABLED
                                   and platform.system() == "Linux"
                                   and not self._is_video_file)
        self._consecutive_failures = 0
        self._reconnect_attempt = 0
        self._last_reconnect_time = 0.0  # 上次重连尝试的时间戳

        self._open_camera()

    # ── 公开接口 ──────────────────────────────────────────────────────────

    def is_opened(self) -> bool:
        return self.cap is not None and self.cap.isOpened()

    def read(self):
        """
        读取一帧。

        返回: (frame, timestamp) 或 (None, None)
          frame:     BGR 格式的 numpy 数组
          timestamp: 采集时刻的 Unix 时间戳
        """
        # ── 设备已断开 → 尝试重连 ──
        if not self.is_opened():
            if not self._reconnect_enabled:
                return None, None
            if self._reconnect_attempt >= config.CAMERA_RECONNECT_MAX_RETRIES:
                return None, None  # 重试耗尽，让决策层触发 STOP
            return self._try_reconnect()

        # ── 正常读取 ──
        try:
            ret, frame = self.cap.read()
        except cv2.error:
            ret, frame = False, None

        if not ret or frame is None:
            self._on_read_failure()
            return None, None

        # ── 读取成功：重置故障计数 ──
        self._consecutive_failures = 0
        self._reconnect_attempt = 0

        # 视频文件播放完毕 → 循环
        if self._is_video_file:
            pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            total = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
            if pos >= total:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # resize 到统一分辨率（保持宽高比）
        frame = self._resize_frame(frame)
        return frame, time.time()

    def release(self):
        if self.cap:
            self.cap.release()
            self.cap = None
        print(f"[Camera] 已释放 (设备: {self.camera_index})")

    # ── 画面缩放 ──────────────────────────────────────────────────────────

    def _resize_frame(self, frame):
        """
        将帧缩放到目标分辨率。
        若 PRESERVE_ASPECT_RATIO=True，使用 letterbox（保持宽高比+填充灰边）；
        否则使用暴力拉伸（兼容旧行为）。
        """
        if frame is None:
            return None
        src_h, src_w = frame.shape[:2]
        if src_w == self.width and src_h == self.height:
            return frame

        if config.PRESERVE_ASPECT_RATIO:
            scale = min(self.width / src_w, self.height / src_h)
            new_w, new_h = int(src_w * scale), int(src_h * scale)
            resized = cv2.resize(frame, (new_w, new_h))
            canvas = np.full((self.height, self.width, 3),
                             config.LETTERBOX_COLOR, dtype=np.uint8)
            y_off = (self.height - new_h) // 2
            x_off = (self.width - new_w) // 2
            canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
            return canvas
        else:
            return cv2.resize(frame, (self.width, self.height))

    # ── 曝光控制（隧道出口防过曝）──────────────────────────────────────

    def lock_exposure(self, value: int = -5):
        """
        锁定相机曝光为手动模式，固定曝光值。

        隧道场景：当 AE 被暗区拖累时，出口会严重过曝。
        锁曝光后，暗区由 CLAHE 补偿，亮区不再饱和。

        参数:
          value: 曝光值（V4L2 绝对值，默认 -5 = 偏暗以保护高光）
                 -13 ~ -1 递减（越小越暗），具体范围因相机而异

        返回: (success, message)
        """
        if self._is_video_file:
            return True, "视频文件无需曝光控制"

        ok = False
        msg_parts = []

        # 方式 1: OpenCV CAP_PROP（适用于多数 UVC 摄像头）
        try:
            if self.cap is not None:
                # 手动模式 (0.25 = manual in OpenCV's V4L2 mapping)
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # 1 = manual
                self.cap.set(cv2.CAP_PROP_EXPOSURE, value)
                msg_parts.append(f"OpenCV: exposure={value}, manual=1")
                ok = True
        except Exception as e:
            msg_parts.append(f"OpenCV曝光设置失败: {e}")

        # 方式 2: v4l2-ctl（Linux 回退方案）
        if platform.system() == "Linux":
            import subprocess
            dev = f"/dev/video{self.camera_index}" if isinstance(self.camera_index, int) else None
            if dev:
                try:
                    subprocess.run(
                        ["v4l2-ctl", "-d", dev, "-c", "auto_exposure=1"],
                        capture_output=True, timeout=2,
                    )
                    subprocess.run(
                        ["v4l2-ctl", "-d", dev, "-c", f"exposure_absolute={value}"],
                        capture_output=True, timeout=2,
                    )
                    msg_parts.append(f"v4l2-ctl: {dev} exposure={value}")
                    ok = True
                except Exception as e:
                    msg_parts.append(f"v4l2-ctl失败: {e}")

        msg = "; ".join(msg_parts)
        if ok:
            print(f"[Camera] 曝光已锁定: {msg}")
        else:
            print(f"[Camera] ⚠️ 曝光锁定失败: {msg}")
        return ok, msg

    def set_auto_exposure(self):
        """恢复自动曝光。"""
        if self._is_video_file or self.cap is None:
            return
        try:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)  # 3 = auto
            print("[Camera] 曝光已恢复自动模式")
        except Exception:
            pass

    def set_exposure_bias(self, bias: int = -4):
        """
        设置曝光补偿（不锁手动模式，仅偏置 AE）。
        部分摄像头支持，隧道场景建议 bias=-4（偏暗防过曝）。

        参数:
          bias: 曝光补偿值（通常 -8~8，负值=偏暗）
        """
        if self._is_video_file or self.cap is None:
            return
        try:
            # CAP_PROP_EXPOSURE 在 auto 模式下通常无效
            # 尝试用 v4l2 的 exposure_auto + exposure_absolute 组合
            if platform.system() == "Linux":
                import subprocess
                dev = f"/dev/video{self.camera_index}" if isinstance(self.camera_index, int) else None
                if dev:
                    subprocess.run(
                        ["v4l2-ctl", "-d", dev, "-c", f"exposure_auto=3"],
                        capture_output=True, timeout=2,
                    )
                    subprocess.run(
                        ["v4l2-ctl", "-d", dev, "-c", f"exposure_absolute={bias}"],
                        capture_output=True, timeout=2,
                    )
        except Exception:
            pass

    @staticmethod
    def enumerate() -> list[dict] | None:
        """
        手动枚举系统中所有摄像头（仅 Linux）。
        返回: [{'index': 0, 'path': '/dev/video0', 'name': '...'}, ...]
        仅在调试/运维时调用，不影响正常采集流程。
        """
        if platform.system() != "Linux":
            return None
        try:
            from hardware_discovery import discover_cameras
            return discover_cameras()
        except Exception:
            return None

    # ── 内部实现 ──────────────────────────────────────────────────────────

    def _open_camera(self) -> bool:
        """打开摄像头。返回 True 表示成功。"""
        try:
            self.cap = cv2.VideoCapture(self.camera_index, self._backend)
            if not self.cap.isOpened():
                print(f"[Camera] 无法打开设备: {self.camera_index}")
                self.cap = None
                return False

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)

            actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"[Camera] 已打开: {self.camera_index}, "
                  f"分辨率: {actual_w}x{actual_h}")
            return True

        except Exception as e:
            print(f"[Camera] 初始化异常: {e}")
            self.cap = None
            return False

    def _on_read_failure(self):
        """读取失败时关闭设备、递增计数。"""
        self._consecutive_failures += 1
        if self.cap:
            self.cap.release()
            self.cap = None

    def _try_reconnect(self):
        """
        尝试重连摄像头，使用指数退避。
        返回 (frame, timestamp) 或 (None, None)。

        退避策略（无死循环保证）:
          - 每次读失败后 _on_read_failure() 关闭设备
          - 下次 read() 到达此处时检查：距上次重连是否过了退避时间？
          - 未到时间 → 直接返回 None，不阻塞、不空转
          - 已到时间 → 尝试打开设备，递增尝试计数
          - 超过 max_retries → 永久放弃，由决策层 STOP 兜底
        """
        now = time.time()
        attempt = self._reconnect_attempt

        # 计算本次退避延迟：base * 2^attempt，上限 max
        delay = min(
            config.CAMERA_RECONNECT_BASE_DELAY_SEC * (2 ** attempt),
            config.CAMERA_RECONNECT_MAX_DELAY_SEC,
        )

        # 还没到重试时间 → 跳过，不阻塞主循环
        if now - self._last_reconnect_time < delay:
            return None, None

        self._last_reconnect_time = now
        self._reconnect_attempt += 1

        if self._open_camera():
            print(f"[Camera] 断线重连成功 "
                  f"(第 {self._reconnect_attempt} 次尝试)")
            # 重连成功，立即读一帧
            try:
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    frame = self._resize_frame(frame)
                    return frame, time.time()
            except cv2.error:
                pass
            # 读取失败，关闭重来
            self._on_read_failure()
            return None, None

        # 重连失败
        remaining = config.CAMERA_RECONNECT_MAX_RETRIES - self._reconnect_attempt
        if remaining > 0:
            print(f"[Camera] 重连失败, 剩余尝试: {remaining}")
        else:
            print(f"[Camera] 重连已耗尽 ({config.CAMERA_RECONNECT_MAX_RETRIES} 次), "
                  f"请检查硬件连接")
        return None, None
