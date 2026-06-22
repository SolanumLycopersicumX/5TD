"""
车辆控制接口模块 — 极创 JC-VCU-02 (差速驱动)。
通过 Modbus-RTU (RS232) 与 VCU 通信，下发线速度和角速度指令。

v2.2: 从阿克曼转向切换为差速驱动。
  - 运动模式从 Mode 1 (阿克曼) 改为 Mode 2 (原地旋转/差速)
  - steering 字段重新解释为归一化角速度 [-1, 1]
  - 写入 REG_ANGULAR_VEL (1041) 替代 REG_STEER_ANGLE (1042)
"""

import time
import struct
import logging
import serial

import config
from utils import Decision

logger = logging.getLogger(__name__)

# ── Modbus-RTU CRC16 查表 ────────────────────────────────────────────────
_CRC_TABLE = None

def _make_crc_table():
    global _CRC_TABLE
    if _CRC_TABLE is not None:
        return _CRC_TABLE
    table = []
    for i in range(256):
        crc = i
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
        table.append(crc)
    _CRC_TABLE = table
    return table


def _modbus_crc(data: bytes) -> int:
    """计算 Modbus-RTU CRC16，返回小端序的 16 位值。"""
    table = _make_crc_table()
    crc = 0xFFFF
    for b in data:
        crc = (crc >> 8) ^ table[(crc ^ b) & 0xFF]
    return crc


# ── VCU 寄存器地址 ───────────────────────────────────────────────────────
REG_LINEAR_VEL  = 1040   # P0.00  线速度，0.001 m/s，signed int16
REG_ANGULAR_VEL = 1041   # P0.01  角速度，0.001 rad/s，signed int16
REG_STEER_ANGLE = 1042   # P0.02  前轮角度，0.1 deg，signed int16
REG_MOTION_MODE = 1044   # P0.04  1=默认, 2=原地旋转, 3=侧移, 4=独立控制
REG_FUNC_CTRL   = 1045   # P0.05  bit0=急停, bit1=驻车, bit2=超声波避障, bit3=激光雷达避障
REG_DRIVER_EN   = 1049   # P0.09  1=使能, 2=失能


class VehicleController:
    """
    极创 JC-VCU-02 差速驱动底盘控制器 (v2.2)。

    Modbus-RTU over RS232:
      - 波特率 115200, 8N1
      - 从站地址 0x06
      - 功能码 06H (写单个寄存器), 10H (写多个)

    控制量:
      - 线速度 v (m/s): speed ∈ [0, 1] → linear_mps ∈ [0, MAX_LINEAR_SPEED_MPS]
      - 角速度 ω (rad/s): steering ∈ [-1, 1] → angular_radps ∈ [-MAX_ANGULAR_VELOCITY_RADPS, +MAX_ANGULAR_VELOCITY_RADPS]
    """

    def __init__(self, port=None, baudrate=None, vcu_addr=None):
        self._enabled = True
        self._emergency = False
        self._vcu_addr = vcu_addr if vcu_addr is not None else getattr(config, 'VCU_MODBUS_ADDR', 0x06)
        self._max_speed_ms = getattr(config, 'MAX_LINEAR_SPEED_MPS', 3.0)
        self._max_angular_radps = getattr(config, 'MAX_ANGULAR_VELOCITY_RADPS', 3.0)
        self._motion_mode = getattr(config, 'VCU_MOTION_MODE', 2)

        port = port or getattr(config, 'VCU_SERIAL_PORT', '/dev/ttyUSB0')
        baudrate = baudrate or getattr(config, 'VCU_BAUD_RATE', 115200)

        self._ser = None
        self._connect(port, baudrate)

    # ── 连接管理 ──────────────────────────────────────────────────────

    def _connect(self, port: str, baudrate: int):
        """打开串口并初始化 VCU (差速驱动模式)。"""
        try:
            self._ser = serial.Serial(
                port=port,
                baudrate=baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.05,
            )
            logger.info("VCU 串口已连接: %s @ %d bps", port, baudrate)

            # 初始化：使能驱动器 + 设为差速运动模式
            time.sleep(0.1)
            self._write_reg(REG_DRIVER_EN, 1)     # 使能
            time.sleep(0.02)
            self._write_reg(REG_MOTION_MODE, self._motion_mode)  # v2.2: Mode 2 = 差速
            time.sleep(0.02)
            self._write_reg(REG_LINEAR_VEL, 0)     # 线速度清零
            time.sleep(0.02)
            self._write_reg(REG_ANGULAR_VEL, 0)    # v2.2: 角速度清零 (替代前轮转角归零)
            logger.info("VCU 初始化完成：驱动器已使能，运动模式=%d (Differential)", self._motion_mode)

        except serial.SerialException as e:
            logger.warning("VCU 串口不可用 (%s)，回退到占位模式: %s", port, e)
            self._ser = None
        except Exception as e:
            logger.warning("VCU 初始化失败，回退到占位模式: %s", e)
            self._ser = None

    def is_connected(self) -> bool:
        return self._ser is not None and self._ser.is_open

    # ── 主控制接口 ────────────────────────────────────────────────────

    def send(self, decision: Decision):
        """
        下发控制指令到 VCU (v2.2: 差速驱动)。

        转换关系:
          decision.speed (0~1)     → 线速度 m/s → 寄存器值 = m/s × 1000
          decision.steering (-1~1) → 角速度 rad/s → 寄存器值 = rad/s × 1000
        """
        if decision is None:
            return
        if self._emergency:
            return
        if not self._enabled:
            return

        # 归一化值 → 物理量 (v2.2: steering → 角速度)
        linear_mps = decision.speed * self._max_speed_ms
        angular_radps = decision.steering * self._max_angular_radps

        # 物理量 → 寄存器整数（有符号 int16）
        lin_val = self._clamp_int16(int(linear_mps * 1000))
        ang_val = self._clamp_int16(int(angular_radps * 1000))

        # 调试输出
        if config.DEBUG_VIEW:
            print(f"[VC] {decision.command:<12} SPD={linear_mps:.2f}m/s ANG={angular_radps:+.2f}rad/s "
                  f"(reg: lin={lin_val}, ang={ang_val}) | {decision.reason}")

        if self._ser is None:
            return  # 占位模式，不发送

        try:
            # v2.2: 批量写入线速度 + 角速度（寄存器 1040~1041，共 2 个）
            self._write_multi_regs(REG_LINEAR_VEL, [lin_val, ang_val])
        except serial.SerialException as e:
            logger.error("VCU 写入失败: %s", e)
        except Exception as e:
            logger.error("VCU 通信异常: %s", e)

    def emergency_stop(self):
        """紧急停止：通过寄存器 P0.05 bit0 触发急停。"""
        self._emergency = True
        try:
            if self._ser is not None:
                self._write_reg(REG_FUNC_CTRL, 0x01)  # bit0=急停
                self._write_reg(REG_LINEAR_VEL, 0)
        except Exception as e:
            logger.error("急停指令发送失败: %s", e)
        logger.info("紧急停止！底盘已锁定。")

    def emergency_release(self):
        """释放紧急停止。"""
        self._emergency = False
        try:
            if self._ser is not None:
                self._write_reg(REG_FUNC_CTRL, 0x00)  # 清除急停
        except Exception as e:
            logger.error("急停释放失败: %s", e)
        logger.info("紧急停止已释放。")

    def enable(self):
        self._enabled = True
        if self._ser is not None:
            self._write_reg(REG_DRIVER_EN, 1)

    def disable(self):
        self._enabled = False
        if self._ser is not None:
            self._write_reg(REG_LINEAR_VEL, 0)
            self._write_reg(REG_DRIVER_EN, 2)  # 失能

    def close(self):
        """关闭串口连接。"""
        if self._ser is not None:
            try:
                self._write_reg(REG_LINEAR_VEL, 0)
                self._write_reg(REG_ANGULAR_VEL, 0)    # v2.2: 角速度清零
            except Exception:
                pass
            self._ser.close()
            logger.info("VCU 串口已关闭。")

    # ── Modbus 帧构建与发送 ───────────────────────────────────────────

    def _write_reg(self, addr: int, value: int):
        """功能码 06H — 写单个寄存器。"""
        frame = struct.pack('>BBHH',
            self._vcu_addr,   # 从站地址
            0x06,             # 功能码
            addr,             # 寄存器地址
            value & 0xFFFF,   # 数据值
        )
        crc = _modbus_crc(frame)
        frame += struct.pack('<H', crc)
        self._ser.write(frame)
        self._ser.flush()
        # 读回应（丢弃，只确认收到了）
        _ = self._ser.read(8)

    def _write_multi_regs(self, start_addr: int, values: list):
        """功能码 10H — 写多个寄存器。"""
        n = len(values)
        payload = struct.pack('>BBHHB',
            self._vcu_addr,      # 从站地址
            0x10,                # 功能码
            start_addr,          # 起始地址
            n,                   # 寄存器个数
            n * 2,               # 字节数
        )
        for v in values:
            payload += struct.pack('>H', v & 0xFFFF)
        crc = _modbus_crc(payload)
        payload += struct.pack('<H', crc)
        self._ser.write(payload)
        self._ser.flush()
        # 读回应
        _ = self._ser.read(8)

    # ── 工具 ──────────────────────────────────────────────────────────

    @staticmethod
    def _clamp_int16(val: int) -> int:
        """钳制到 signed int16 范围。"""
        return max(-32768, min(32767, val))
