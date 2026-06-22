#!/usr/bin/env python3
"""
极创 JC-VCU-02 Modbus-RTU 通信测试脚本。

用法:
  python3 test_vcu.py                # 交互测试（默认 /dev/ttyUSB0）
  python3 test_vcu.py /dev/ttyUSB1   # 指定串口
  python3 test_vcu.py --auto         # 交互 + 自动测试序列

功能:
  1. 检查串口是否可打开
  2. 测试 Modbus 通信（读取寄存器 4000 = 总线电压）
  3. 测试运动控制（速度 / 转向 / 急停）
  4. 读取底盘状态信息

依赖: pip install pyserial
"""

import sys
import time
import struct


# ── Modbus CRC16 ──────────────────────────────────────────────────────────
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


def modbus_crc(data: bytes) -> int:
    table = _make_crc_table()
    crc = 0xFFFF
    for b in data:
        crc = (crc >> 8) ^ table[(crc ^ b) & 0xFF]
    return crc


# ── Modbus 帧 ────────────────────────────────────────────────────────────
VCU_ADDR = 0x06


def make_read_frame(reg_addr: int, count: int) -> bytes:
    """功能码 03H — 读 N 个寄存器。"""
    frame = struct.pack('>BBHH', VCU_ADDR, 0x03, reg_addr, count)
    crc = modbus_crc(frame)
    return frame + struct.pack('<H', crc)


def make_write_frame(reg_addr: int, value: int) -> bytes:
    """功能码 06H — 写单个寄存器。"""
    frame = struct.pack('>BBHH', VCU_ADDR, 0x06, reg_addr, value & 0xFFFF)
    crc = modbus_crc(frame)
    return frame + struct.pack('<H', crc)


def make_write_multi(reg_start: int, values: list) -> bytes:
    """功能码 10H — 写多个寄存器。"""
    n = len(values)
    frame = struct.pack('>BBHHB', VCU_ADDR, 0x10, reg_start, n, n * 2)
    for v in values:
        frame += struct.pack('>H', v & 0xFFFF)
    crc = modbus_crc(frame)
    return frame + struct.pack('<H', crc)


# ── 寄存器定义 ───────────────────────────────────────────────────────────
REG_VOLTAGE     = 4000   # R0.00  总线电压
REG_LINEAR_VEL  = 1040   # P0.00  线速度
REG_STEER_ANGLE = 1042   # P0.02  前轮角度
REG_MOTION_MODE = 1044   # P0.04  运动模式
REG_FUNC_CTRL   = 1045   # P0.05  功能控制
REG_DRIVER_EN   = 1049   # P0.09  驱动器使能


def clamp16(val: int) -> int:
    return max(-32768, min(32767, val))


def run_test(ser, port: str):
    """完整自动测试序列。"""

    def send_and_read(frame: bytes, expect: int = 8, label: str = ""):
        ser.write(frame)
        ser.flush()
        time.sleep(0.02)
        resp = ser.read(expect)
        hex_str = resp.hex(' ') if resp else "(无响应)"
        if label:
            print(f"  {label}: {hex_str}")
        return resp

    print(f"\n{'='*60}")
    print(f"VCU 通信测试 — {port}")
    print(f"{'='*60}")

    # === 测试 1: 基本通信 — 读总线电压 ===
    print("\n[1/5] 测试 Modbus 通信 — 读取总线电压 (R0.00, 地址 4000)...")
    frame = make_read_frame(REG_VOLTAGE, 1)
    ser.write(frame)
    ser.flush()
    time.sleep(0.05)
    resp = ser.read(7)  # addr + func + byte_count + 2 data + 2 crc = 7
    print(f"  原始响应: {resp.hex(' ')}")

    if len(resp) >= 5:
        # 解析: [addr, func, byte_count, data_hi, data_lo, crc_lo, crc_hi]
        voltage_raw = (resp[3] << 8) | resp[4]
        voltage = voltage_raw * 0.1  # 精度 0.1V/0.01V (以实际为准)
        print(f"  ✅ 通信正常！总线电压 ≈ {voltage:.1f}V (raw={voltage_raw})")
    else:
        print(f"  ❌ 无有效响应，请检查: 1)接线 2)波特率 3)VCU地址")
        return False

    # === 测试 2: 驱动器使能 ===
    print("\n[2/5] 使能驱动器 (P0.09=1)...")
    send_and_read(make_write_frame(REG_DRIVER_EN, 1), 8, "响应")
    time.sleep(0.05)

    # === 测试 3: 设置运动模式 ===
    print("\n[3/5] 设置阿克曼默认模式 (P0.04=1)...")
    send_and_read(make_write_frame(REG_MOTION_MODE, 1), 8, "响应")
    time.sleep(0.05)

    # === 测试 4: 发送小幅运动指令 ===
    print("\n[4/5] 发送小幅运动测试（⚠️ 确认底盘悬空 / 安全！）")
    confirm = input("  底盘是否已悬空/确保安全？[y/N] ")
    if confirm.lower() != 'y':
        print("  已跳过运动测试。")
    else:
        lin_val = clamp16(int(0.3 * 1000))    # 0.3 m/s
        ang_val = clamp16(int(10 * 10))        # 10 deg
        print(f"  发送: 线速度={lin_val} (0.3m/s), 前轮角={ang_val} (10deg)")
        send_and_read(make_write_multi(REG_LINEAR_VEL, [lin_val, 0, ang_val]), 8, "响应")
        time.sleep(2.0)

        print("  停止...")
        send_and_read(make_write_multi(REG_LINEAR_VEL, [0, 0, 0]), 8, "响应")

    # === 测试 5: 急停测试 ===
    print("\n[5/5] 测试急停 (P0.05 bit0=1)...")
    send_and_read(make_write_frame(REG_FUNC_CTRL, 0x01), 8, "急停开启")
    time.sleep(0.5)
    print("  释放急停...")
    send_and_read(make_write_frame(REG_FUNC_CTRL, 0x00), 8, "急停释放")

    print(f"\n{'='*60}")
    print("测试完成！")
    print(f"{'='*60}")
    return True


def interactive(ser):
    """交互式命令控制。"""
    print("\n交互模式 — 输入命令:")
    print("  s <速度>         设置线速度 m/s (如: s 0.5)")
    print("  t <角度>         设置前轮角度 deg (如: t 15)")
    print("  stop / 0         停车")
    print("  estop            急停")
    print("  release          释放急停")
    print("  enable/disable   驱动器使能/失能")
    print("  status           读取总线电压")
    print("  q                退出")
    print()

    while True:
        try:
            cmd = input("VCU> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not cmd:
            continue
        if cmd == 'q':
            break

        parts = cmd.split()
        op = parts[0].lower()

        try:
            if op == 's' and len(parts) > 1:
                v = float(parts[1])
                val = clamp16(int(v * 1000))
                ser.write(make_write_frame(REG_LINEAR_VEL, val))
                ser.flush()
                _ = ser.read(8)
                print(f"  → 速度={v:.2f} m/s (reg={val})")

            elif op == 't' and len(parts) > 1:
                a = float(parts[1])
                val = clamp16(int(a * 10))
                ser.write(make_write_frame(REG_STEER_ANGLE, val))
                ser.flush()
                _ = ser.read(8)
                print(f"  → 转角={a:.1f}° (reg={val})")

            elif op in ('stop', '0'):
                ser.write(make_write_multi(REG_LINEAR_VEL, [0, 0, 0]))
                ser.flush()
                _ = ser.read(8)
                print("  → 停车")

            elif op == 'estop':
                ser.write(make_write_frame(REG_FUNC_CTRL, 0x01))
                ser.flush()
                _ = ser.read(8)
                print("  → 急停已触发！")

            elif op == 'release':
                ser.write(make_write_frame(REG_FUNC_CTRL, 0x00))
                ser.flush()
                _ = ser.read(8)
                print("  → 急停已释放")

            elif op == 'enable':
                ser.write(make_write_frame(REG_DRIVER_EN, 1))
                ser.flush(); _ = ser.read(8)
                print("  → 驱动器已使能")

            elif op == 'disable':
                ser.write(make_write_frame(REG_LINEAR_VEL, 0))
                ser.flush(); _ = ser.read(8)
                ser.write(make_write_frame(REG_DRIVER_EN, 2))
                ser.flush(); _ = ser.read(8)
                print("  → 驱动器已失能")

            elif op == 'status':
                ser.write(make_read_frame(REG_VOLTAGE, 1))
                ser.flush()
                time.sleep(0.05)
                resp = ser.read(7)
                if len(resp) >= 5:
                    v_raw = (resp[3] << 8) | resp[4]
                    print(f"  → 总线电压 ≈ {v_raw * 0.1:.1f}V (raw={v_raw})")
                else:
                    print("  → 无响应")

            else:
                print(f"  未知命令: {cmd}")

        except Exception as e:
            print(f"  ✗ 通信错误: {e}")


def main():
    import serial

    port = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyUSB0"
    mode = sys.argv[2] if len(sys.argv) > 2 else ""

    print(f"打开串口 {port} @ 115200 8N1...")

    try:
        ser = serial.Serial(
            port=port,
            baudrate=115200,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=0.1,
        )
    except serial.SerialException as e:
        print(f"❌ 无法打开串口 {port}: {e}")
        print("   请检查: 1)设备是否插入 2)权限 (sudo usermod -aG dialout $USER)")
        sys.exit(1)

    print(f"✅ 串口已打开: {ser.name}")

    if mode == "--auto":
        run_test(ser, port)
    else:
        # 先做一个快速通信检查
        print("通信检查...", end=" ")
        ser.write(make_read_frame(REG_VOLTAGE, 1))
        ser.flush()
        time.sleep(0.05)
        resp = ser.read(7)
        if len(resp) >= 5:
            print("✅ VCU 在线")
        else:
            print("⚠️ 无响应（可能是接线/参数问题，交互模式仍可用）")

        interactive(ser)

    ser.close()
    print("串口已关闭。")


if __name__ == "__main__":
    main()
