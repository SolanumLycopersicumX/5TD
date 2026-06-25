#!/usr/bin/env python3
"""
RS232 / Modbus RTU Vehicle Controller
Keya VCU Protocol - 115200 baud, 8N1

Control registers (write with 06H or 10H):
  1040: linear velocity   (unit: 0.001 m/s, signed int16)
  1041: angular velocity  (unit: 0.001 rad/s, signed int16)
  1045: function control  (bit0=emergency stop, bit1=park, bit2=ultrasonic, bit3=lidar)
  1049: driver enable     (1=enable, 2=disable)

Status registers (read with 03H):
  4000: bus voltage        (0.01V)
  4002: battery max temp   (°C)
  4004: SOC                (%)
  4006: function status
  4009: device status
  4013: actual linear vel  (0.001 m/s)
  4014: actual angular vel (0.001 rad/s)
  4015: motor speed 1      (rpm)
  4016: motor speed 2      (rpm)
  4039: drive fault code 1
"""

import serial
import struct
import time

# Default node address (ADDR field in Modbus frame)
NODE_ADDR = 0x06

# Control register addresses
REG_LINEAR_VEL  = 1040   # 0.001 m/s
REG_ANGULAR_VEL = 1041   # 0.001 rad/s
REG_FUNC_CTRL   = 1045   # bit0=e-stop, bit1=park
REG_DRV_ENABLE  = 1049   # 1=enable, 2=disable

# Status register addresses
REG_VOLTAGE     = 4000
REG_TEMP_MAX    = 4002
REG_SOC         = 4004
REG_FUNC_STATUS = 4006
REG_DEV_STATUS  = 4009
REG_ACT_LINEAR  = 4013
REG_ACT_ANGULAR = 4014
REG_MOTOR_SPD1  = 4015
REG_MOTOR_SPD2  = 4016
REG_FAULT1      = 4039


def crc16(data: bytes) -> int:
    """Modbus CRC16"""
    crc = 0xFFFF
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = (crc >> 1) ^ 0xA001 if crc & 1 else crc >> 1
    return crc


def _read_regs(ser, addr, reg, count):
    """Modbus 03H: read `count` registers from `reg`. Returns list of signed int16, or None."""
    frame = bytes([addr, 0x03, reg >> 8, reg & 0xFF, count >> 8, count & 0xFF])
    crc = crc16(frame)
    ser.write(frame + bytes([crc & 0xFF, crc >> 8]))

    expected = 3 + count * 2 + 2
    resp = ser.read(expected)
    if len(resp) < expected:
        return None
    if crc16(resp[:-2]) != (resp[-2] | resp[-1] << 8):
        print("CRC error on read response")
        return None
    n = resp[2]
    return [struct.unpack('>h', resp[3 + i:5 + i])[0] for i in range(0, n, 2)]


def _write_reg(ser, addr, reg, value):
    """Modbus 06H: write single register."""
    val = value & 0xFFFF
    frame = bytes([addr, 0x06, reg >> 8, reg & 0xFF, val >> 8, val & 0xFF])
    crc = crc16(frame)
    ser.write(frame + bytes([crc & 0xFF, crc >> 8]))
    resp = ser.read(8)
    return len(resp) == 8


def _write_regs(ser, addr, reg, values):
    """Modbus 10H: write N registers (N >= 2, even). values = list of ints."""
    count = len(values)
    assert count >= 2 and count % 2 == 0, "10H requires N>=2 and N even"
    byte_count = count * 2
    header = bytes([addr, 0x10, reg >> 8, reg & 0xFF, count >> 8, count & 0xFF, byte_count])
    data = b''.join(struct.pack('>H', v & 0xFFFF) for v in values)
    frame = header + data
    crc = crc16(frame)
    ser.write(frame + bytes([crc & 0xFF, crc >> 8]))
    print("TX:", (frame + bytes([crc & 0xFF, crc >> 8])).hex())
    resp = ser.read(8)
    print("RX:", resp.hex())
    return len(resp) == 8


class VehicleController:
    def __init__(self, port='/dev/ttyUSB0', addr=NODE_ADDR, baudrate=115200):
        self.addr = addr
        self.ser = None
        try:
            self.ser = serial.Serial(
                port=port,
                baudrate=baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.5
            )
            print(f"Serial opened: {port}, addr=0x{addr:02X}")
        except Exception as e:
            print(f"Failed to open serial port: {e}")

    # === Motion Control ===
    # linear_vel : m/s   (positive = forward)
    # angular_vel: rad/s (positive = turn left / CCW)

    def set_velocity(self, linear_mps, angular_radps):
        """Set linear (m/s) and angular (rad/s) velocity."""
        lin = int(linear_mps * 1000)    # convert to 0.001 m/s units
        ang = int(angular_radps * 1000) # convert to 0.001 rad/s units
        _write_regs(self.ser, self.addr, REG_LINEAR_VEL, [lin, ang])

    def stop(self):
        self.set_velocity(0, 0)

    def emergency_stop(self):
        """Set emergency stop bit (bit0=1 in function control register)."""
        _write_reg(self.ser, self.addr, REG_FUNC_CTRL, 0x0001)

    def emergency_release(self):
        """Clear emergency stop."""
        _write_reg(self.ser, self.addr, REG_FUNC_CTRL, 0x0000)

    def enable(self):
        _write_reg(self.ser, self.addr, REG_DRV_ENABLE, 1)

    def disable(self):
        _write_reg(self.ser, self.addr, REG_DRV_ENABLE, 2)

    # === Differential Drive Helpers ===

    def forward(self, speed_mps=0.5):
        self.set_velocity(speed_mps, 0)

    def backward(self, speed_mps=0.5):
        self.set_velocity(-speed_mps, 0)

    def turn_left(self, angular_radps=0.5):
        """Rotate counter-clockwise in place."""
        self.set_velocity(0, angular_radps)

    def turn_right(self, angular_radps=0.5):
        """Rotate clockwise in place."""
        self.set_velocity(0, -angular_radps)

    def arc_left(self, speed_mps=0.5, angular_radps=0.3):
        """Move forward while turning left."""
        self.set_velocity(speed_mps, angular_radps)

    def arc_right(self, speed_mps=0.5, angular_radps=0.3):
        """Move forward while turning right."""
        self.set_velocity(speed_mps, -angular_radps)

    # === Read Status ===

    def read_voltage(self):
        """Returns bus voltage in Volts."""
        r = _read_regs(self.ser, self.addr, REG_VOLTAGE, 1)
        return r[0] * 0.01 if r else None

    def read_temperature(self):
        """Returns battery max temperature in °C."""
        r = _read_regs(self.ser, self.addr, REG_TEMP_MAX, 1)
        return r[0] if r else None

    def read_soc(self):
        """Returns state of charge in %."""
        r = _read_regs(self.ser, self.addr, REG_SOC, 1)
        return r[0] if r else None

    def read_actual_velocity(self):
        """Returns (linear_mps, angular_radps) tuple."""
        r = _read_regs(self.ser, self.addr, REG_ACT_LINEAR, 2)
        return (r[0] * 0.001, r[1] * 0.001) if r else None

    def read_motor_speeds(self):
        """Returns (speed1_rpm, speed2_rpm) tuple."""
        r = _read_regs(self.ser, self.addr, REG_MOTOR_SPD1, 2)
        return (r[0], r[1]) if r else None

    def read_fault(self):
        """Returns fault code for drive motor 1 (255=no fault, 0=device unused)."""
        r = _read_regs(self.ser, self.addr, REG_FAULT1, 1)
        return r[0] if r else None

    def read_device_status(self):
        """Returns device status word (see section 4.4)."""
        r = _read_regs(self.ser, self.addr, REG_DEV_STATUS, 1)
        return r[0] if r else None

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()


# ============ TEST CODE ============
if __name__ == "__main__":
    SERIAL_PORT   = '/dev/ttyUSB0'   # e.g. COM3 on Windows
    NODE_ADDR_CFG = 0x06             # default Keya VCU address
    TEST_SPEED    = 0.05              # m/s
    TEST_DURATION = 1.0              # seconds

    print("=" * 50)
    print("RS232 Modbus RTU Vehicle Controller Test")
    print("=" * 50)

    vehicle = VehicleController(port=SERIAL_PORT, addr=NODE_ADDR_CFG)

    if not vehicle.ser:
        print("Failed to initialize. Exiting.")
        exit(1)

    try:
        print("\n--- Reading Status ---")
        voltage = vehicle.read_voltage()
        if voltage is not None:
            print(f"Bus voltage : {voltage:.2f} V")

        soc = vehicle.read_soc()
        if soc is not None:
            print(f"SOC         : {soc} %")

        temp = vehicle.read_temperature()
        if temp is not None:
            print(f"Battery temp: {temp} °C")

        fault = vehicle.read_fault()
        if fault is not None:
            print(f"Fault code  : {fault} (255=OK)")

        # print("\n--- Enabling driver ---")
        # vehicle.enable()
        # time.sleep(0.2)

        # print("\n--- Movement Tests ---")

        # print(f"Forward at {TEST_SPEED} m/s ...")
        # vehicle.forward(TEST_SPEED)
        # time.sleep(TEST_DURATION)
        # vehicle.stop()
        # time.sleep(0.5)

        # print(f"Backward at {TEST_SPEED} m/s ...")
        # vehicle.backward(TEST_SPEED)
        # time.sleep(TEST_DURATION)
        # vehicle.stop()
        # time.sleep(0.5)

        # print("Turn left ...")
        # vehicle.turn_left(0.3)
        # time.sleep(TEST_DURATION)
        # vehicle.stop()
        # time.sleep(0.5)

        # print("Turn right ...")
        # vehicle.turn_right(0.3)
        # time.sleep(TEST_DURATION)
        # vehicle.stop()

        print("\n--- Tests Complete ---")

    except KeyboardInterrupt:
        print("\nInterrupted!")
    finally:
        print("Stopping and closing...")
        vehicle.stop()
        vehicle.close()
