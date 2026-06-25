#!/usr/bin/env python3
"""
CAN Two-Wheel Differential Drive Vehicle Controller - Qt GUI
Dark theme with custom painted widgets
"""

import sys
import struct

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QGroupBox, QSlider, QPushButton
)
from PyQt5.QtCore import Qt, QTimer, QPointF, QRectF
from PyQt5.QtGui import QPainter, QColor, QPen, QBrush, QFont, QLinearGradient

import can

# ==================== Vehicle Controller ====================

class VehicleController:
    WRITE_4BYTE = 0x23
    WRITE_2BYTE = 0x2B
    WRITE_1BYTE = 0x2F
    READ_CMD = 0x40
    
    IDX_MOTOR_CMD = 0x2000
    IDX_EMERGENCY_STOP = 0x200C
    IDX_EMERGENCY_RELEASE = 0x200D
    IDX_STOP_ALL = 0x200E
    IDX_READ_CURRENT = 0x2100
    IDX_READ_SPEED = 0x2103
    IDX_READ_ENCODER = 0x2104
    IDX_READ_VOLTAGE = 0x210D
    IDX_READ_TEMP = 0x210F
    IDX_READ_FAULT = 0x2112
    
    def __init__(self, channel='can0', bustype='socketcan', node_id=1, bitrate=250000):
        self.node_id = node_id
        self.tx_id = 0x600 + node_id
        self.rx_id = 0x580 + node_id
        self.bus = None
        self.connected = False
        
        try:
            self.bus = can.interface.Bus(channel=channel, bustype=bustype, bitrate=bitrate)
            self.connected = True
        except Exception as e:
            print(f"CAN init failed: {e}")
    
    def _send(self, data):
        if not self.bus:
            return False
        try:
            msg = can.Message(arbitration_id=self.tx_id, data=data, is_extended_id=False)
            self.bus.send(msg)
            return True
        except:
            return False
    
    def _recv(self, timeout=0.1):
        if not self.bus:
            return None
        msg = self.bus.recv(timeout)
        if msg and msg.arbitration_id == self.rx_id:
            return msg.data
        return None
    
    def _write_sdo(self, index, subindex, value, size=4):
        cmd = {1: self.WRITE_1BYTE, 2: self.WRITE_2BYTE, 4: self.WRITE_4BYTE}.get(size, self.WRITE_4BYTE)
        idx_lo, idx_hi = index & 0xFF, (index >> 8) & 0xFF
        if size == 1:
            data = [cmd, idx_lo, idx_hi, subindex, value & 0xFF, 0, 0, 0]
        elif size == 2:
            data = [cmd, idx_lo, idx_hi, subindex, value & 0xFF, (value >> 8) & 0xFF, 0, 0]
        else:
            val_bytes = struct.pack('<i', value)
            data = [cmd, idx_lo, idx_hi, subindex] + list(val_bytes)
        return self._send(data)
    
    def _read_sdo(self, index, subindex):
        idx_lo, idx_hi = index & 0xFF, (index >> 8) & 0xFF
        if self._send([self.READ_CMD, idx_lo, idx_hi, subindex, 0, 0, 0, 0]):
            resp = self._recv()
            if resp:
                cmd = resp[0]
                if cmd == 0x4F:
                    return struct.unpack('<b', bytes([resp[4]]))[0]
                elif cmd == 0x4B:
                    return struct.unpack('<h', bytes(resp[4:6]))[0]
                elif cmd in (0x47, 0x43):
                    return struct.unpack('<i', bytes(resp[4:8]))[0]
        return None
    
    def set_speed(self, left, right):
        left = max(-1000, min(1000, int(left)))
        right = max(-1000, min(1000, int(right)))
        self._write_sdo(self.IDX_MOTOR_CMD, 1, right, 4)
        self._write_sdo(self.IDX_MOTOR_CMD, 2, left, 4)
    
    def stop(self):
        self._write_sdo(self.IDX_STOP_ALL, 1, 0, 1)
    
    def emergency_stop(self):
        self._write_sdo(self.IDX_EMERGENCY_STOP, 0, 0, 1)
    
    def emergency_release(self):
        self._write_sdo(self.IDX_EMERGENCY_RELEASE, 0, 0, 1)
    
    def read_voltage(self):
        return self._read_sdo(self.IDX_READ_VOLTAGE, 2)
    
    def read_temperature(self, ch=1):
        return self._read_sdo(self.IDX_READ_TEMP, ch)
    
    def read_current(self, ch):
        return self._read_sdo(self.IDX_READ_CURRENT, ch)
    
    def read_speed(self, ch):
        return self._read_sdo(self.IDX_READ_SPEED, ch)
    
    def read_encoder(self, ch):
        return self._read_sdo(self.IDX_READ_ENCODER, ch)
    
    def read_fault(self):
        return self._read_sdo(self.IDX_READ_FAULT, 0)
    
    def close(self):
        if self.bus:
            self.bus.shutdown()


# ==================== Custom Widgets ====================

class JoystickWidget(QWidget):
    """Interactive joystick widget for vehicle control."""
    
    def __init__(self, title="Control"):
        super().__init__()
        self.title = title
        self.x = 0.0
        self.y = 0.0
        self.dragging = False
        self.setMinimumSize(200, 200)
        self.setMouseTracking(True)
    
    def get_position(self):
        return (self.x, self.y)
    
    def reset(self):
        self.x = 0.0
        self.y = 0.0
        self.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        w, h = self.width(), self.height()
        cx, cy = w // 2, h // 2
        radius = min(w, h) // 2 - 25
        
        # Draw title
        painter.setPen(QColor(200, 200, 200))
        painter.setFont(QFont('Arial', 9, QFont.Bold))
        painter.drawText(0, 5, w, 20, Qt.AlignCenter, self.title)
        
        # Draw outer circle with gradient
        gradient = QLinearGradient(cx - radius, cy - radius, cx + radius, cy + radius)
        gradient.setColorAt(0, QColor(50, 50, 55))
        gradient.setColorAt(1, QColor(35, 35, 40))
        painter.setPen(QPen(QColor(80, 80, 85), 2))
        painter.setBrush(QBrush(gradient))
        painter.drawEllipse(cx - radius, cy - radius, radius * 2, radius * 2)
        
        # Draw crosshairs
        painter.setPen(QPen(QColor(70, 70, 75), 1))
        painter.drawLine(cx - radius + 10, cy, cx + radius - 10, cy)
        painter.drawLine(cx, cy - radius + 10, cx, cy + radius - 10)
        
        # Draw center circle
        painter.setPen(QPen(QColor(60, 60, 65), 1))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(cx - 20, cy - 20, 40, 40)
        
        # Draw stick position
        stick_x = cx + int(self.x * radius * 0.85)
        stick_y = cy - int(self.y * radius * 0.85)
        
        # Stick shadow
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor(0, 0, 0, 50)))
        painter.drawEllipse(stick_x - 16, stick_y - 14, 36, 36)
        
        # Stick gradient
        stick_gradient = QLinearGradient(stick_x - 18, stick_y - 18, stick_x + 18, stick_y + 18)
        if self.dragging:
            stick_gradient.setColorAt(0, QColor(0, 180, 255))
            stick_gradient.setColorAt(1, QColor(0, 120, 200))
        else:
            stick_gradient.setColorAt(0, QColor(0, 150, 230))
            stick_gradient.setColorAt(1, QColor(0, 100, 180))
        
        painter.setPen(QPen(QColor(0, 200, 255), 2))
        painter.setBrush(QBrush(stick_gradient))
        painter.drawEllipse(stick_x - 18, stick_y - 18, 36, 36)
        
        # Draw values
        painter.setPen(QColor(150, 150, 150))
        painter.setFont(QFont('Courier', 8))
        painter.drawText(5, h - 5, f"X:{self.x:+.2f} Y:{self.y:+.2f}")
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = True
            self._update_position(event.pos())
    
    def mouseMoveEvent(self, event):
        if self.dragging:
            self._update_position(event.pos())
    
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = False
            self.reset()
    
    def _update_position(self, pos):
        w, h = self.width(), self.height()
        cx, cy = w // 2, h // 2
        radius = min(w, h) // 2 - 25
        
        dx = (pos.x() - cx) / radius
        dy = -(pos.y() - cy) / radius
        
        # Clamp to circle
        dist = (dx**2 + dy**2) ** 0.5
        if dist > 1:
            dx /= dist
            dy /= dist
        
        self.x = max(-1, min(1, dx))
        self.y = max(-1, min(1, dy))
        self.update()


class SpeedGaugeWidget(QWidget):
    """Vertical speed gauge widget."""
    
    def __init__(self, title="Speed", color=QColor(0, 180, 255)):
        super().__init__()
        self.title = title
        self.value = 0
        self.color = color
        self.setMinimumSize(60, 180)
    
    def set_value(self, value):
        self.value = max(-1000, min(1000, value))
        self.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        w, h = self.width(), self.height()
        
        # Draw title
        painter.setPen(QColor(200, 200, 200))
        painter.setFont(QFont('Arial', 9, QFont.Bold))
        painter.drawText(0, 5, w, 18, Qt.AlignCenter, self.title)
        
        # Gauge dimensions
        gauge_x = w // 2 - 15
        gauge_y = 28
        gauge_w = 30
        gauge_h = h - 70
        center_y = gauge_y + gauge_h // 2
        
        # Draw gauge background
        painter.setPen(QPen(QColor(70, 70, 75), 1))
        painter.setBrush(QBrush(QColor(40, 40, 45)))
        painter.drawRoundedRect(gauge_x, gauge_y, gauge_w, gauge_h, 5, 5)
        
        # Draw center line
        painter.setPen(QPen(QColor(100, 100, 105), 1))
        painter.drawLine(gauge_x + 5, center_y, gauge_x + gauge_w - 5, center_y)
        
        # Draw tick marks
        painter.setPen(QPen(QColor(80, 80, 85), 1))
        for i in range(5):
            y = gauge_y + int(i * gauge_h / 4)
            painter.drawLine(gauge_x - 3, y, gauge_x, y)
        
        # Draw value bar
        if self.value != 0:
            bar_height = abs(self.value) / 1000 * (gauge_h // 2 - 5)
            
            if self.value > 0:
                gradient = QLinearGradient(0, center_y - bar_height, 0, center_y)
                gradient.setColorAt(0, self.color)
                gradient.setColorAt(1, self.color.darker(150))
                painter.setBrush(QBrush(gradient))
                painter.setPen(Qt.NoPen)
                painter.drawRoundedRect(gauge_x + 4, int(center_y - bar_height), 
                                       gauge_w - 8, int(bar_height), 3, 3)
            else:
                gradient = QLinearGradient(0, center_y, 0, center_y + bar_height)
                gradient.setColorAt(0, self.color.darker(150))
                gradient.setColorAt(1, self.color)
                painter.setBrush(QBrush(gradient))
                painter.setPen(Qt.NoPen)
                painter.drawRoundedRect(gauge_x + 4, center_y, 
                                       gauge_w - 8, int(bar_height), 3, 3)
        
        # Draw value text
        painter.setPen(QColor(200, 200, 200))
        painter.setFont(QFont('Courier', 10, QFont.Bold))
        painter.drawText(0, h - 35, w, 20, Qt.AlignCenter, f"{self.value:+d}")
        
        # Draw scale labels
        painter.setFont(QFont('Courier', 7))
        painter.setPen(QColor(120, 120, 120))
        painter.drawText(0, gauge_y - 2, w, 12, Qt.AlignCenter, "+1000")
        painter.drawText(0, gauge_y + gauge_h - 8, w, 12, Qt.AlignCenter, "-1000")


class StatusIndicator(QWidget):
    """Small status indicator with label and value."""
    
    def __init__(self, title="Status"):
        super().__init__()
        self.title = title
        self.value = "--"
        self.status = 0  # 0=normal, 1=warning, 2=error
        self.setMinimumSize(80, 50)
    
    def set_value(self, value, status=0):
        self.value = value
        self.status = status
        self.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        w, h = self.width(), self.height()
        
        # Background
        painter.setPen(QPen(QColor(60, 60, 65), 1))
        painter.setBrush(QBrush(QColor(45, 45, 50)))
        painter.drawRoundedRect(2, 2, w - 4, h - 4, 5, 5)
        
        # Title
        painter.setPen(QColor(150, 150, 150))
        painter.setFont(QFont('Arial', 8))
        painter.drawText(5, 5, w - 10, 16, Qt.AlignCenter, self.title)
        
        # Value with status color
        colors = [QColor(100, 255, 150), QColor(255, 220, 100), QColor(255, 100, 100)]
        painter.setPen(colors[self.status])
        painter.setFont(QFont('Courier', 11, QFont.Bold))
        painter.drawText(5, 22, w - 10, 22, Qt.AlignCenter, str(self.value))


class KeyButton(QWidget):
    """Keyboard key style button."""
    
    def __init__(self, text, key_hint=""):
        super().__init__()
        self.text = text
        self.key_hint = key_hint
        self.pressed = False
        self.setMinimumSize(60, 50)
    
    def set_pressed(self, pressed):
        self.pressed = pressed
        self.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        w, h = self.width(), self.height()
        
        # Button background
        if self.pressed:
            painter.setBrush(QBrush(QColor(0, 150, 200)))
            painter.setPen(QPen(QColor(0, 180, 230), 2))
        else:
            gradient = QLinearGradient(0, 0, 0, h)
            gradient.setColorAt(0, QColor(70, 70, 75))
            gradient.setColorAt(1, QColor(50, 50, 55))
            painter.setBrush(QBrush(gradient))
            painter.setPen(QPen(QColor(90, 90, 95), 1))
        
        painter.drawRoundedRect(3, 3, w - 6, h - 6, 6, 6)
        
        # Text
        painter.setPen(QColor(220, 220, 220) if self.pressed else QColor(180, 180, 180))
        painter.setFont(QFont('Arial', 14, QFont.Bold))
        painter.drawText(0, 0, w, h - 8, Qt.AlignCenter, self.text)
        
        # Key hint
        if self.key_hint:
            painter.setFont(QFont('Arial', 8))
            painter.setPen(QColor(120, 120, 120))
            painter.drawText(0, h - 18, w, 15, Qt.AlignCenter, self.key_hint)


# ==================== Main Window ====================

class ControllerGUI(QMainWindow):
    def __init__(self, channel='can0', node_id=1):
        super().__init__()
        self.vehicle = VehicleController(channel=channel, node_id=node_id)
        self.max_speed = 100
        self.keys_pressed = set()
        
        self.init_ui()
        
        # Timers
        self.control_timer = QTimer()
        self.control_timer.timeout.connect(self.update_control)
        self.control_timer.start(50)
        
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_status)
        self.status_timer.start(500)
    
    def init_ui(self):
        self.setWindowTitle('CAN Vehicle Controller')
        self.setStyleSheet("""
            QMainWindow, QWidget { background-color: #2b2b2b; color: #ffffff; }
            QGroupBox { 
                border: 1px solid #404045; 
                border-radius: 6px; 
                margin-top: 12px; 
                padding-top: 10px;
                font-weight: bold;
            }
            QGroupBox::title { 
                subcontrol-origin: margin; 
                left: 10px; 
                padding: 0 5px;
                color: #aaaaaa;
            }
            QSlider::groove:horizontal {
                height: 8px;
                background: #404045;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #00a0d0;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
            QPushButton {
                background-color: #404045;
                border: 1px solid #505055;
                border-radius: 5px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #505055; }
            QPushButton:pressed { background-color: #00a0d0; }
        """)
        
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(10)
        
        # Title and connection status
        title_layout = QHBoxLayout()
        title = QLabel('Differential Drive Controller')
        title.setFont(QFont('Arial', 14, QFont.Bold))
        title_layout.addWidget(title)
        title_layout.addStretch()
        
        self.conn_status = QLabel('● Disconnected')
        self.conn_status.setStyleSheet('color: #ff6666;')
        if self.vehicle.connected:
            self.conn_status.setText('● Connected')
            self.conn_status.setStyleSheet('color: #66ff66;')
        title_layout.addWidget(self.conn_status)
        layout.addLayout(title_layout)
        
        # Main content
        content = QHBoxLayout()
        layout.addLayout(content)
        
        # Left - Joystick control
        left_group = QGroupBox("Joystick Control")
        left_layout = QVBoxLayout(left_group)
        self.joystick = JoystickWidget("Drag to Control")
        left_layout.addWidget(self.joystick)
        content.addWidget(left_group)
        
        # Center - Speed gauges
        center_group = QGroupBox("Motor Output")
        center_layout = QHBoxLayout(center_group)
        self.left_gauge = SpeedGaugeWidget("Left", QColor(0, 200, 100))
        self.right_gauge = SpeedGaugeWidget("Right", QColor(0, 150, 255))
        center_layout.addWidget(self.left_gauge)
        center_layout.addWidget(self.right_gauge)
        content.addWidget(center_group)
        
        # Right - Keyboard control
        right_group = QGroupBox("Keyboard Control")
        right_layout = QVBoxLayout(right_group)
        
        # Direction buttons
        btn_grid = QGridLayout()
        btn_grid.setSpacing(5)
        
        self.btn_w = KeyButton("▲", "W")
        self.btn_a = KeyButton("◄", "A")
        self.btn_s = KeyButton("▼", "S")
        self.btn_d = KeyButton("►", "D")
        self.btn_stop = KeyButton("■", "Space")
        
        btn_grid.addWidget(self.btn_w, 0, 1)
        btn_grid.addWidget(self.btn_a, 1, 0)
        btn_grid.addWidget(self.btn_stop, 1, 1)
        btn_grid.addWidget(self.btn_d, 1, 2)
        btn_grid.addWidget(self.btn_s, 2, 1)
        
        right_layout.addLayout(btn_grid)
        right_layout.addStretch()
        content.addWidget(right_group)
        
        # Speed slider
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("Max Speed:"))
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(10, 100)
        self.speed_slider.setValue(10)
        self.speed_slider.valueChanged.connect(self.on_speed_change)
        speed_layout.addWidget(self.speed_slider)
        self.speed_label = QLabel("10")
        self.speed_label.setMinimumWidth(40)
        speed_layout.addWidget(self.speed_label)
        layout.addLayout(speed_layout)
        
        # Status indicators
        status_group = QGroupBox("Vehicle Status")
        status_layout = QHBoxLayout(status_group)
        
        self.voltage_ind = StatusIndicator("Voltage")
        self.temp_ind = StatusIndicator("Temp")
        self.current_l_ind = StatusIndicator("Current L")
        self.current_r_ind = StatusIndicator("Current R")
        self.encoder_l_ind = StatusIndicator("Encoder L")
        self.encoder_r_ind = StatusIndicator("Encoder R")
        self.fault_ind = StatusIndicator("Fault")
        
        for ind in [self.voltage_ind, self.temp_ind, self.current_l_ind, 
                    self.current_r_ind, self.encoder_l_ind, self.encoder_r_ind, self.fault_ind]:
            status_layout.addWidget(ind)
        
        layout.addWidget(status_group)
        
        # Emergency buttons
        emg_layout = QHBoxLayout()
        
        self.emg_stop_btn = QPushButton("EMERGENCY STOP")
        self.emg_stop_btn.setStyleSheet("""
            QPushButton { background-color: #cc3333; color: white; font-size: 14px; padding: 12px; }
            QPushButton:hover { background-color: #dd4444; }
            QPushButton:pressed { background-color: #ff5555; }
        """)
        self.emg_stop_btn.clicked.connect(self.emergency_stop)
        emg_layout.addWidget(self.emg_stop_btn)
        
        self.emg_release_btn = QPushButton("Release Emergency")
        self.emg_release_btn.clicked.connect(self.emergency_release)
        emg_layout.addWidget(self.emg_release_btn)
        
        layout.addLayout(emg_layout)
        
        self.setFixedSize(700, 550)
        self.setFocusPolicy(Qt.StrongFocus)
    
    def on_speed_change(self, value):
        self.max_speed = value
        self.speed_label.setText(str(value))
    
    def update_control(self):
        # Get control input from joystick or keyboard
        jx, jy = self.joystick.get_position()
        
        # Keyboard override
        if self.keys_pressed:
            jx, jy = 0, 0
            if Qt.Key_W in self.keys_pressed:
                jy = 1
            if Qt.Key_S in self.keys_pressed:
                jy = -1
            if Qt.Key_A in self.keys_pressed:
                jx = -1
            if Qt.Key_D in self.keys_pressed:
                jx = 1
        
        # Differential drive mixing
        left = int((jy + jx) * self.max_speed)
        right = int((jy - jx) * self.max_speed)
        
        left = max(-1000, min(1000, left))
        right = max(-1000, min(1000, right))
        
        # Update gauges
        self.left_gauge.set_value(left)
        self.right_gauge.set_value(right)
        
        # Send to vehicle
        if self.vehicle.connected:
            self.vehicle.set_speed(left, right)
    
    def update_status(self):
        if not self.vehicle.connected:
            return
        
        v = self.vehicle.read_voltage()
        if v is not None:
            status = 0 if 20 < v/10 < 30 else (1 if 18 < v/10 < 32 else 2)
            self.voltage_ind.set_value(f"{v/10:.1f}V", status)
        
        t = self.vehicle.read_temperature(1)
        if t is not None:
            status = 0 if t < 50 else (1 if t < 70 else 2)
            self.temp_ind.set_value(f"{t}°C", status)
        
        cl = self.vehicle.read_current(1)
        if cl is not None:
            self.current_l_ind.set_value(f"{cl/10:.1f}A")
        
        cr = self.vehicle.read_current(2)
        if cr is not None:
            self.current_r_ind.set_value(f"{cr/10:.1f}A")
        
        el = self.vehicle.read_encoder(1)
        if el is not None:
            self.encoder_l_ind.set_value(str(el))
        
        er = self.vehicle.read_encoder(2)
        if er is not None:
            self.encoder_r_ind.set_value(str(er))
        
        f = self.vehicle.read_fault()
        if f is not None:
            self.fault_ind.set_value(str(f), 2 if f > 0 else 0)
    
    def emergency_stop(self):
        if self.vehicle.connected:
            self.vehicle.emergency_stop()
        self.joystick.reset()
        self.left_gauge.set_value(0)
        self.right_gauge.set_value(0)
    
    def emergency_release(self):
        if self.vehicle.connected:
            self.vehicle.emergency_release()
    
    def keyPressEvent(self, event):
        if event.isAutoRepeat():
            return
        key = event.key()
        if key in (Qt.Key_W, Qt.Key_A, Qt.Key_S, Qt.Key_D):
            self.keys_pressed.add(key)
            self._update_key_buttons()
        elif key == Qt.Key_Space:
            self.emergency_stop()
            self.btn_stop.set_pressed(True)
    
    def keyReleaseEvent(self, event):
        if event.isAutoRepeat():
            return
        key = event.key()
        if key in (Qt.Key_W, Qt.Key_A, Qt.Key_S, Qt.Key_D):
            self.keys_pressed.discard(key)
            self._update_key_buttons()
        elif key == Qt.Key_Space:
            self.btn_stop.set_pressed(False)
    
    def _update_key_buttons(self):
        self.btn_w.set_pressed(Qt.Key_W in self.keys_pressed)
        self.btn_a.set_pressed(Qt.Key_A in self.keys_pressed)
        self.btn_s.set_pressed(Qt.Key_S in self.keys_pressed)
        self.btn_d.set_pressed(Qt.Key_D in self.keys_pressed)
    
    def closeEvent(self, event):
        if self.vehicle:
            self.vehicle.stop()
            self.vehicle.close()
        event.accept()


# ==================== Main ====================

if __name__ == "__main__":
    # === Configuration ===
    CAN_CHANNEL = 'can0'
    NODE_ID = 1
    
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = ControllerGUI(channel=CAN_CHANNEL, node_id=NODE_ID)
    window.show()
    
    sys.exit(app.exec_())
