#!/usr/bin/env python3
"""Low-speed RS232 / Modbus RTU vehicle control GUI."""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from PyQt5.QtCore import QPointF, QRectF, Qt, QTimer
    from PyQt5.QtGui import QColor, QFont, QLinearGradient, QPainter, QPen, QBrush
    from PyQt5.QtWidgets import (
        QApplication,
        QCheckBox,
        QDoubleSpinBox,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QPushButton,
        QSpinBox,
        QVBoxLayout,
        QWidget,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - user-facing import guard
    raise SystemExit(
        "PyQt5 is required. Run with /usr/bin/python3, for example:\n"
        "  /usr/bin/python3 tools/robot/rs232_vehicle_gui.py"
    ) from exc

from src.tunnel_nav.manual_control import (
    ManualControlLimits,
    manual_command_from_axes,
    ramp_command,
)

DRIVER_PATH = ROOT / "1" / "driver_controller.py"


def parse_node_addr(value: str) -> int:
    addr = int(value, 0)
    if not 0 <= addr <= 0xFF:
        raise argparse.ArgumentTypeError("node address must be between 0x00 and 0xFF")
    return addr


def load_driver_module():
    spec = importlib.util.spec_from_file_location("legacy_rs232_driver_controller", DRIVER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load RS232 driver from {DRIVER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Rs232VehicleAdapter:
    """Thin runtime adapter around the unmodified legacy RS232 driver."""

    def __init__(self):
        self.controller = None
        self.connected = False

    def connect(self, *, port: str, addr: int, baudrate: int, timeout_s: float = 0.05) -> None:
        self.close()
        module = load_driver_module()
        controller = module.VehicleController(port=port, addr=addr, baudrate=baudrate)
        if not getattr(controller, "ser", None):
            raise RuntimeError(f"Failed to open serial port {port}")
        controller.ser.timeout = timeout_s
        self.controller = controller
        self.connected = True

    def _call(self, name: str, *args):
        if not self.connected or self.controller is None:
            return None
        method = getattr(self.controller, name)
        with contextlib.redirect_stdout(io.StringIO()):
            return method(*args)

    def set_velocity(self, linear_mps: float, angular_radps: float) -> None:
        self._call("set_velocity", linear_mps, angular_radps)

    def soft_stop(self) -> None:
        self._call("stop")

    def emergency_stop(self) -> None:
        self._call("emergency_stop")

    def emergency_release(self) -> None:
        self._call("emergency_release")

    def enable(self) -> None:
        self._call("enable")

    def disable(self) -> None:
        self._call("disable")

    def read_voltage(self):
        return self._call("read_voltage")

    def read_soc(self):
        return self._call("read_soc")

    def read_fault(self):
        return self._call("read_fault")

    def close(self) -> None:
        if self.controller is not None:
            try:
                self.soft_stop()
                time.sleep(0.05)
                self.disable()
            finally:
                self.controller.close()
        self.controller = None
        self.connected = False


class JoystickWidget(QWidget):
    """Small on-screen joystick with center-return behavior."""

    def __init__(self):
        super().__init__()
        self.x = 0.0
        self.y = 0.0
        self.dragging = False
        self.setMinimumSize(220, 220)
        self.setMouseTracking(True)

    def get_position(self) -> tuple[float, float]:
        return self.x, self.y

    def reset(self) -> None:
        self.x = 0.0
        self.y = 0.0
        self.dragging = False
        self.update()

    def paintEvent(self, event):  # noqa: N802 - Qt override
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w, h = self.width(), self.height()
        cx, cy = w // 2, h // 2
        radius = min(w, h) // 2 - 28

        gradient = QLinearGradient(cx - radius, cy - radius, cx + radius, cy + radius)
        gradient.setColorAt(0, QColor(48, 52, 56))
        gradient.setColorAt(1, QColor(31, 34, 37))
        painter.setPen(QPen(QColor(88, 96, 104), 2))
        painter.setBrush(QBrush(gradient))
        painter.drawEllipse(cx - radius, cy - radius, radius * 2, radius * 2)

        painter.setPen(QPen(QColor(96, 104, 112), 1))
        painter.drawLine(cx - radius + 12, cy, cx + radius - 12, cy)
        painter.drawLine(cx, cy - radius + 12, cx, cy + radius - 12)

        knob_x = cx + int(self.x * radius * 0.82)
        knob_y = cy - int(self.y * radius * 0.82)
        painter.setPen(QPen(QColor(40, 170, 220), 2))
        painter.setBrush(QBrush(QColor(0, 135, 200) if self.dragging else QColor(0, 110, 170)))
        painter.drawEllipse(knob_x - 18, knob_y - 18, 36, 36)

        painter.setPen(QColor(215, 220, 224))
        painter.setFont(QFont("Arial", 9))
        painter.drawText(QRectF(0, h - 24, w, 20), Qt.AlignCenter, f"X {self.x:+.2f}  Y {self.y:+.2f}")

    def mousePressEvent(self, event):  # noqa: N802 - Qt override
        if event.button() == Qt.LeftButton:
            self.dragging = True
            self._update_position(event.pos())

    def mouseMoveEvent(self, event):  # noqa: N802 - Qt override
        if self.dragging:
            self._update_position(event.pos())

    def mouseReleaseEvent(self, event):  # noqa: N802 - Qt override
        if event.button() == Qt.LeftButton:
            self.reset()

    def _update_position(self, pos):
        w, h = self.width(), self.height()
        cx, cy = w // 2, h // 2
        radius = min(w, h) // 2 - 28
        dx = (pos.x() - cx) / radius
        dy = -(pos.y() - cy) / radius
        dist = (dx * dx + dy * dy) ** 0.5
        if dist > 1.0:
            dx /= dist
            dy /= dist
        self.x = max(-1.0, min(1.0, dx))
        self.y = max(-1.0, min(1.0, dy))
        self.update()


class KeyButton(QPushButton):
    def __init__(self, text: str):
        super().__init__(text)
        self.setFixedSize(54, 42)
        self.setCheckable(False)
        self.setStyleSheet(
            "QPushButton { background: #343a40; color: #f1f3f5; border: 1px solid #59616a; "
            "border-radius: 5px; font-size: 16px; font-weight: bold; }"
            "QPushButton:pressed { background: #1971c2; }"
        )

    def set_pressed(self, pressed: bool) -> None:
        self.setStyleSheet(
            "QPushButton { background: "
            + ("#1971c2" if pressed else "#343a40")
            + "; color: #f1f3f5; border: 1px solid #59616a; border-radius: 5px; "
            "font-size: 16px; font-weight: bold; }"
        )


class MainWindow(QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.vehicle = Rs232VehicleAdapter()
        self.keys_pressed: set[int] = set()
        self.current_command = (0.0, 0.0)
        self.target_command = (0.0, 0.0)
        self.control_active = False
        self.last_update = time.monotonic()

        self.setWindowTitle("RS232 Vehicle Control")
        self.setFocusPolicy(Qt.StrongFocus)
        self._build_ui(args)

        self.control_timer = QTimer(self)
        self.control_timer.timeout.connect(self.update_control)
        self.control_timer.start(100)

        self.status_timer = QTimer(self)
        self.status_timer.timeout.connect(self.update_status)
        self.status_timer.start(1000)

    def _build_ui(self, args) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        conn_group = QGroupBox("Connection")
        conn = QGridLayout(conn_group)
        self.port_edit = QLineEdit(args.port)
        self.addr_edit = QLineEdit(f"0x{args.addr:02X}")
        self.baud_edit = QLineEdit(str(args.baudrate))
        self.connect_btn = QPushButton("Connect")
        self.disconnect_btn = QPushButton("Disconnect")
        self.disconnect_btn.setEnabled(False)
        self.connect_btn.clicked.connect(self.connect_vehicle)
        self.disconnect_btn.clicked.connect(self.disconnect_vehicle)
        conn.addWidget(QLabel("Port"), 0, 0)
        conn.addWidget(self.port_edit, 0, 1)
        conn.addWidget(QLabel("Addr"), 0, 2)
        conn.addWidget(self.addr_edit, 0, 3)
        conn.addWidget(QLabel("Baud"), 0, 4)
        conn.addWidget(self.baud_edit, 0, 5)
        conn.addWidget(self.connect_btn, 0, 6)
        conn.addWidget(self.disconnect_btn, 0, 7)
        root.addWidget(conn_group)

        controls = QHBoxLayout()
        joystick_group = QGroupBox("Joystick")
        joystick_layout = QVBoxLayout(joystick_group)
        self.joystick = JoystickWidget()
        joystick_layout.addWidget(self.joystick)
        controls.addWidget(joystick_group)

        key_group = QGroupBox("Keyboard")
        key_layout = QVBoxLayout(key_group)
        grid = QGridLayout()
        self.btn_w = KeyButton("W")
        self.btn_a = KeyButton("A")
        self.btn_s = KeyButton("S")
        self.btn_d = KeyButton("D")
        self.btn_stop = KeyButton("■")
        grid.addWidget(self.btn_w, 0, 1)
        grid.addWidget(self.btn_a, 1, 0)
        grid.addWidget(self.btn_stop, 1, 1)
        grid.addWidget(self.btn_d, 1, 2)
        grid.addWidget(self.btn_s, 2, 1)
        key_layout.addLayout(grid)
        key_layout.addWidget(QLabel("W/S forward/back, A/D rotate. Space/K = soft stop."))
        controls.addWidget(key_group)
        root.addLayout(controls)

        tuning_group = QGroupBox("Soft Control Limits")
        tuning = QGridLayout(tuning_group)
        self.linear_spin = self._double_spin(0.005, 0.20, args.linear, 0.005, " m/s")
        self.angular_spin = self._double_spin(0.02, 0.60, args.angular, 0.01, " rad/s")
        self.deadzone_spin = self._double_spin(0.00, 0.50, args.deadzone, 0.01, "")
        self.lin_accel_spin = self._double_spin(0.01, 0.50, args.linear_accel, 0.01, " m/s²")
        self.ang_accel_spin = self._double_spin(0.03, 1.50, args.angular_accel, 0.01, " rad/s²")
        self.turn_start_spin = self._double_spin(0.00, 0.30, args.turn_start, 0.005, " rad/s")
        tuning.addWidget(QLabel("Max linear"), 0, 0)
        tuning.addWidget(self.linear_spin, 0, 1)
        tuning.addWidget(QLabel("Max angular"), 0, 2)
        tuning.addWidget(self.angular_spin, 0, 3)
        tuning.addWidget(QLabel("Deadzone"), 1, 0)
        tuning.addWidget(self.deadzone_spin, 1, 1)
        tuning.addWidget(QLabel("Linear accel"), 1, 2)
        tuning.addWidget(self.lin_accel_spin, 1, 3)
        tuning.addWidget(QLabel("Angular accel"), 1, 4)
        tuning.addWidget(self.ang_accel_spin, 1, 5)
        tuning.addWidget(QLabel("Turn start"), 2, 0)
        tuning.addWidget(self.turn_start_spin, 2, 1)
        root.addWidget(tuning_group)

        safety = QHBoxLayout()
        self.active_box = QCheckBox("Control active")
        self.active_box.stateChanged.connect(self.on_active_changed)
        self.release_btn = QPushButton("Release E-Stop")
        self.enable_btn = QPushButton("Enable Driver")
        self.disable_btn = QPushButton("Disable Driver")
        self.stop_btn = QPushButton("Soft Stop")
        self.estop_btn = QPushButton("EMERGENCY STOP")
        self.estop_btn.setStyleSheet("QPushButton { background: #c92a2a; color: white; font-weight: bold; padding: 10px; }")
        self.release_btn.clicked.connect(lambda: self.safe_call("emergency_release"))
        self.enable_btn.clicked.connect(lambda: self.safe_call("enable"))
        self.disable_btn.clicked.connect(lambda: self.safe_call("disable"))
        self.stop_btn.clicked.connect(self.soft_stop)
        self.estop_btn.clicked.connect(self.emergency_stop)
        for widget in [self.active_box, self.release_btn, self.enable_btn, self.disable_btn, self.stop_btn, self.estop_btn]:
            safety.addWidget(widget)
        root.addLayout(safety)

        telemetry = QGroupBox("Command / Status")
        tel = QGridLayout(telemetry)
        self.target_label = QLabel("target: 0.000 m/s, 0.000 rad/s")
        self.current_label = QLabel("current: 0.000 m/s, 0.000 rad/s")
        self.status_label = QLabel("Disconnected")
        self.status_label.setWordWrap(True)
        tel.addWidget(self.target_label, 0, 0, 1, 2)
        tel.addWidget(self.current_label, 1, 0, 1, 2)
        tel.addWidget(self.status_label, 2, 0, 1, 2)
        root.addWidget(telemetry)

        self.resize(820, 620)

    @staticmethod
    def _double_spin(low: float, high: float, value: float, step: float, suffix: str) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setDecimals(3)
        spin.setRange(low, high)
        spin.setValue(value)
        spin.setSingleStep(step)
        spin.setSuffix(suffix)
        return spin

    def limits(self) -> ManualControlLimits:
        return ManualControlLimits(
            max_linear_mps=self.linear_spin.value(),
            max_angular_radps=self.angular_spin.value(),
            deadzone=self.deadzone_spin.value(),
            response_exponent=2.0,
            max_linear_accel_mps2=self.lin_accel_spin.value(),
            max_angular_accel_radps2=self.ang_accel_spin.value(),
            min_turn_start_radps=self.turn_start_spin.value(),
        )

    def connect_vehicle(self) -> None:
        try:
            self.vehicle.connect(
                port=self.port_edit.text().strip(),
                addr=parse_node_addr(self.addr_edit.text().strip()),
                baudrate=int(self.baud_edit.text().strip()),
            )
            self.connect_btn.setEnabled(False)
            self.disconnect_btn.setEnabled(True)
            self.status_label.setText("Connected. Use Release E-Stop and Enable Driver only when the area is clear.")
            self.vehicle.soft_stop()
        except Exception as exc:  # pragma: no cover - hardware path
            self.status_label.setText(f"Connect failed: {exc}")

    def disconnect_vehicle(self) -> None:
        self.control_active = False
        self.active_box.setChecked(False)
        self.target_command = (0.0, 0.0)
        self.current_command = (0.0, 0.0)
        self.joystick.reset()
        self.keys_pressed.clear()
        self.vehicle.close()
        self.connect_btn.setEnabled(True)
        self.disconnect_btn.setEnabled(False)
        self.status_label.setText("Disconnected")

    def on_active_changed(self) -> None:
        self.control_active = self.active_box.isChecked()
        if not self.control_active:
            self.soft_stop()

    def soft_stop(self) -> None:
        self.target_command = (0.0, 0.0)
        self.joystick.reset()
        self.keys_pressed.clear()
        self._update_key_buttons()

    def emergency_stop(self) -> None:
        self.soft_stop()
        self.current_command = (0.0, 0.0)
        if self.vehicle.connected:
            self.vehicle.set_velocity(0.0, 0.0)
            self.vehicle.emergency_stop()
        self.status_label.setText("Emergency stop sent")

    def safe_call(self, method_name: str) -> None:
        if not self.vehicle.connected:
            self.status_label.setText("Not connected")
            return
        try:
            getattr(self.vehicle, method_name)()
            self.status_label.setText(f"{method_name} sent")
        except Exception as exc:  # pragma: no cover - hardware path
            self.status_label.setText(f"{method_name} failed: {exc}")

    def update_control(self) -> None:
        now = time.monotonic()
        dt = min(0.25, max(0.0, now - self.last_update))
        self.last_update = now

        forward_axis, turn_axis = self._input_axes()
        if self.control_active:
            self.target_command = manual_command_from_axes(forward_axis, turn_axis, self.limits())
        else:
            self.target_command = (0.0, 0.0)

        self.current_command = ramp_command(self.current_command, self.target_command, self.limits(), dt)
        self.target_label.setText(f"target: {self.target_command[0]:+.3f} m/s, {self.target_command[1]:+.3f} rad/s")
        self.current_label.setText(f"current: {self.current_command[0]:+.3f} m/s, {self.current_command[1]:+.3f} rad/s")

        if self.vehicle.connected:
            self.vehicle.set_velocity(*self.current_command)

    def _input_axes(self) -> tuple[float, float]:
        if self.keys_pressed:
            forward = 0.0
            turn = 0.0
            if Qt.Key_W in self.keys_pressed:
                forward += 1.0
            if Qt.Key_S in self.keys_pressed:
                forward -= 1.0
            if Qt.Key_A in self.keys_pressed:
                turn += 1.0
            if Qt.Key_D in self.keys_pressed:
                turn -= 1.0
            return forward, turn

        jx, jy = self.joystick.get_position()
        return jy, -jx

    def update_status(self) -> None:
        if not self.vehicle.connected:
            return
        voltage = self.vehicle.read_voltage()
        soc = self.vehicle.read_soc()
        fault = self.vehicle.read_fault()
        parts = ["Connected"]
        if voltage is not None:
            parts.append(f"Voltage {voltage:.2f} V")
        if soc is not None:
            parts.append(f"SOC {soc}%")
        if fault is not None:
            parts.append(f"Fault {fault}")
        self.status_label.setText(" | ".join(parts))

    def keyPressEvent(self, event):  # noqa: N802 - Qt override
        if event.isAutoRepeat():
            return
        key = event.key()
        if key in (Qt.Key_W, Qt.Key_A, Qt.Key_S, Qt.Key_D):
            self.keys_pressed.add(key)
            self._update_key_buttons()
        elif key in (Qt.Key_Space, Qt.Key_K):
            self.soft_stop()
            self.btn_stop.set_pressed(True)
        elif key == Qt.Key_Q:
            self.close()

    def keyReleaseEvent(self, event):  # noqa: N802 - Qt override
        if event.isAutoRepeat():
            return
        key = event.key()
        if key in (Qt.Key_W, Qt.Key_A, Qt.Key_S, Qt.Key_D):
            self.keys_pressed.discard(key)
            self._update_key_buttons()
        elif key in (Qt.Key_Space, Qt.Key_K):
            self.btn_stop.set_pressed(False)

    def _update_key_buttons(self) -> None:
        self.btn_w.set_pressed(Qt.Key_W in self.keys_pressed)
        self.btn_a.set_pressed(Qt.Key_A in self.keys_pressed)
        self.btn_s.set_pressed(Qt.Key_S in self.keys_pressed)
        self.btn_d.set_pressed(Qt.Key_D in self.keys_pressed)

    def closeEvent(self, event):  # noqa: N802 - Qt override
        self.control_timer.stop()
        self.status_timer.stop()
        self.vehicle.close()
        event.accept()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--addr", type=parse_node_addr, default=0x06)
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--linear", type=float, default=0.04)
    parser.add_argument("--angular", type=float, default=0.12)
    parser.add_argument("--deadzone", type=float, default=0.18)
    parser.add_argument("--linear-accel", type=float, default=0.08)
    parser.add_argument("--angular-accel", type=float, default=0.60)
    parser.add_argument("--turn-start", type=float, default=0.06)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow(args)
    window.show()
    raise SystemExit(app.exec_())


if __name__ == "__main__":
    main()
