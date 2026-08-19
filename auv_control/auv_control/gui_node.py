#!/usr/bin/env python3
"""
ROS 2 + PyQt6 AUV Saha Kontrol & Telemetri Arayüzü

Kullanım:
1) Workspace'i derleyin:
   colcon build --packages-select auv_control --symlink-install
2) Ortamı source edin:
   source install/setup.bash
3) GUI node'u çalıştırın:
   ros2 run auv_control gui_node

Bu arayüz gerçek araç servislerine/topic'lerine bağlanır:
  - /arm            (std_srvs/srv/SetBool)          -> Arm/Disarm
  - /change_mode    (auv_interfaces/srv/SetVehicleMode) -> Mod değiştirme
  - vehicle/state    (auv_interfaces/msg/VehicleStatus)  -> mod/armed/bağlantı
  - baro_data2       (std_msgs/msg/Float64)               -> derinlik (m)
  - current_attitude (geometry_msgs/msg/Vector3)          -> roll/pitch/yaw (deg)
  - /battery/status  (sensor_msgs/msg/BatteryState)       -> voltaj/akım/yüzde
  - target_depth     (std_msgs/msg/Float64)               -> hedef derinlik komutu
  - target_attitude  (geometry_msgs/msg/Vector3)          -> hedef yönelim komutu
  - kamera görüntüsü (sensor_msgs/msg/Image | CompressedImage)

Tüm topic/servis isimleri parametre olarak değiştirilebilir, örn:
ros2 run auv_control gui_node --ros-args -p camera_topic:=/my/image_raw
"""

import re
import sys
import time
from functools import partial

import numpy as np
import rclpy
from auv_interfaces.msg import VehicleStatus
from auv_interfaces.srv import GoToGpsTarget, SetVehicleMode
from geometry_msgs.msg import Point, Twist, Vector3
from PyQt6.QtCore import QObject, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import BatteryState, CompressedImage, Image, Imu
from std_msgs.msg import Bool, Float32, Float64, Int32, String
from std_srvs.srv import SetBool


SUPPORTED_MSG_TYPES = {
    "std_msgs/msg/String": String,
    "std_msgs/msg/Bool": Bool,
    "std_msgs/msg/Float32": Float32,
    "std_msgs/msg/Float64": Float64,
    "std_msgs/msg/Int32": Int32,
    "geometry_msgs/msg/Twist": Twist,
    "geometry_msgs/msg/Vector3": Vector3,
    "sensor_msgs/msg/Imu": Imu,
}

MODE_OPTIONS = ["STABILIZE", "ACRO", "ALT_HOLD", "AUTO", "MANUAL", "GUIDED"]
CAMERA_STALE_SEC = 2.0
CONNECTION_STALE_SEC = 3.0

# EKF/GPS origin - pixhawk_bridge2.py (POOL_ORIGIN_LAT/LON_DEG) ile aynı nokta olmalı.
POOL_ORIGIN_LAT_DEG = 39.85701944444445
POOL_ORIGIN_LON_DEG = 32.69128611111111

DMS_PAIR_RE = re.compile(r"(\d{1,3})[°:\s]+(\d{1,2})['’:\s]+([\d.]+)[\"”]?\s*([NSEWnsew])")


def parse_dms_coordinate(text: str):
    """'39°51'25.28\"N 32°41'28.64\"E' gibi metni (lat, lon) ondalık dereceye çevirir."""
    lat = lon = None
    for deg, minutes, seconds, hemi in DMS_PAIR_RE.findall(text):
        value = float(deg) + float(minutes) / 60.0 + float(seconds) / 3600.0
        hemi = hemi.upper()
        if hemi in ("S", "W"):
            value = -value
        if hemi in ("N", "S"):
            lat = value
        else:
            lon = value
    if lat is None or lon is None:
        raise ValueError(
            "Koordinat ayrıştırılamadı. Beklenen format: 39°51'25.28\"N 32°41'28.64\"E"
        )
    return lat, lon

COLOR_OK = "#1e8e3e"
COLOR_DANGER = "#d93025"
COLOR_WARN = "#e8a000"
COLOR_NEUTRAL = "#5f6368"
COLOR_ACCENT = "#1a73e8"

STYLESHEET = f"""
QWidget {{
    background-color: #f4f5f7;
    color: #1b1f24;
    font-size: 11pt;
}}
QMainWindow {{
    background-color: #eef0f3;
}}
QGroupBox {{
    background-color: #ffffff;
    border: 1px solid #d7dbe0;
    border-radius: 8px;
    margin-top: 14px;
    font-weight: 600;
    padding: 10px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    color: #202124;
}}
QPushButton {{
    background-color: {COLOR_ACCENT};
    color: white;
    border: none;
    border-radius: 6px;
    padding: 8px 14px;
    font-weight: 600;
}}
QPushButton:hover {{
    background-color: #1666c9;
}}
QPushButton:pressed {{
    background-color: #0f52a5;
}}
QPushButton#armButton {{
    background-color: {COLOR_OK};
}}
QPushButton#armButton:hover {{
    background-color: #167233;
}}
QPushButton#disarmButton {{
    background-color: {COLOR_NEUTRAL};
}}
QPushButton#estopButton {{
    background-color: {COLOR_DANGER};
    font-size: 14pt;
    padding: 14px;
}}
QPushButton#estopButton:hover {{
    background-color: #b0271c;
}}
QPushButton[active="true"] {{
    border: 2px solid #202124;
}}
QLineEdit, QTextEdit, QDoubleSpinBox, QComboBox {{
    background-color: white;
    border: 1px solid #c7cbd1;
    border-radius: 5px;
    padding: 4px;
}}
QLabel#badge {{
    border-radius: 6px;
    padding: 6px 12px;
    font-weight: 700;
    color: white;
    background-color: {COLOR_NEUTRAL};
}}
QLabel#bigReadout {{
    font-size: 22pt;
    font-weight: 700;
}}
QProgressBar {{
    border: 1px solid #c7cbd1;
    border-radius: 5px;
    text-align: center;
    background-color: white;
}}
QProgressBar::chunk {{
    background-color: {COLOR_OK};
    border-radius: 4px;
}}
"""


def format_msg(msg):
    if isinstance(msg, String):
        return msg.data
    if isinstance(msg, Bool):
        return str(msg.data)
    if isinstance(msg, (Float32, Float64, Int32)):
        return str(msg.data)
    if isinstance(msg, Vector3):
        return f"x={msg.x:.3f}, y={msg.y:.3f}, z={msg.z:.3f}"
    if isinstance(msg, Twist):
        return (
            f"lin=({msg.linear.x:.3f}, {msg.linear.y:.3f}, {msg.linear.z:.3f}) "
            f"ang=({msg.angular.x:.3f}, {msg.angular.y:.3f}, {msg.angular.z:.3f})"
        )
    if isinstance(msg, Imu):
        return (
            "orientation="
            f"({msg.orientation.x:.3f}, {msg.orientation.y:.3f}, "
            f"{msg.orientation.z:.3f}, {msg.orientation.w:.3f}) | "
            "gyro="
            f"({msg.angular_velocity.x:.3f}, {msg.angular_velocity.y:.3f}, {msg.angular_velocity.z:.3f}) | "
            "acc="
            f"({msg.linear_acceleration.x:.3f}, {msg.linear_acceleration.y:.3f}, {msg.linear_acceleration.z:.3f})"
        )
    return str(msg)


def ros_image_to_qimage(msg: Image):
    encoding = msg.encoding.lower()
    try:
        buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
        if encoding in ("bgr8", "rgb8"):
            frame = buf.reshape((msg.height, msg.step // 3, 3))[:, : msg.width, :]
            if encoding == "bgr8":
                frame = frame[:, :, ::-1]
            frame = np.ascontiguousarray(frame)
            qimg = QImage(frame.data, msg.width, msg.height, frame.strides[0], QImage.Format.Format_RGB888)
        elif encoding in ("mono8", "8uc1"):
            frame = buf.reshape((msg.height, msg.step))[:, : msg.width]
            frame = np.ascontiguousarray(frame)
            qimg = QImage(frame.data, msg.width, msg.height, frame.strides[0], QImage.Format.Format_Grayscale8)
        else:
            return None
        return qimg.copy()
    except Exception:
        return None


def ros_compressed_image_to_qimage(msg: CompressedImage):
    qimg = QImage()
    if qimg.loadFromData(bytes(msg.data)):
        return qimg
    return None


class GuiSignals(QObject):
    telemetry_signal = pyqtSignal(str, str)
    log_signal = pyqtSignal(str)
    image_signal = pyqtSignal(QImage)
    debug_image_signal = pyqtSignal(QImage)
    status_signal = pyqtSignal(str, bool, bool)
    depth_signal = pyqtSignal(float)
    attitude_signal = pyqtSignal(float, float, float)
    battery_signal = pyqtSignal(float, float, float)
    local_position_signal = pyqtSignal(float, float)


class ControlTelemetryNode(Node):
    def __init__(self, signals: GuiSignals):
        super().__init__("control_telemetry_gui")
        self.signals = signals
        self._string_publishers = {}
        self._telemetry_subs = []

        self.declare_parameter(
            "telemetry_topics",
            [
                "/cmd_vel:geometry_msgs/msg/Twist",
                "/imu/data:sensor_msgs/msg/Imu",
            ],
        )
        self.declare_parameter("status_log_topic", "/gui/status_log")
        self.declare_parameter("command_topic", "/gui/command")
        self.declare_parameter("camera_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("debug_camera_topic", "/pnp_debug_image/compressed")
        self.declare_parameter("vehicle_state_topic", "vehicle/state")
        self.declare_parameter("depth_topic", "baro_data2")
        self.declare_parameter("attitude_topic", "current_attitude")
        self.declare_parameter("local_position_topic", "vehicle/local_position")
        self.declare_parameter("battery_topic", "/battery/status")
        self.declare_parameter("arm_service", "/arm")
        self.declare_parameter("mode_service", "/change_mode")
        self.declare_parameter("gps_service", "/compute_and_go_gps")
        self.declare_parameter("origin_lat", POOL_ORIGIN_LAT_DEG)
        self.declare_parameter("origin_lon", POOL_ORIGIN_LON_DEG)
        self.declare_parameter("target_depth_topic", "target_depth")
        self.declare_parameter("target_attitude_topic", "target_attitude")

        self.command_topic = self.get_parameter("command_topic").value
        status_log_topic = self.get_parameter("status_log_topic").value
        camera_topic = self.get_parameter("camera_topic").value
        debug_camera_topic = self.get_parameter("debug_camera_topic").value
        vehicle_state_topic = self.get_parameter("vehicle_state_topic").value
        depth_topic = self.get_parameter("depth_topic").value
        attitude_topic = self.get_parameter("attitude_topic").value
        local_position_topic = self.get_parameter("local_position_topic").value
        battery_topic = self.get_parameter("battery_topic").value
        self.arm_service = self.get_parameter("arm_service").value
        self.mode_service = self.get_parameter("mode_service").value
        self.gps_service = self.get_parameter("gps_service").value
        self.origin_lat = self.get_parameter("origin_lat").value
        self.origin_lon = self.get_parameter("origin_lon").value
        target_depth_topic = self.get_parameter("target_depth_topic").value
        target_attitude_topic = self.get_parameter("target_attitude_topic").value
        telemetry_specs = self.get_parameter("telemetry_topics").value

        self.command_pub = self.create_publisher(String, self.command_topic, 10)
        self.target_depth_pub = self.create_publisher(Float64, target_depth_topic, 10)
        self.target_attitude_pub = self.create_publisher(Vector3, target_attitude_topic, 10)

        self.arm_client = self.create_client(SetBool, self.arm_service)
        self.mode_client = self.create_client(SetVehicleMode, self.mode_service)
        self.gps_client = self.create_client(GoToGpsTarget, self.gps_service)

        self._telemetry_subs.append(
            self.create_subscription(String, status_log_topic, self._status_log_callback, 50)
        )
        self._telemetry_subs.append(
            self.create_subscription(
                Image, camera_topic, self._camera_callback, qos_profile_sensor_data
            )
        )
        self._telemetry_subs.append(
            self.create_subscription(
                CompressedImage, debug_camera_topic, self._debug_camera_callback, qos_profile_sensor_data
            )
        )
        self._telemetry_subs.append(
            self.create_subscription(VehicleStatus, vehicle_state_topic, self._vehicle_state_callback, 10)
        )
        self._telemetry_subs.append(
            self.create_subscription(Float64, depth_topic, self._depth_callback, 10)
        )
        self._telemetry_subs.append(
            self.create_subscription(Vector3, attitude_topic, self._attitude_callback, 10)
        )
        self._telemetry_subs.append(
            self.create_subscription(Point, local_position_topic, self._local_position_callback, 10)
        )
        self._telemetry_subs.append(
            self.create_subscription(BatteryState, battery_topic, self._battery_callback, 10)
        )

        for spec in telemetry_specs:
            self._create_telemetry_subscription(spec)

        self._emit_log("GUI node hazır.")

    def _emit_log(self, text: str):
        self.get_logger().info(text)
        self.signals.log_signal.emit(text)

    def _status_log_callback(self, msg: String):
        self.signals.log_signal.emit(msg.data)

    def _camera_callback(self, msg: Image):
        qimg = ros_image_to_qimage(msg)
        if qimg is not None:
            self.signals.image_signal.emit(qimg)

    def _debug_camera_callback(self, msg: CompressedImage):
        qimg = ros_compressed_image_to_qimage(msg)
        if qimg is not None:
            self.signals.debug_image_signal.emit(qimg)

    def _vehicle_state_callback(self, msg: VehicleStatus):
        self.signals.status_signal.emit(msg.mode, msg.is_armed, msg.is_connected)

    def _depth_callback(self, msg: Float64):
        self.signals.depth_signal.emit(msg.data)

    def _attitude_callback(self, msg: Vector3):
        self.signals.attitude_signal.emit(msg.x, msg.y, msg.z)

    def _local_position_callback(self, msg: Point):
        self.signals.local_position_signal.emit(msg.x, msg.y)

    def _battery_callback(self, msg: BatteryState):
        self.signals.battery_signal.emit(msg.voltage, msg.current, msg.percentage)

    def _create_telemetry_subscription(self, spec: str):
        if ":" not in spec:
            self._emit_log(f"Geçersiz telemetry spec atlandı: {spec}")
            return
        topic_name, type_name = [x.strip() for x in spec.split(":", 1)]
        msg_type = SUPPORTED_MSG_TYPES.get(type_name)
        if msg_type is None:
            self._emit_log(f"Desteklenmeyen mesaj tipi atlandı: {type_name}")
            return

        callback = partial(self._telemetry_callback, topic_name=topic_name)
        sub = self.create_subscription(msg_type, topic_name, callback, 10)
        self._telemetry_subs.append(sub)
        self.signals.telemetry_signal.emit(topic_name, "bekleniyor...")
        self._emit_log(f"Telemetry aboneliği açıldı: {topic_name} ({type_name})")

    def _telemetry_callback(self, msg, topic_name: str):
        self.signals.telemetry_signal.emit(topic_name, format_msg(msg))

    def call_arm(self, arm: bool):
        if not self.arm_client.service_is_ready():
            self._emit_log(f"Hata: {self.arm_service} servisi hazır değil.")
            return
        request = SetBool.Request()
        request.data = arm
        future = self.arm_client.call_async(request)
        future.add_done_callback(partial(self._on_arm_response, arm=arm))

    def _on_arm_response(self, future, arm: bool):
        action = "Arm" if arm else "Disarm"
        try:
            response = future.result()
        except Exception as exc:
            self._emit_log(f"{action} hatası: {exc}")
            return
        if response.success:
            self._emit_log(f"{action} başarılı: {response.message}")
        else:
            self._emit_log(f"{action} başarısız: {response.message}")

    def call_mode(self, mode_name: str):
        if not self.mode_client.service_is_ready():
            self._emit_log(f"Hata: {self.mode_service} servisi hazır değil.")
            return
        request = SetVehicleMode.Request()
        request.mode_name = mode_name
        future = self.mode_client.call_async(request)
        future.add_done_callback(partial(self._on_mode_response, mode_name=mode_name))

    def _on_mode_response(self, future, mode_name: str):
        try:
            response = future.result()
        except Exception as exc:
            self._emit_log(f"Mod değiştirme hatası: {exc}")
            return
        if response.success:
            self._emit_log(f"Mod değiştirildi: {mode_name}")
        else:
            self._emit_log(f"Mod değiştirme başarısız: {mode_name} — {response.message}")

    def call_go_to_gps(self, lat: float, lon: float, depth: float):
        if not self.gps_client.service_is_ready():
            self._emit_log(f"Hata: {self.gps_service} servisi hazır değil.")
            return
        request = GoToGpsTarget.Request()
        request.baslangic_lat = self.origin_lat
        request.baslangic_lon = self.origin_lon
        request.hedef_lat = lat
        request.hedef_lon = lon
        request.target_depth = depth
        self._emit_log(f"GPS hedefi gönderiliyor: lat={lat:.7f}, lon={lon:.7f}, derinlik={depth:.2f} m")
        future = self.gps_client.call_async(request)
        future.add_done_callback(self._on_gps_response)

    def _on_gps_response(self, future):
        try:
            response = future.result()
        except Exception as exc:
            self._emit_log(f"GPS hedef hatası: {exc}")
            return
        if response.success:
            self._emit_log(
                f"GPS hedefi kabul edildi: X={response.calculated_x:.2f} m, Y={response.calculated_y:.2f} m"
            )
        else:
            self._emit_log("GPS hedefi reddedildi.")

    def publish_target_depth(self, depth: float):
        self.target_depth_pub.publish(Float64(data=depth))
        self._emit_log(f"Hedef derinlik gönderildi: {depth:.2f} m")

    def publish_target_attitude(self, roll: float, pitch: float, yaw: float):
        self.target_attitude_pub.publish(Vector3(x=roll, y=pitch, z=yaw))
        self._emit_log(f"Hedef yönelim gönderildi: roll={roll:.1f} pitch={pitch:.1f} yaw={yaw:.1f}")

    def publish_string(self, topic_name: str, payload: str):
        if not topic_name:
            self._emit_log("Hata: Topic boş olamaz.")
            return

        publisher = self._string_publishers.get(topic_name)
        if publisher is None:
            publisher = self.create_publisher(String, topic_name, 10)
            self._string_publishers[topic_name] = publisher
        publisher.publish(String(data=payload))
        self._emit_log(f"Mesaj yayınlandı [{topic_name}]: {payload}")


class RosSpinThread(QThread):
    def __init__(self, node: Node):
        super().__init__()
        self.node = node

    def run(self):
        while rclpy.ok() and not self.isInterruptionRequested():
            rclpy.spin_once(self.node, timeout_sec=0.1)

    def stop(self):
        self.requestInterruption()


class AspectImageLabel(QLabel):
    def __init__(self, placeholder: str, parent=None):
        super().__init__(parent)
        self._raw_pixmap = None
        self._placeholder = placeholder
        self.setMinimumSize(480, 360)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background-color: #14161a; color: #9aa0a6; border-radius: 6px;")
        self.show_placeholder(self._placeholder)

    def show_placeholder(self, text: str):
        self._raw_pixmap = None
        self.setText(text)

    def set_image(self, qimage: QImage):
        self._raw_pixmap = QPixmap.fromImage(qimage)
        self._rescale()

    def resizeEvent(self, event):
        self._rescale()
        super().resizeEvent(event)

    def _rescale(self):
        if self._raw_pixmap is None or self._raw_pixmap.isNull():
            return
        scaled = self._raw_pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setText("")
        self.setPixmap(scaled)


class GuiWindow(QMainWindow):
    def __init__(self, node: ControlTelemetryNode, signals: GuiSignals):
        super().__init__()
        self.node = node
        self.signals = signals
        self.telemetry_fields = {}
        self.mode_buttons = {}
        self._last_status_time = None
        self._last_camera_time = None
        self._last_debug_camera_time = None
        self._last_camera_image = None
        self._last_debug_image = None

        self.setWindowTitle("AUV Saha Kontrol & Telemetri Arayüzü")
        self.resize(1440, 860)
        self.setStyleSheet(STYLESHEET)

        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(10, 10, 10, 10)
        root_layout.addWidget(self._build_status_bar())

        splitter = QSplitter(Qt.Orientation.Vertical)
        top_splitter = QSplitter(Qt.Orientation.Horizontal)
        top_splitter.addWidget(self._build_camera_group())
        top_splitter.addWidget(self._build_right_panel())
        top_splitter.setStretchFactor(0, 3)
        top_splitter.setStretchFactor(1, 2)
        splitter.addWidget(top_splitter)
        splitter.addWidget(self._build_log_group())
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)
        root_layout.addWidget(splitter)

        self.signals.telemetry_signal.connect(self.update_telemetry)
        self.signals.log_signal.connect(self.append_log)
        self.signals.image_signal.connect(self.on_camera_image)
        self.signals.debug_image_signal.connect(self.on_debug_image)
        self.signals.status_signal.connect(self.update_status)
        self.signals.depth_signal.connect(self.update_depth)
        self.signals.attitude_signal.connect(self.update_attitude)
        self.signals.battery_signal.connect(self.update_battery)
        self.signals.local_position_signal.connect(self.update_local_position)

        self.watchdog_timer = QTimer(self)
        self.watchdog_timer.timeout.connect(self._check_watchdogs)
        self.watchdog_timer.start(1000)

    # ---------- top status bar ----------

    def _build_status_bar(self):
        frame = QFrame()
        frame.setStyleSheet("background-color: #ffffff; border: 1px solid #d7dbe0; border-radius: 8px;")
        layout = QHBoxLayout(frame)

        self.connection_badge = self._make_badge("BAĞLANTI YOK")
        self.armed_badge = self._make_badge("DISARMED")
        self.mode_badge = self._make_badge("MOD: -")

        title = QLabel("AUV Saha Kontrol Paneli")
        title.setStyleSheet("font-size: 14pt; font-weight: 700;")

        layout.addWidget(title)
        layout.addStretch()
        layout.addWidget(self.connection_badge)
        layout.addWidget(self.armed_badge)
        layout.addWidget(self.mode_badge)
        return frame

    def _make_badge(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("badge")
        return label

    def _style_badge(self, label: QLabel, text: str, color: str):
        label.setText(text)
        label.setStyleSheet(f"border-radius: 6px; padding: 6px 12px; font-weight: 700; color: white; background-color: {color};")

    # ---------- camera ----------

    def _build_camera_group(self):
        group = QGroupBox("Kamera")
        vbox = QVBoxLayout(group)

        source_row = QHBoxLayout()
        source_row.addWidget(QLabel("Kaynak:"))
        self.camera_source_combo = QComboBox()
        self.camera_source_combo.addItems(["Ham Görüntü", "Tespit / Debug Görüntüsü"])
        self.camera_source_combo.currentIndexChanged.connect(self._on_camera_source_changed)
        source_row.addWidget(self.camera_source_combo)
        source_row.addStretch()
        vbox.addLayout(source_row)

        self.camera_view = AspectImageLabel("Kamera görüntüsü bekleniyor...")
        vbox.addWidget(self.camera_view, stretch=1)
        return group

    def _on_camera_source_changed(self):
        self._refresh_camera_display()

    def _refresh_camera_display(self):
        if self.camera_source_combo.currentIndex() == 0:
            if self._last_camera_image is not None:
                self.camera_view.set_image(self._last_camera_image)
            else:
                self.camera_view.show_placeholder("Kamera görüntüsü bekleniyor...")
        else:
            if self._last_debug_image is not None:
                self.camera_view.set_image(self._last_debug_image)
            else:
                self.camera_view.show_placeholder("Debug görüntüsü bekleniyor...")

    def on_camera_image(self, qimage: QImage):
        self._last_camera_image = qimage
        self._last_camera_time = time.time()
        if self.camera_source_combo.currentIndex() == 0:
            self.camera_view.set_image(qimage)

    def on_debug_image(self, qimage: QImage):
        self._last_debug_image = qimage
        self._last_debug_camera_time = time.time()
        if self.camera_source_combo.currentIndex() == 1:
            self.camera_view.set_image(qimage)

    # ---------- right panel: telemetry + control ----------

    def _build_right_panel(self):
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._build_telemetry_group())
        layout.addWidget(self._build_battery_group())
        layout.addWidget(self._build_control_group())
        layout.addWidget(self._build_gps_group())
        layout.addWidget(self._build_advanced_group())
        layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(container)
        return scroll

    def _build_telemetry_group(self):
        group = QGroupBox("Telemetri")
        grid = QGridLayout(group)

        grid.addWidget(QLabel("Derinlik"), 0, 0)
        self.depth_label = QLabel("-- m")
        self.depth_label.setObjectName("bigReadout")
        grid.addWidget(self.depth_label, 0, 1)

        grid.addWidget(QLabel("Roll / Pitch / Yaw"), 1, 0)
        self.attitude_label = QLabel("-- / -- / --")
        grid.addWidget(self.attitude_label, 1, 1)

        grid.addWidget(QLabel("Local Konum (X / Y)"), 2, 0)
        self.local_position_label = QLabel("-- / -- m")
        grid.addWidget(self.local_position_label, 2, 1)
        return group

    def _build_battery_group(self):
        group = QGroupBox("Batarya")
        grid = QGridLayout(group)

        grid.addWidget(QLabel("Voltaj / Akım"), 0, 0)
        self.battery_label = QLabel("-- V / -- A")
        grid.addWidget(self.battery_label, 0, 1)

        self.battery_bar = QProgressBar()
        self.battery_bar.setRange(0, 100)
        self.battery_bar.setValue(0)
        grid.addWidget(self.battery_bar, 1, 0, 1, 2)
        return group

    def _build_control_group(self):
        group = QGroupBox("Kontrol")
        vbox = QVBoxLayout(group)

        arm_row = QHBoxLayout()
        arm_button = QPushButton("ARM")
        arm_button.setObjectName("armButton")
        disarm_button = QPushButton("DISARM")
        disarm_button.setObjectName("disarmButton")
        arm_button.clicked.connect(lambda: self.node.call_arm(True))
        disarm_button.clicked.connect(lambda: self.node.call_arm(False))
        arm_row.addWidget(arm_button)
        arm_row.addWidget(disarm_button)
        vbox.addLayout(arm_row)

        mode_grid = QGridLayout()
        for i, mode_name in enumerate(MODE_OPTIONS):
            button = QPushButton(mode_name)
            button.clicked.connect(partial(self.node.call_mode, mode_name))
            self.mode_buttons[mode_name] = button
            mode_grid.addWidget(button, i // 3, i % 3)
        vbox.addLayout(mode_grid)

        estop_button = QPushButton("ACİL DURDURMA (DISARM)")
        estop_button.setObjectName("estopButton")
        estop_button.clicked.connect(lambda: self.node.call_arm(False))
        vbox.addWidget(estop_button)

        target_form = QFormLayout()
        self.depth_spin = QDoubleSpinBox()
        self.depth_spin.setRange(-5.0, 50.0)
        self.depth_spin.setSingleStep(0.1)
        self.depth_spin.setSuffix(" m")
        depth_send = QPushButton("Hedef Derinlik Gönder")
        depth_send.clicked.connect(lambda: self.node.publish_target_depth(self.depth_spin.value()))
        target_form.addRow("Hedef Derinlik", self.depth_spin)
        target_form.addRow(depth_send)

        self.roll_spin = QDoubleSpinBox()
        self.pitch_spin = QDoubleSpinBox()
        self.yaw_spin = QDoubleSpinBox()
        for spin in (self.roll_spin, self.pitch_spin, self.yaw_spin):
            spin.setRange(-180.0, 180.0)
            spin.setSingleStep(1.0)
            spin.setSuffix(" °")
        attitude_row = QHBoxLayout()
        attitude_row.addWidget(self.roll_spin)
        attitude_row.addWidget(self.pitch_spin)
        attitude_row.addWidget(self.yaw_spin)
        attitude_send = QPushButton("Hedef Yönelim Gönder")
        attitude_send.clicked.connect(
            lambda: self.node.publish_target_attitude(
                self.roll_spin.value(), self.pitch_spin.value(), self.yaw_spin.value()
            )
        )
        target_form.addRow("Hedef Roll/Pitch/Yaw", attitude_row)
        target_form.addRow(attitude_send)
        vbox.addLayout(target_form)

        return group

    def _build_gps_group(self):
        group = QGroupBox("GPS Git")
        vbox = QVBoxLayout(group)

        form = QFormLayout()
        self.gps_coord_input = QLineEdit()
        self.gps_coord_input.setPlaceholderText("39°51'25.28\"N 32°41'28.64\"E")
        form.addRow("Koordinat", self.gps_coord_input)

        self.gps_depth_spin = QDoubleSpinBox()
        self.gps_depth_spin.setRange(-5.0, 50.0)
        self.gps_depth_spin.setSingleStep(0.1)
        self.gps_depth_spin.setValue(1.5)
        self.gps_depth_spin.setSuffix(" m")
        form.addRow("Hedef Derinlik", self.gps_depth_spin)
        vbox.addLayout(form)

        self.gps_preview_label = QLabel("—")
        self.gps_preview_label.setStyleSheet("color: #5f6368;")
        vbox.addWidget(self.gps_preview_label)

        gps_go_button = QPushButton("Git")
        gps_go_button.clicked.connect(self.on_gps_go_clicked)
        vbox.addWidget(gps_go_button)

        return group

    def on_gps_go_clicked(self):
        text = self.gps_coord_input.text().strip()
        if not text:
            QMessageBox.warning(self, "Koordinat Boş", "Önce bir GPS koordinatı yapıştırın.")
            return
        try:
            lat, lon = parse_dms_coordinate(text)
        except ValueError as exc:
            self.gps_preview_label.setText(str(exc))
            QMessageBox.warning(self, "Geçersiz Koordinat", str(exc))
            return
        self.gps_preview_label.setText(f"lat={lat:.7f}, lon={lon:.7f}")
        self.node.call_go_to_gps(lat, lon, self.gps_depth_spin.value())

    def _build_advanced_group(self):
        group = QGroupBox("Gelişmiş / Hata Ayıklama")
        group.setCheckable(True)
        group.setChecked(False)
        outer = QVBoxLayout(group)

        content = QWidget()
        content.setVisible(False)
        group.toggled.connect(content.setVisible)
        vbox = QVBoxLayout(content)
        vbox.setContentsMargins(0, 0, 0, 0)

        self.telemetry_form = QFormLayout()
        self.telemetry_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        vbox.addLayout(self.telemetry_form)

        publish_box = QGroupBox("Serbest String Yayını")
        publish_layout = QFormLayout(publish_box)
        self.topic_input = QLineEdit(self.node.command_topic)
        self.payload_input = QLineEdit()
        self.payload_input.setPlaceholderText("Gönderilecek mesaj metni")
        send_button = QPushButton("Gönder")
        send_button.clicked.connect(self.on_send_clicked)
        publish_layout.addRow("Topic", self.topic_input)
        publish_layout.addRow("Mesaj", self.payload_input)
        publish_layout.addRow(send_button)
        vbox.addWidget(publish_box)

        outer.addWidget(content)
        return group

    def _build_log_group(self):
        group = QGroupBox("Konsol / Log Akışı")
        vbox = QVBoxLayout(group)
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setPlaceholderText("ROS log akışı burada gösterilir...")
        vbox.addWidget(self.log_view)
        return group

    # ---------- slots ----------

    def on_send_clicked(self):
        topic = self.topic_input.text().strip()
        payload = self.payload_input.text()
        if not topic:
            QMessageBox.warning(self, "Geçersiz Topic", "Topic alanı boş bırakılamaz.")
            return
        self.node.publish_string(topic, payload)

    def update_telemetry(self, topic_name: str, value: str):
        field = self.telemetry_fields.get(topic_name)
        if field is None:
            field = QLineEdit()
            field.setReadOnly(True)
            self.telemetry_fields[topic_name] = field
            self.telemetry_form.addRow(topic_name, field)
        field.setText(value)

    def append_log(self, text: str):
        self.log_view.append(text)

    def update_status(self, mode: str, armed: bool, connected: bool):
        self._last_status_time = time.time()
        self._style_badge(self.mode_badge, f"MOD: {mode}", COLOR_ACCENT)
        self._style_badge(self.armed_badge, "ARMED" if armed else "DISARMED", COLOR_OK if armed else COLOR_NEUTRAL)
        self._style_badge(self.connection_badge, "BAĞLI" if connected else "BAĞLANTI YOK", COLOR_OK if connected else COLOR_DANGER)

        for mode_name, button in self.mode_buttons.items():
            button.setProperty("active", mode_name == mode)
            button.style().unpolish(button)
            button.style().polish(button)

    def update_depth(self, depth: float):
        self.depth_label.setText(f"{depth:.2f} m")

    def update_attitude(self, roll: float, pitch: float, yaw: float):
        self.attitude_label.setText(f"{roll:.1f}° / {pitch:.1f}° / {yaw:.1f}°")

    def update_local_position(self, x: float, y: float):
        self.local_position_label.setText(f"{x:.2f} / {y:.2f} m")

    def update_battery(self, voltage: float, current: float, percentage: float):
        self.battery_label.setText(f"{voltage:.2f} V / {current:.2f} A")
        pct = max(0, min(100, round(percentage * 100)))
        self.battery_bar.setValue(pct)
        if pct <= 20:
            color = COLOR_DANGER
        elif pct <= 45:
            color = COLOR_WARN
        else:
            color = COLOR_OK
        self.battery_bar.setStyleSheet(
            "QProgressBar { border: 1px solid #c7cbd1; border-radius: 5px; text-align: center; background-color: white; }"
            f"QProgressBar::chunk {{ background-color: {color}; border-radius: 4px; }}"
        )

    def _check_watchdogs(self):
        now = time.time()
        if self._last_status_time is not None and (now - self._last_status_time) > CONNECTION_STALE_SEC:
            self._style_badge(self.connection_badge, "BAĞLANTI YOK", COLOR_DANGER)

        if self._last_camera_time is not None and (now - self._last_camera_time) > CAMERA_STALE_SEC:
            if self.camera_source_combo.currentIndex() == 0:
                self.camera_view.show_placeholder("Görüntü kesildi...")
        if self._last_debug_camera_time is not None and (now - self._last_debug_camera_time) > CAMERA_STALE_SEC:
            if self.camera_source_combo.currentIndex() == 1:
                self.camera_view.show_placeholder("Görüntü kesildi...")


def main(args=None):
    rclpy.init(args=args)
    app = QApplication(sys.argv)

    signals = GuiSignals()
    node = ControlTelemetryNode(signals)
    spin_thread = RosSpinThread(node)
    spin_thread.start()

    window = GuiWindow(node, signals)
    window.show()

    exit_code = 0
    try:
        exit_code = app.exec()
    finally:
        spin_thread.stop()
        spin_thread.wait(2000)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
