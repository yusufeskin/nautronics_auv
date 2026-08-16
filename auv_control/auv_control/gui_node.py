#!/usr/bin/env python3
"""
ROS 2 + PyQt6 Kontrol & Telemetri Arayüzü

Kullanım:
1) Workspace'i derleyin:
   colcon build --packages-select auv_control --symlink-install
2) Ortamı source edin:
   source install/setup.bash
3) GUI node'u çalıştırın:
   ros2 run auv_control gui_node

Opsiyonel parametre örneği:
ros2 run auv_control gui_node --ros-args \
  -p telemetry_topics:="['/cmd_vel:geometry_msgs/msg/Twist','/imu/data:sensor_msgs/msg/Imu']"
"""

import sys
import threading
from functools import partial

import rclpy
from geometry_msgs.msg import Twist, Vector3
from PyQt6.QtCore import QObject, Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import Bool, Float32, Float64, Int32, String


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


class GuiSignals(QObject):
    telemetry_signal = pyqtSignal(str, str)
    log_signal = pyqtSignal(str)


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


class ControlTelemetryNode(Node):
    def __init__(self, signals: GuiSignals):
        super().__init__("control_telemetry_gui")
        self.signals = signals
        self._lock = threading.Lock()
        self._string_publishers = {}
        self._subscriptions = []

        self.declare_parameter(
            "telemetry_topics",
            [
                "/cmd_vel:geometry_msgs/msg/Twist",
                "/current_attitude:geometry_msgs/msg/Vector3",
                "/imu/data:sensor_msgs/msg/Imu",
            ],
        )
        self.declare_parameter("status_log_topic", "/gui/status_log")
        self.declare_parameter("command_topic", "/gui/command")
        self.declare_parameter("mode_topic", "/gui/mode")
        self.declare_parameter("estop_topic", "/gui/estop")

        self.command_topic = self.get_parameter("command_topic").value
        self.mode_topic = self.get_parameter("mode_topic").value
        self.estop_topic = self.get_parameter("estop_topic").value
        status_log_topic = self.get_parameter("status_log_topic").value
        telemetry_specs = self.get_parameter("telemetry_topics").value

        self.command_pub = self.create_publisher(String, self.command_topic, 10)
        self.mode_pub = self.create_publisher(String, self.mode_topic, 10)
        self.estop_pub = self.create_publisher(Bool, self.estop_topic, 10)

        self._subscriptions.append(
            self.create_subscription(String, status_log_topic, self._status_log_callback, 50)
        )

        for spec in telemetry_specs:
            self._create_telemetry_subscription(spec)

        self._emit_log("GUI node hazır.")

    def _emit_log(self, text: str):
        self.get_logger().info(text)
        self.signals.log_signal.emit(text)

    def _status_log_callback(self, msg: String):
        self.signals.log_signal.emit(msg.data)

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
        self._subscriptions.append(sub)
        self.signals.telemetry_signal.emit(topic_name, "bekleniyor...")
        self._emit_log(f"Telemetry aboneliği açıldı: {topic_name} ({type_name})")

    def _telemetry_callback(self, msg, topic_name: str):
        self.signals.telemetry_signal.emit(topic_name, format_msg(msg))

    def publish_mode(self, mode: str):
        with self._lock:
            self.mode_pub.publish(String(data=mode))
        self._emit_log(f"Mod komutu yayınlandı: {mode}")

    def publish_estop(self, state: bool):
        with self._lock:
            self.estop_pub.publish(Bool(data=state))
        self._emit_log(f"E-Stop yayınlandı: {state}")

    def publish_string(self, topic_name: str, payload: str):
        if not topic_name:
            self._emit_log("Hata: Topic boş olamaz.")
            return

        with self._lock:
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


class GuiWindow(QMainWindow):
    def __init__(self, node: ControlTelemetryNode, signals: GuiSignals):
        super().__init__()
        self.node = node
        self.signals = signals
        self.telemetry_fields = {}

        self.setWindowTitle("AUV Kontrol & Telemetri Arayüzü")
        self.resize(1200, 700)

        root = QWidget()
        self.setCentralWidget(root)
        layout = QGridLayout(root)
        layout.setColumnStretch(0, 3)
        layout.setColumnStretch(1, 2)

        telemetry_group = self._build_telemetry_group()
        control_group = self._build_control_group()

        layout.addWidget(telemetry_group, 0, 0)
        layout.addWidget(control_group, 0, 1)

        self.signals.telemetry_signal.connect(self.update_telemetry)
        self.signals.log_signal.connect(self.append_log)

    def _build_telemetry_group(self):
        group = QGroupBox("Telemetri / Topic İzleme")
        vbox = QVBoxLayout(group)

        self.telemetry_form = QFormLayout()
        self.telemetry_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        vbox.addLayout(self.telemetry_form)

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setPlaceholderText("ROS log akışı burada gösterilir...")
        vbox.addWidget(QLabel("Konsol / Log Akışı"))
        vbox.addWidget(self.log_view)
        return group

    def _build_control_group(self):
        group = QGroupBox("Kontrol ve Yayın Paneli")
        vbox = QVBoxLayout(group)

        row_1 = QHBoxLayout()
        arm_button = QPushButton("Arm")
        disarm_button = QPushButton("Disarm")
        arm_button.clicked.connect(lambda: self.node.publish_mode("ARM"))
        disarm_button.clicked.connect(lambda: self.node.publish_mode("DISARM"))
        row_1.addWidget(arm_button)
        row_1.addWidget(disarm_button)

        row_2 = QHBoxLayout()
        manual_button = QPushButton("Manuel")
        auto_button = QPushButton("Otonom")
        manual_button.clicked.connect(lambda: self.node.publish_mode("MANUAL"))
        auto_button.clicked.connect(lambda: self.node.publish_mode("AUTONOMOUS"))
        row_2.addWidget(manual_button)
        row_2.addWidget(auto_button)

        row_3 = QHBoxLayout()
        estop_button = QPushButton("Acil Durdurma")
        clear_estop_button = QPushButton("Acil Durdurma Kaldır")
        estop_button.clicked.connect(lambda: self.node.publish_estop(True))
        clear_estop_button.clicked.connect(lambda: self.node.publish_estop(False))
        row_3.addWidget(estop_button)
        row_3.addWidget(clear_estop_button)

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

        vbox.addLayout(row_1)
        vbox.addLayout(row_2)
        vbox.addLayout(row_3)
        vbox.addWidget(publish_box)
        vbox.addStretch()
        return group

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
