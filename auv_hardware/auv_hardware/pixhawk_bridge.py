#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from pymavlink import mavutil
from .pwm_handler import PwmHandler
from .mode_handler import ModeHandler
from .telemetry_handler import TelemetryHandler
from .baro_handler import BaroHandler
from auv_interfaces.srv import SetVehicleMode
from auv_interfaces.msg import VehicleStatus
from std_msgs.msg import Float64, UInt16MultiArray
from .set_depth_handler import SetDepthHandler
from .set_attitude_handler import SetAttitudeHandler
from geometry_msgs.msg import Vector3

class PixhawkBridge(Node):
    def __init__(self):
        super().__init__('pixhawk_bridge_node')
        self.master = mavutil.mavlink_connection('udpin:0.0.0.0:14550', baud=57600)
        self.master.wait_heartbeat()
        self.get_logger().info("Pixhawk'a bağlanıldı!")

        self.status_publisher = self.create_publisher(VehicleStatus, 'vehicle/state', 10)
        self.baro_publisher   = self.create_publisher(Float64, 'baro_data', 10)

        self.pwm_module       = PwmHandler(self.master, self.get_logger())
        self.mode_module      = ModeHandler(self.master, self.get_logger())
        self.telemetry_module = TelemetryHandler(self, self.status_publisher)
        self.baro_module      = BaroHandler(self, self.baro_publisher)
        self.set_depth_module = SetDepthHandler(self.master, self.get_logger())
        self.set_attitude_module   = SetAttitudeHandler(self.master, self.get_logger())

        self.msg_handlers = {
            'HEARTBEAT': [self.telemetry_module.handle_message],
            #'VFR_HUD':   [self.baro_module.handle_message], # useless, will be adjusted
            'GLOBAL_POSITION_INT':   [self.baro_module.handle_message]

        }

        self.mode_change_service = self.create_service(
            SetVehicleMode, '/change_mode', self.mode_module.change_mode_callback
        )
        self.pwm_subscription = self.create_subscription(
            UInt16MultiArray, 'pwm_router', self.pwm_callback, 10
        )

        self.set_depth_subscription = self.create_subscription(
            Float64, 'target_depth', self.set_depth_callback, 10
        )

        self.set_attitude_subscription = self.create_subscription(
            Vector3, 'target_attitude', self.set_attitude_callback, 10
        )

        self.mavlink_timer = self.create_timer(0.02, self.dispatch_mavlink)  # 50 Hz

    def dispatch_mavlink(self):
        while True:
            msg = self.master.recv_match(blocking=False)
            if msg is None:
                break
            msg_type = msg.get_type()
            if msg_type in self.msg_handlers:
                for handler in self.msg_handlers[msg_type]:
                    handler(msg)

    def pwm_callback(self, msg):
        self.pwm_module.send_pwm(msg.data)

    def set_depth_callback(self, msg):
        self.set_depth_module.set_target_depth(msg.data)

    def set_attitude_callback(self, msg):
        roll = msg.x
        pitch = msg.y
        yaw = msg.z
        self.set_attitude_module.set_target_attitude(roll, pitch, yaw)

def main(args=None):
    rclpy.init(args=args)
    node = PixhawkBridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()