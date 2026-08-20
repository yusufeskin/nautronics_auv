#!/usr/bin/env python3
import time

import rclpy
from rclpy.node import Node
from pymavlink import mavutil
from .pwm_handler import PwmHandler
from .mode_handler import ModeHandler
from .telemetry_handler import TelemetryHandler
from .baro_handler import BaroHandler
from std_srvs.srv import SetBool
from auv_interfaces.srv import SetVehicleMode
from auv_interfaces.msg import VehicleStatus
from std_msgs.msg import Float64, UInt16MultiArray
from .set_depth_handler import SetDepthHandler
from .set_attitude_handler import SetAttitudeHandler
from .attitude_handler import AttitudeHandler
from .baro_handler2 import BaroHandler2
from geometry_msgs.msg import Vector3
from std_msgs.msg import UInt16
from .led_handler import LedHandler
from .dvl_odom_handler import DvlOdomHandler
from .set_position_target_handler import SetPositionTargetHandler
from .local_position_handler import LocalPositionHandler
from geometry_msgs.msg import Point

POOL_ORIGIN_LAT_DEG = 40.7580083333   # 40deg45'28.83"N
POOL_ORIGIN_LON_DEG = 29.9014166667   # 29deg54'05.10"E
POOL_ORIGIN_ALT_M = 0.0


SET_ORIGIN_RETRIES = 3
SET_ORIGIN_RETRY_DELAY_S = 0.5


class PixhawkBridge(Node):
    def __init__(self):
        super().__init__('pixhawk_bridge_node')

        self.master = mavutil.mavlink_connection('udp:127.0.0.1:14551', baud=57600)
        self.master.wait_heartbeat()
        self.get_logger().info("Pixhawk'a bağlanıldı!")

        self.send_gps_global_origin()

        self.request_message_interval(mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE, 50)
        self.request_message_interval(mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT, 10)
        self.request_message_interval(mavutil.mavlink.MAVLINK_MSG_ID_SCALED_PRESSURE2, 10)
        self.request_message_interval(mavutil.mavlink.MAVLINK_MSG_ID_LOCAL_POSITION_NED, 10)

        self.status_publisher   = self.create_publisher(VehicleStatus, 'vehicle/state', 10)
        self.baro_publisher     = self.create_publisher(Float64, 'baro_data', 10)
        self.baro_publisher2    = self.create_publisher(Float64, 'baro_data2', 10)
        self.attitude_publisher = self.create_publisher(Vector3, 'current_attitude', 10)
        self.local_position_publisher = self.create_publisher(Point, 'vehicle/local_position', 10)

        self.pwm_module            = PwmHandler(self.master, self.get_logger())
        self.mode_module           = ModeHandler(self.master, self.get_logger())
        self.telemetry_module      = TelemetryHandler(self, self.status_publisher)
        self.baro_module           = BaroHandler(self, self.baro_publisher)
        self.baro_module2          = BaroHandler2(self, self.baro_publisher2) 
        self.set_depth_module      = SetDepthHandler(self.master, self.get_logger())
        self.set_attitude_module   = SetAttitudeHandler(self.master, self.get_logger())
        self.attitude_module       = AttitudeHandler(self, self.attitude_publisher)
        self.led_module            = LedHandler(self.master, self.get_logger())
        self.dvl_module            = DvlOdomHandler(self, self.master, self.get_logger())
        self.set_position_module   = SetPositionTargetHandler(self.master, self.get_logger())
        self.local_position_module = LocalPositionHandler(self, self.local_position_publisher)
        
        self.msg_handlers = {
            'HEARTBEAT': [self.telemetry_module.handle_message],
            #'VFR_HUD':   [self.baro_module.handle_message], # useless, will be adjusted
            'GLOBAL_POSITION_INT':   [self.baro_module.handle_message],
            'ATTITUDE': [self.attitude_module.handle_message, self.dvl_module.handle_attitude],
            'SCALED_PRESSURE2': [self.baro_module2.handle_message],
            'LOCAL_POSITION_NED': [self.local_position_module.handle_message]
        }

        self.mode_change_service = self.create_service(
            SetVehicleMode, '/change_mode', self.mode_module.change_mode_callback
        )
        self.arm_service = self.create_service(
            SetBool, '/arm', self.mode_module.arm_callback
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

        self.led_subscription = self.create_subscription(
            UInt16, 'led_control', self.led_callback, 10
        )

        self.set_position_subscription = self.create_subscription(
            Point, 'target_position_local', self.set_position_callback, 10
        )

        self.mavlink_timer = self.create_timer(0.02, self.dispatch_mavlink)  # 50 Hz

    def send_gps_global_origin(self):
        lat_e7 = int(round(POOL_ORIGIN_LAT_DEG * 1e7))
        lon_e7 = int(round(POOL_ORIGIN_LON_DEG * 1e7))
        alt_mm = int(round(POOL_ORIGIN_ALT_M * 1000.0))

        for _ in range(SET_ORIGIN_RETRIES):
            self.master.mav.set_gps_global_origin_send(
                self.master.target_system,
                lat_e7,
                lon_e7,
                alt_mm,
            )
            time.sleep(SET_ORIGIN_RETRY_DELAY_S)

        self.get_logger().info(
            f"EKF origin gonderildi: {POOL_ORIGIN_LAT_DEG:.7f}, {POOL_ORIGIN_LON_DEG:.7f}"
        )

    def request_message_interval(self, message_id, frequency_hz):
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
            0,
            message_id,
            1e6 / frequency_hz,
            0, 0, 0, 0, 0
        )

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

    def led_callback(self, msg):
        self.led_module.set_led_pwm(msg.data)

    def set_position_callback(self, msg):
        # msg.x is North, msg.y is East, msg.z is Down (positive depth)
        self.set_position_module.set_target_position_local(msg.x, msg.y, msg.z)


def main(args=None):
    rclpy.init(args=args)
    node = PixhawkBridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
