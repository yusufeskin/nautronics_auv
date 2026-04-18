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
from std_msgs.msg import Float64MultiArray, MultiArrayDimension, UInt16MultiArray

class PixhawkBridge(Node):
    def __init__(self):
        super().__init__('pixhawk_bridge_node')
        self.master = mavutil.mavlink_connection('udpin:0.0.0.0:14550', baud=57600)
        self.master.wait_heartbeat()
        self.get_logger().info("connected to Pixhawk!")


        self.status_publisher = self.create_publisher(
            VehicleStatus,
            'vehicle/state',
            10
        )

        self.baro_publisher = self.create_publisher(
            Float64MultiArray,
            'baro_data',
            10
        )

        self.pwm_module = PwmHandler(self.master, self.get_logger())
        self.mode_module = ModeHandler(self.master, self.get_logger())
        self.telemetry_module = TelemetryHandler(self, self.master, self.status_publisher)
        self.baro_module = BaroHandler(self, self.master, self.baro_publisher)




        self.mode_change_service = self.create_service(
            SetVehicleMode, 
            '/change_mode', 
            self.mode_module.change_mode_callback
        )
        
        self.pwm_router = self.create_subscription(
            UInt16MultiArray, 
            'pwm_router', 
            self.pwm_callback, 
            10
        )

        self.telemetry_timer = self.create_timer(0.1, self.telemetry_module.read_and_publish)
        self.baro_timer = self.create_timer(0.1, self.baro_module.read_and_publish)


    def pwm_callback(self, msg):
        self.pwm_module.send_pwm(msg.data)


def main(args=None):
    rclpy.init(args=args)
    node = PixhawkBridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()