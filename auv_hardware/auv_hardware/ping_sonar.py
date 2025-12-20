#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Range
from brping import Ping1D
import time
import sys

class PingUARTNode(Node):
    def __init__(self):
        super().__init__('ping_sonar_node')
        self.USE_SERIAL = True
        self.SERIAL_PORT = '/dev/ttyTHS1'
        self.BAUDRATE = 115200
        self.publisher_ = self.create_publisher(Range, 'sonar', 10)
        
        self.ping = Ping1D()
        self._initialize_ping()

        self.timer_period = 0.1  # seconds
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        self.get_logger().info(f'Ping Sonar UART Node Started on {self.SERIAL_PORT} @ {self.BAUDRATE} bps')

    def _initialize_ping(self):        
        connected = False
        try:
            self.ping.connect_serial(self.SERIAL_PORT, self.BAUDRATE)
            connected = True
        except Exception as e:
            self.get_logger().fatal(f"Serial connection error ({self.SERIAL_PORT}): {e}")
        
        if not connected:
            self.get_logger().fatal("Could not establish connection to Ping device. Exiting.")
            sys.exit(1)
                    
        self.get_logger().info("Ping device successfully initialized.")
        
        # Static parameters
        self.range_msg = Range()
        self.range_msg.radiation_type = Range.ULTRASOUND
        self.range_msg.field_of_view = 0.052 
        self.range_msg.min_range = 0.2
        self.range_msg.max_range = 50.0 
        self.range_msg.header.frame_id = 'ping_sonar_link' 

    def timer_callback(self):
        data = self.ping.get_distance()
        
        if data:
            distance_mm = data["distance"]
            distance_m = distance_mm / 1000.0
            confidence = data["confidence"]
            if confidence > 10: # Confidence threshold
                self.range_msg.header.stamp = self.get_clock().now().to_msg()
                self.range_msg.range = distance_m
                self.publisher_.publish(self.range_msg)
            self.get_logger().debug(f"Distance: {distance_m:.3f} m, Confidence: {confidence}%")
            print(f"Distance: {distance_m:.3f} m, Confidence: {confidence}%")
        else:
            self.get_logger().warn("Failed to retrieve distance data.")

def main(args=None):
    rclpy.init(args=args)
    ping_node = PingUARTNode()
    try:
        rclpy.spin(ping_node)
    except KeyboardInterrupt:
        pass
    finally:
        ping_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()