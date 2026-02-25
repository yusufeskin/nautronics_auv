#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Empty, Float64
from std_srvs.srv import Trigger
import time

class TorpedoService(Node):
    def __init__(self):
        super().__init__('torpedo_service_node')
        self.detach_pub = self.create_publisher(Empty, '/torpedo/fire', 10)
        self.thrust_pub = self.create_publisher(Float64, '/torpedo/cmd_thrust', 10)
        self.srv = self.create_service(Trigger, '/torpedo/fire_service', self.fire_callback)
        self.get_logger().info('ready to trigger...')
    def fire_callback(self, request, response):
        try:
            self.detach_pub.publish(Empty())
            time.sleep(0.05)
            thrust_msg = Float64()
            thrust_msg.data = 80.0
            self.thrust_pub.publish(thrust_msg)
            self.get_logger().info('impulse!') 
            time.sleep(0.01)            
            thrust_msg.data = 0.0
            self.thrust_pub.publish(thrust_msg)
            response.success = True
            response.message = "Succeeded, launched"
        except Exception as e:
            response.success = False
            response.message = f"error: {str(e)}"
            self.get_logger().error(response.message)
            
        return response

def main(args=None):
    rclpy.init(args=args)
    node = TorpedoService()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass 
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()