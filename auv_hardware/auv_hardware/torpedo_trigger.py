#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Empty, Float64
from std_srvs.srv import Trigger
import time

class TorpedoService(Node):
    def __init__(self):
        super().__init__('torpedo_service_node')
        self.fired_count = 0 
        self.detach_pub_1 = self.create_publisher(Empty, '/torpedo/fire1', 10)
        self.thrust_pub_1 = self.create_publisher(Float64, '/torpedo/cmd_thrust1', 10)
        self.detach_pub_2 = self.create_publisher(Empty, '/torpedo/fire2', 10)
        self.thrust_pub_2 = self.create_publisher(Float64, '/torpedo/cmd_thrust2', 10)
        self.srv = self.create_service(Trigger, '/torpedo/fire_service', self.fire_callback)
        self.get_logger().info('ready to launch 2 torpedo...')

    def fire_callback(self, request, response):
        if self.fired_count == 0:
            self.get_logger().info('first torpedo...')
            self.execute_launch(self.detach_pub_1, self.thrust_pub_1)
            self.fired_count = 1
            response.success = True
            response.message = "1st torpedo launched."
            
        elif self.fired_count == 1:
            self.get_logger().info('second torpedo..')
            self.execute_launch(self.detach_pub_2, self.thrust_pub_2)
            self.fired_count = 2
            response.success = True
            response.message = "2nd torpedo launched."
            
        else:
            self.get_logger().warn('no more ammo!')
            response.success = False
            response.message = "there is no more torpedo."
            
        return response

    def execute_launch(self, detach_pub, thrust_pub):
        try:
            detach_pub.publish(Empty())
            time.sleep(0.1)
            thrust_msg = Float64()
            thrust_msg.data = 4.0
            thrust_pub.publish(thrust_msg)
            time.sleep(0.2)            
            thrust_msg.data = 0.0
            thrust_pub.publish(thrust_msg)
            
        except Exception as e:
            self.get_logger().error(f"err: {str(e)}")

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