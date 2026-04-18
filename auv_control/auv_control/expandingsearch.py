#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import Twist

from auv_interfaces.action import YawAndScan 

class TimeBasedExpandingSquare(Node):
    def __init__(self):
        super().__init__('time_based_search_node')

        self.state = "FORWARD" 
        
        self.base_time = 1.0          
        self.current_target_time = 1.0 
        self.leg_count = 0            
        self.forward_speed = 0.50    
        self.turn_angle = -90.0        
        
        self.start_time = None        

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.turn_client = ActionClient(self, YawAndScan, '/yaw_and_scan')
        
        self.timer = self.create_timer(0.1, self.control_loop)
        
        self.get_logger().info("Genisleyen kutu aramasi basladi.")

    def control_loop(self):
        if self.state == "DONE":
            return

        if self.state == "FORWARD":
            now_sec = self.get_clock().now().nanoseconds / 1e9

            if self.start_time is None:
                self.start_time = now_sec
                self.get_logger().info(f"Ileri hareket: Hedef {self.current_target_time}s")

            elapsed_time = now_sec - self.start_time

            if elapsed_time < self.current_target_time:
                twist_msg = Twist()
                twist_msg.linear.x = self.forward_speed
                self.cmd_pub.publish(twist_msg)
            else:
                self.get_logger().info("Hedef sureye ulasildi, motorlar durduruluyor.")
                self.cmd_pub.publish(Twist())
                self.state = "SENDING_TURN"

        elif self.state == "SENDING_TURN":
            self.send_turn_goal()
            self.state = "WAITING_TURN"

        elif self.state == "WAITING_TURN":
            pass 

    def send_turn_goal(self):
        if not self.turn_client.wait_for_server(timeout_sec=3.0):
            self.get_logger().error("Action Server bulunamadi, gorev iptal ediliyor.")
            self.state = "DONE"
            return

        goal_msg = YawAndScan.Goal()
        goal_msg.target_angle_deg = self.turn_angle
        goal_msg.angular_speed = 0.5
        
        self._send_goal_future = self.turn_client.send_goal_async(goal_msg)
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Donus hedefi reddedildi.")
            self.state = "DONE"
            return
            
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        if result.success:
            self.leg_count += 1
            
            if self.leg_count % 2 == 0:
                self.current_target_time += self.base_time
                
            self.start_time = None 
            self.state = "FORWARD"
            self.get_logger().info("Donus tamamlandi. Yeni bacak basliyor.")
        else:
            self.get_logger().error("Donus basarisiz oldu.")
            self.state = "DONE"

def main(args=None):
    rclpy.init(args=args)
    node = TimeBasedExpandingSquare()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cmd_pub.publish(Twist())
        node.destroy_node()
        rclpy.try_shutdown()

if __name__ == '__main__':
    main()