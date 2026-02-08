#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import UInt16MultiArray
import numpy as np

class ThrusterMixer(Node):
    def __init__(self):
        super().__init__('thruster_mixer')
        
        self.subscription = self.create_subscription(
            Twist,
            'cmd_vel',
            self.cmd_vel_callback,
            10)
    
        self.pwm_publisher = self.create_publisher(
            UInt16MultiArray,
            'pwm_router',
            10)

        self.pwm_min = 1100
        self.pwm_max = 1900
        self.pwm_neutral = 1500
        
        self.mixing_matrix = np.array([
            [ 0.0,      0.0,    0.0,     0.0,    0.0,     0.0 ],  
            [ 0.0,      0.0,    0.0,     0.0,    0.0,     0.0 ],  
            [ 0.0,      0.0,    1.0,     0.0,    0.0,     0.0 ],  
            [ 0.0,      0.0,    0.0,     0.0,    0.0,     1.0 ], 
            
            [ 1.0,      0.0,    0.0,     0.0,    0.0,     0.0 ],  
            [ 0.0,      0.0,    0.0,     0.0,    0.0,     0.0 ],  
            [ 0.0,      0.0,    0.0,     0.0,    0.0,     0.0 ],  
            [ 0.0,      0.0,    0.0,     0.0,    0.0,     0.0 ]   
        ])
        
        self.get_logger().info('Thruster Mixer Node Started (3-Thruster Mode) - Logging Enabled')

    def cmd_vel_callback(self, msg):
        control_vector = np.array([
            msg.linear.x,  
            msg.linear.y,  
            msg.linear.z, 
            msg.angular.x, 
            msg.angular.y, 
            msg.angular.z  
        ])
        
        motor_inputs = np.dot(self.mixing_matrix, control_vector)
        
        max_val = np.max(np.abs(motor_inputs))
        if max_val > 1.0:
            motor_inputs = motor_inputs / max_val
            
        pwm_values = []
        for input_val in motor_inputs:
            pwm = int(self.pwm_neutral + (input_val * (self.pwm_max - self.pwm_neutral)))
            pwm = max(self.pwm_min, min(self.pwm_max, pwm))
            pwm_values.append(pwm)
            
        array_msg = UInt16MultiArray()
        array_msg.data = pwm_values
        self.pwm_publisher.publish(array_msg)

def main(args=None):
    rclpy.init(args=args)
    node = ThrusterMixer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()