#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import UInt16MultiArray
import numpy as np

class ThrusterMixer(Node):
    def __init__(self):
        super().__init__('thruster_mixer')
        
        # Subscribe to cmd_vel (Twist)
        # Assuming cmd_vel is in the vehicle's body frame (Forward-Left-Up)
        self.subscription = self.create_subscription(
            Twist,
            'cmd_vel',
            self.cmd_vel_callback,
            10)
        
        # Publisher for PWM values (to be picked up by pwm_router)
        self.pwm_publisher = self.create_publisher(
            UInt16MultiArray,
            'pwm_router',
            10)

        # Thruster Configuration Parameters
        # PWM Range
        self.pwm_min = 1100
        self.pwm_max = 1900
        self.pwm_neutral = 1500
        
        # Mixing Matrix
        # Maps [surge, sway, heave, roll, pitch, yaw] -> [T1, T2, T3, T4, T5, T6, T7, T8]
        # Based on BlueROV2 Heavy configuration (8 Thrusters)
        # Adjust signs based on actual propeller directions (CW/CCW) and wiring
        
        # Approximate mix for standard vectoral 8-thruster layout
        # T1-T4: Horizontal (Surge, Sway, Yaw) - Vectoral at 45 deg
        # T5-T8: Vertical (Heave, Roll, Pitch)
        
        # Input Vector: [Surge (x), Sway (y), Heave (z), Roll (x), Pitch (y), Yaw (z)]
        # Output Vector: [T1, T2, T3, T4, T5, T6, T7, T8]
        
        # Weights (can be tuned)
        self.mixing_matrix = np.array([
            # x(surge) y(sway) z(heave) r(roll) p(pitch) yw(yaw)
            [ 0.0,      0.0,    0.0,     0.0,   -1.0,     0.0 ],  # T1
            [ 0.0,      0.0,    0.0,     1.0,    0.0,     0.0 ],  # T2
            [ 0.0,      0.0,    1.0,     0.0,    0.0,     0.0 ],  # T3
            [ 0.0,      0.0,    0.0,     0.0,    0.0,    -1.0 ],  # T4
            
            [ 1.0,      0.0,    0.0,     0.0,    0.0,     0.0 ],  # T5
            [ 0.0,     -1.0,    0.0,     0.0,    0.0,     0.0 ],  # T6
            [ 0.0,      0.0,    0.0,     0.0,    0.0,     0.0 ],  # T7
            [ 0.0,      0.0,    0.0,     0.0,    0.0,     0.0 ]   # T8
        ])
        
        # Note: T5-T8 mixing depends heavily on exact geometry (which is +Roll vs -Roll).
        # This is a starting point.
        
        self.get_logger().info('Thruster Mixer Node Started')

    def cmd_vel_callback(self, msg):
        # Extract control vector from Twist message
        # ROS Body Frame: X=Forward, Y=Left, Z=Up
        control_vector = np.array([
            msg.linear.x,  # Surge
            msg.linear.y,  # Sway
            msg.linear.z,  # Heave
            msg.angular.x, # Roll
            msg.angular.y, # Pitch
            msg.angular.z  # Yaw
        ])
        
        # Calculate Motor Inputs (-1.0 to 1.0 nominally, but can exceed)
        motor_inputs = np.dot(self.mixing_matrix, control_vector)
        
        # Normalize if any value exceeds 1.0 to maintain ratios (optional but good practice)
        max_val = np.max(np.abs(motor_inputs))
        if max_val > 1.0:
            motor_inputs = motor_inputs / max_val
            
        # Convert to PWM
        pwm_values = []
        for input_val in motor_inputs:
            # Map -1..1 to 1100..1900
            pwm = int(self.pwm_neutral + (input_val * (self.pwm_max - self.pwm_neutral)))
            # Clamp to safe range
            pwm = max(self.pwm_min, min(self.pwm_max, pwm))
            pwm_values.append(pwm)
            
        # Publish
        array_msg = UInt16MultiArray()
        array_msg.data = pwm_values
        self.pwm_publisher.publish(array_msg)
        # self.get_logger().info(f'Published PWM: {pwm_values}')

def main(args=None):
    rclpy.init(args=args)
    node = ThrusterMixer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()