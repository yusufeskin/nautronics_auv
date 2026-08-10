#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import UInt16MultiArray
import numpy as np

# cmd_vel  1.0 ----> pwm 2000
# cmd_vel  0.9 ----> pwm 1933
# cmd_vel  0.8 ----> pwm 1898
# cmd_vel  0.7 ----> pwm 1860
# cmd_vel  0.6 ----> pwm 1821
# cmd_vel  0.5 ----> pwm 1788
# cmd_vel  0.4 ----> pwm 1760
# cmd_vel  0.3 ----> pwm 1724
# cmd_vel  0.2 ----> pwm 1675
# cmd_vel  0.1 ----> pwm 1596
# cmd_vel  0.0 ----> pwm 1500
# cmd_vel -0.1 ----> pwm 1428
# cmd_vel -0.2 ----> pwm 1357
# cmd_vel -0.3 ----> pwm 1299
# cmd_vel -0.4 ----> pwm 1246
# cmd_vel -0.5 ----> pwm 1210
# cmd_vel -0.6 ----> pwm 1149
# cmd_vel -0.7 ----> pwm 1122
# cmd_vel -0.8 ----> pwm 1089
# cmd_vel -0.9 ----> pwm 1038
# cmd_vel -1.0 ----> pwm 1000

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
        self.pwm_min = 1000
        self.pwm_max = 2000
        self.pwm_neutral = 1500
        
        # 16V Thruster Data Mapping (PWM vs Thrust in kgf)
        # Reverse thrusts are represented as negative values
        self.thruster_data_pwm = np.array([
            1000, 1050, 1100, 1150, 1200, 1250, 1300, 1350, 1500, 
            1650, 1700, 1750, 1800, 1850, 1900, 1950, 2000
        ])
        self.thruster_data_thrust = np.array([
            -3.52, -3.06, -2.75, -2.11, -1.86, -1.37, -1.05, -0.74, 0.0,
             0.73,  1.13,  1.69,  2.54,  3.15,  3.76,  4.43,  4.68
        ])
        self.max_forward_thrust = 4.68
        self.max_reverse_thrust = 3.52
        
        # Weights (can be tuned)
        self.mixing_matrix = np.array([
            # x(surge) y(sway) z(heave) r(roll) p(pitch) yw(yaw)
            [ 0.0,      0.0,    0.0,     0.0,   -1.0,     0.0 ],  # T1
            [ 0.0,      0.0,    0.0,     1.0,    0.0,     0.0 ],  # T2
            [ 0.0,      0.0,    1.0,     0.0,    0.0,     0.0 ],  # T3
            [ 0.0,      0.0,    0.0,     0.0,    0.0,     1.0 ],  # T4
            
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
            
        pwm_values = []
        for input_val in motor_inputs:
            # Scale normalized input [-1.0, 1.0] to target thrust in kgf
            if input_val >= 0:
                target_thrust = input_val * self.max_forward_thrust
            else:
                target_thrust = input_val * self.max_reverse_thrust
            
            # Interpolate to find the required PWM for the target thrust
            pwm = int(np.interp(target_thrust, self.thruster_data_thrust, self.thruster_data_pwm))
            
            # Clamp to safe range just in case
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