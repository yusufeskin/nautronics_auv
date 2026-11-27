#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import UInt16MultiArray
import numpy as np
from auv_control import mixer_math

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

        # Thruster Configuration Parameters (values live in mixer_math)
        # PWM Range
        self.pwm_min = mixer_math.PWM_MIN
        self.pwm_max = mixer_math.PWM_MAX
        self.pwm_neutral = mixer_math.PWM_NEUTRAL

        # 16V Thruster Data Mapping (PWM vs Thrust in kgf)
        # Reverse thrusts are represented as negative values
        self.thruster_data_pwm = mixer_math.THRUSTER_DATA_PWM.copy()
        self.thruster_data_thrust = mixer_math.THRUSTER_DATA_THRUST.copy()
        self.max_forward_thrust = mixer_math.MAX_FORWARD_THRUST
        self.max_reverse_thrust = mixer_math.MAX_REVERSE_THRUST

        # Weights (can be tuned) - rows T1..T8, columns surge sway heave roll pitch yaw
        self.mixing_matrix = mixer_math.MIXING_MATRIX.copy()

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
        # Normalize if any value exceeds 1.0 to maintain ratios (optional but good practice)
        motor_inputs = mixer_math.mix(self.mixing_matrix, control_vector)

        pwm_values = []
        for input_val in motor_inputs:
            # Scale to thrust, interpolate PWM from thruster data, clamp to safe range
            pwm = mixer_math.input_to_pwm(
                input_val,
                self.thruster_data_thrust,
                self.thruster_data_pwm,
                self.max_forward_thrust,
                self.max_reverse_thrust,
                self.pwm_min,
                self.pwm_max)
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