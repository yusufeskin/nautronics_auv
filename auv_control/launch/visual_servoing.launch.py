from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # 1. Thruster Mixer
        Node(
            package='auv_control',
            executable='thruster_mixer',
            name='thruster_mixer',
            output='screen'
        ),
        
        # 2. PWM Router
        Node(
            package='auv_hardware',
            executable='pwm_router',
            name='pwm_router',
            output='screen'
        ),
        
        # 3. Object Keypoint Detector
        Node(
            package='auv_vision',
            executable='object_keypoint_detector',
            name='object_keypoint_detector',
            output='screen'
        ),
        
        # 4. Visual Servoing Action
        Node(
            package='auv_control',
            executable='visual_servoing_action',
            name='visual_servoing_action',
            output='screen'
        )
    ])