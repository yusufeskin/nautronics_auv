"""Jetson motion control: guidance + position_controller + rc_output.

Send a goal (odom frame) once both are enabled:
    ros2 topic pub --once /guidance/goal geometry_msgs/msg/PoseStamped \
      "{header: {frame_id: odom}, pose: {position: {x: 3.0, y: 0.0, z: -0.5}, orientation: {w: 1.0}}}"

Needs, already running: the EKF (auv_bringup localization.launch.py) and
pixhawk_bridge2 (it turns pwm_router into RC_CHANNELS_OVERRIDE).
Do NOT run the legacy thruster_mixer at the same time - both publish pwm_router.

Both nodes start disabled:
    ros2 service call /rc_output/enable std_srvs/srv/SetBool "{data: true}"
    ros2 service call /controller/enable std_srvs/srv/SetBool "{data: true}"
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    config = os.path.join(get_package_share_directory('auv_control'), 'config', 'motion_control.yaml')
    use_sim_time = ParameterValue(LaunchConfiguration('use_sim_time'), value_type=bool)

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        Node(
            package='auv_control',
            executable='guidance',
            name='guidance',
            parameters=[config, {'use_sim_time': use_sim_time}],
            output='screen',
        ),
        Node(
            package='auv_control',
            executable='position_controller',
            name='position_controller',
            parameters=[config, {'use_sim_time': use_sim_time}],
            output='screen',
        ),
        Node(
            package='auv_control',
            executable='rc_output',
            name='rc_output',
            parameters=[config, {'use_sim_time': use_sim_time}],
            output='screen',
        ),
    ])
