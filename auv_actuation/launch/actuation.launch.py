"""Bring up Layer 4's hardware boundary.

This starts only the bridge. It does not arm the vehicle, does not change
flight mode, and starts with its output gated off - the operator does all
three deliberately:

    ros2 service call /change_mode        auv_interfaces/srv/SetVehicleMode "{mode_name: 'ALT_HOLD'}"
    ros2 service call /arm                std_srvs/srv/SetBool "{data: true}"
    ros2 service call /ardusub_bridge/enable std_srvs/srv/SetBool "{data: true}"
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('auv_actuation'), 'config', 'ardusub_bridge.yaml'
    )

    use_sim_time = LaunchConfiguration('use_sim_time')
    connection = LaunchConfiguration('connection')

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument(
            'connection',
            default_value='udp:127.0.0.1:14551',
            description='MAVLink endpoint. Same for SITL and the vehicle by default.',
        ),
        Node(
            package='auv_actuation',
            executable='ardusub_bridge',
            name='ardusub_bridge',
            output='screen',
            parameters=[config, {
                'use_sim_time': use_sim_time,
                'connection': connection,
            }],
        ),
    ])
