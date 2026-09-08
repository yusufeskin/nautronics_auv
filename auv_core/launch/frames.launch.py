"""Publish the vehicle's static frame tree.

Include this from every bringup - sim and real alike. Sensor gates and the
estimator both look up base_link -> sensor transforms, so nothing downstream
works until this is running.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_config = os.path.join(
        get_package_share_directory('auv_core'), 'config', 'vehicle.yaml'
    )

    vehicle_config = LaunchConfiguration('vehicle_config')
    use_sim_time = LaunchConfiguration('use_sim_time')
    skip_frames = LaunchConfiguration('skip_frames')

    return LaunchDescription([
        DeclareLaunchArgument(
            'vehicle_config',
            default_value=default_config,
            description='Path to vehicle.yaml holding sensor extrinsics.',
        ),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Follow /clock instead of the wall clock.',
        ),
        DeclareLaunchArgument(
            'skip_frames',
            default_value='',
            description=(
                'Virgulle ayrilmis, baskasinin yayinladigi cerceveler. '
                'Simulasyonda robot_state_publisher URDF linklerini yayinlar; '
                'ayni transformu iki yerden yayinlamak sessiz bir tutarsizliktir.'
            ),
        ),
        Node(
            package='auv_core',
            executable='static_frames',
            name='static_frames',
            output='screen',
            parameters=[{
                'vehicle_config': vehicle_config,
                'use_sim_time': use_sim_time,
                'skip_frames': skip_frames,
            }],
        ),
    ])
