"""Layers 0, 1 and 2 together: frames, gates, and the state estimate.

This is the one to launch when you want a pose out of the vehicle. Pass
sim:=true to wire the gates to the simulator instead of the real drivers.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    sensors_share = get_package_share_directory('auv_sensors')
    core_share = get_package_share_directory('auv_core')
    estimation_share = get_package_share_directory('auv_estimation')

    sim = LaunchConfiguration('sim')
    use_heading = LaunchConfiguration('use_heading')

    estimation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(estimation_share, 'launch', 'estimation.launch.py')
        ),
        launch_arguments={'use_sim_time': sim}.items(),
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'sim',
            default_value='false',
            description='Wire the gates to simulator topics and follow /clock.',
        ),
        DeclareLaunchArgument(
            'use_heading',
            default_value='false',
            description='Run heading_gate. Real vehicle only, and only once the hardware is fitted.',
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(sensors_share, 'launch', 'sensors_sim.launch.py')
            ),
            condition=IfCondition(sim),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(core_share, 'launch', 'frames.launch.py')
            ),
            condition=UnlessCondition(sim),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(sensors_share, 'launch', 'sensors.launch.py')
            ),
            launch_arguments={'use_heading': use_heading}.items(),
            condition=UnlessCondition(sim),
        ),

        estimation,
    ])
