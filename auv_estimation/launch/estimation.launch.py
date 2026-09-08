"""Bring up Layer 2: state estimation.

Three nodes:

  imu_filter_madgwick  gyro + accelerometer -> orientation with roll and pitch
                       referenced to gravity (mag-free)
  ekf_filter_node      fuses DVL velocity, depth, that attitude, and - when
                       fitted - absolute heading, into /odom/filtered, and
                       publishes odom -> base_link
  nav_status           says whether that estimate is aided or coasting

Layer 1 must already be running; this launch file does not start it. Use
auv_sensors/sensors.launch.py or sensors_sim.launch.py first, or
navigation.launch.py to get both at once.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory('auv_estimation')
    ekf_config = os.path.join(share, 'config', 'ekf.yaml')
    madgwick_config = os.path.join(share, 'config', 'madgwick.yaml')

    use_sim_time = LaunchConfiguration('use_sim_time')

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false'),

        # imu_gate strips the driver's orientation; this puts a mag-free one
        # back, so that roll and pitch reach the filter as absolute
        # measurements instead of gyro integrals.
        Node(
            package='imu_filter_madgwick',
            executable='imu_filter_madgwick_node',
            name='imu_filter_madgwick',
            output='screen',
            parameters=[madgwick_config, {'use_sim_time': use_sim_time}],
            remappings=[
                ('imu/data_raw', '/imu_gate/data'),
                ('imu/data', '/imu/attitude'),
            ],
        ),

        Node(
            package='robot_localization',
            executable='ekf_node',
            name='ekf_filter_node',
            output='screen',
            parameters=[ekf_config, {'use_sim_time': use_sim_time}],
            remappings=[
                ('odometry/filtered', '/odom/filtered'),
                ('accel/filtered', '/accel/filtered'),
            ],
        ),

        # Health topics are remapped here rather than hard-coded in the node,
        # so adding a sensor is a launch-file change.
        Node(
            package='auv_estimation',
            executable='nav_status',
            name='nav_status',
            output='screen',
            parameters=[{'use_sim_time': use_sim_time}],
            remappings=[
                ('~/dvl_health', '/dvl_gate/health'),
                ('~/imu_health', '/imu_gate/health'),
                ('~/depth_health', '/depth_gate/health'),
                ('~/heading_health', '/heading_gate/health'),
                ('~/odom', '/odom/filtered'),
                ('~/gyro_bias', '/imu_gate/gyro_bias'),
            ],
        ),
    ])
