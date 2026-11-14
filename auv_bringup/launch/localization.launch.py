
import os

from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    config = os.path.join(
        get_package_share_directory('auv_bringup'),
        'config',
        'ekf.yaml'
    )

    return LaunchDescription([

        # IMU covariance correction
        Node(
            package='auv_bringup',
            executable='imu_covariance.py',
            name='imu_covariance_node',
            output='screen'
        ),

        # Extended Kalman Filter
        Node(
            package='robot_localization',
            executable='ekf_node',
            name='ekf_filter_node',
            parameters=[config],
            output='screen'
        )

    ])

