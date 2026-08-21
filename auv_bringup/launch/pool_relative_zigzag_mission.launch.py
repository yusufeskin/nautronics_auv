import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction, LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    dvl_driver_dir = get_package_share_directory('waterlinked_dvl_driver')

    dvl_driver_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(dvl_driver_dir, 'launch', 'dvl.launch.py')
        )
    )

    pixhawk_bridge = Node(
        package='auv_hardware',
        executable='pixhawk_bridge2',
        name='pixhawk_bridge_node2',
        output='screen',
    )

    relative_zigzag_test = Node(
        package='auv_bt',
        executable='relative_zigzag_test',
        name='relative_zigzag_test',
        output='screen',
    )

    return LaunchDescription([
        LogInfo(msg='[pool_relative_zigzag_mission] Starting DVL driver and pixhawk_bridge2 at t=0s'),
        dvl_driver_launch,
        pixhawk_bridge,

        TimerAction(
            period=8.0,
            actions=[
                LogInfo(msg='[pool_relative_zigzag_mission] Starting relative_zigzag_test mission at t=8s'),
                relative_zigzag_test,
            ]
        ),
    ])
