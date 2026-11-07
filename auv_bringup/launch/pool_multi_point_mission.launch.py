import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction, LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    # Our wrapper, not the driver's launch directly: it supplies the parameter
    # file that actually reaches the node (acoustics on). See real_dvl.launch.py.
    hardware_dir = get_package_share_directory('auv_hardware')

    dvl_driver_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(hardware_dir, 'launch', 'real_dvl.launch.py')
        )
    )

    pixhawk_bridge = Node(
        package='auv_hardware',
        executable='pixhawk_bridge2',
        name='pixhawk_bridge_node2',
        output='screen',
    )

    gtl_transporter = Node(
        package='auv_control',
        executable='gtl_transporter',
        name='gtl_transporter',
        output='screen',
    )

    multi_point_transit = Node(
        package='auv_bt',
        executable='multi_point_transit',
        name='multi_point_transit',
        output='screen',
    )

    return LaunchDescription([
        LogInfo(msg='[pool_multi_point_mission] Starting DVL driver and pixhawk_bridge2 at t=0s'),
        dvl_driver_launch,
        pixhawk_bridge,

        TimerAction(
            period=6.0,
            actions=[
                LogInfo(msg='[pool_multi_point_mission] Starting gtl_transporter at t=6s'),
                gtl_transporter,
            ]
        ),

        TimerAction(
            period=10.0,
            actions=[
                LogInfo(msg='[pool_multi_point_mission] Starting multi_point_transit mission at t=10s'),
                multi_point_transit,
            ]
        ),
    ])
