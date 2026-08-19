import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import TimerAction, LogInfo
from launch_ros.actions import Node


def generate_launch_description():
    bringup_share = get_package_share_directory('auv_bringup')
    bridge_config_path = os.path.join(bringup_share, 'config', 'sim_dvl_bridge.yaml')

    ros_gz_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='sim_dvl_bridge',
        output='screen',
        parameters=[{
            'config_file': bridge_config_path,
            'use_sim_time': True,
        }],
    )

    sim_dvl_adapter = Node(
        package='auv_hardware',
        executable='sim_dvl_adapter',
        name='sim_dvl_adapter',
        output='screen',
        parameters=[{'use_sim_time': True}],
    )

    pixhawk_bridge2 = Node(
        package='auv_hardware',
        executable='pixhawk_bridge2',
        name='pixhawk_bridge2',
        output='screen',
        parameters=[{'use_sim_time': True}],
    )

    gtl_transporter = Node(
        package='auv_bt',
        executable='gtl_transporter',
        name='gtl_transporter',
        output='screen',
        parameters=[{'use_sim_time': True}],
    )

    multi_point_transit = Node(
        package='auv_bt',
        executable='multi_point_transit',
        name='multi_point_transit',
        output='screen',
        parameters=[{'use_sim_time': True}],
    )

    return LaunchDescription([
        LogInfo(msg='[sim_multi_point_mission] t=0: ros_gz_bridge, sim_dvl_adapter, pixhawk_bridge2'),
        ros_gz_bridge,
        sim_dvl_adapter,
        pixhawk_bridge2,

        TimerAction(
            period=6.0,
            actions=[
                LogInfo(msg='[sim_multi_point_mission] t=6: gtl_transporter'),
                gtl_transporter,
            ]
        ),

        TimerAction(
            period=10.0,
            actions=[
                LogInfo(msg='[sim_multi_point_mission] t=10: multi_point_transit'),
                multi_point_transit,
            ]
        ),
    ])
