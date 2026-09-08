"""Bring up Layer 4: the outer control loop and the boundary to ArduSub.

Nothing moves on launch. The vehicle is not armed, the flight mode is not
changed, and both nodes start with their output gated off. Four deliberate
steps, in this order:

    ros2 service call /change_mode  auv_interfaces/srv/SetVehicleMode "{mode_name: 'ALT_HOLD'}"
    ros2 service call /arm          std_srvs/srv/SetBool "{data: true}"
    ros2 service call /ardusub_bridge/enable  std_srvs/srv/SetBool "{data: true}"
    ros2 service call /pose_controller/enable std_srvs/srv/SetBool "{data: true}"

Then send a target:

    ros2 topic pub --once /pose_controller/target_pose geometry_msgs/msg/PoseStamped \
      "{header: {frame_id: odom}, pose: {position: {x: 2.0, y: 0.0, z: -1.5}}}"
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory('auv_actuation')
    bridge_config = os.path.join(share, 'config', 'ardusub_bridge.yaml')
    controller_config = os.path.join(share, 'config', 'pose_controller.yaml')

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
            parameters=[bridge_config, {
                'use_sim_time': use_sim_time,
                'connection': connection,
            }],
        ),
        Node(
            package='auv_actuation',
            executable='pose_controller',
            name='pose_controller',
            output='screen',
            parameters=[controller_config, {'use_sim_time': use_sim_time}],
            remappings=[
                ('~/odom', '/odom/filtered'),
                ('~/cmd_wrench', '/ardusub_bridge/cmd_wrench'),
                ('~/saturated', '/ardusub_bridge/saturated'),
            ],
        ),
    ])
