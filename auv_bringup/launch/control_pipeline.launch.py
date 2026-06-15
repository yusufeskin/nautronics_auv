from launch import LaunchDescription
from launch.actions import LogInfo
from launch_ros.actions import Node

def generate_launch_description():
    pixhawk_bridge = Node(
        package='auv_hardware',
        executable='pixhawk_bridge',
        name='pixhawk_bridge_node',
        output='screen',
    )
    thruster_mixer = Node(
        package='auv_control',
        executable='thruster_mixer',
        name='thruster_mixer',
        output='screen',
    )

    return LaunchDescription([
        LogInfo(msg='[control_pipeline]'),
        pixhawk_bridge,
        thruster_mixer,
    ])
