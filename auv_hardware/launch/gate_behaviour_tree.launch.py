from launch import LaunchDescription
from launch.actions import TimerAction, LogInfo
from launch_ros.actions import Node

def generate_launch_description():
    
    thruster_mixer = Node(
        package='auv_control',
        executable='thruster_mixer',
        name='thruster_mixer',
        output='screen'
    )

    blind_push_action = Node(
        package='auv_control',
        executable='blind_push_action',
        name='blind_push_action',
        output='screen'
    )

    visual_servoing_action = Node(
        package='auv_control',
        executable='visual_servoing_action',
        name='visual_servoing_action',
        output='screen'
    )

    yawer = Node(
        package='auv_control',
        executable='yawer',
        name='yawer',
        output='screen'
    )

    pwm_router_node = Node(
        package='auv_hardware',
        executable='pixhawk_bridge',
        name='pixhawk_bridge_node',
        output='screen',
    )

    object_keypoint_detector = Node(
        package='auv_vision',
        executable='object_keypoint_detector',
        name='object_keypoint_detector',
        output='screen'
    )

    return LaunchDescription([
        TimerAction(
            period=0.0,
            actions=[pwm_router_node]
        ),
        TimerAction(
            period=2.0,
            actions=[thruster_mixer]
        ),
        TimerAction(
            period=4.0,
            actions=[blind_push_action]
        ),
        TimerAction(
            period=6.0,
            actions=[visual_servoing_action]
        ),
        TimerAction(
            period=8.0,
            actions=[yawer]
        ),
        TimerAction(
            period=10.0,
            actions=[object_keypoint_detector]
        ),
        TimerAction(
            period=13.0,
            actions=[LogInfo(msg="===================================\nALL NODES HAVE STARTED SUCCESSFULLY\n===================================")]
        )
    ])
