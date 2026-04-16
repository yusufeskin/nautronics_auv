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


    yawer = Node(
        package='auv_control',
        executable='yawer',
        name='yawer',
        output='screen'
    )

    pwm_router = Node(
        package='auv_hardware',
        executable='pwm_router',
        name='pwm_router',
        output='screen',
    )


    state_publisher = Node(
        package='auv_hardware',
        executable='state_publisher',
        name='state_publisher',
        output='screen'
    )

    change_mode_service = Node(
        package='auv_hardware',
        executable='change_mode_service',
        name='change_mode_service',
        output='screen'
    )




    return LaunchDescription([
        TimerAction(
            period=2.0,
            actions=[blind_push_action]
        ),
        TimerAction(
            period=6.0,
            actions=[yawer]
        ),
        TimerAction(
            period=8.0,
            actions=[pwm_router]
        ),
        TimerAction(
            period=12.0,
            actions=[thruster_mixer]
        ),
        TimerAction(
            period=14.0,
            actions=[change_mode_service]
        ),
        TimerAction(
            period=16.0,
            actions=[state_publisher]
        ),
        TimerAction(
            period=19.0,
            actions=[LogInfo(msg="===================================\nALL NODES HAVE STARTED SUCCESSFULLY\n===================================")]
        )
    ])
