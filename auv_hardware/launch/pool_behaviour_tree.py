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

    pixhawk_bridge = Node(
        package='auv_hardware',
        executable='pixhawk_bridge',
        name='pixhawk_bridge',
        output='screen',
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
            actions=[thruster_mixer]
        ),
        TimerAction(
            period=12.0,
            actions=[pixhawk_bridge]
        ),

        TimerAction(
            period=19.0,
            actions=[LogInfo(msg="===================================\nALL NODES HAVE STARTED SUCCESSFULLY\n===================================")]
        )
    ])

#thanks to murat-bot