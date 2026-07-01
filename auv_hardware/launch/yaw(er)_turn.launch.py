from launch import LaunchDescription
from launch.actions import TimerAction, LogInfo
from launch_ros.actions import Node

def generate_launch_description():
    
    # "pwm_router" ve "pixhawk_bridge" aslında aynı düğümdür (auv_hardware paketindeki pixhawk_bridge).
    # Bu yüzden sadece bir kere başlatıyoruz.
    pixhawk_bridge = Node(
        package='auv_hardware',
        executable='pixhawk_bridge2',
        name='pixhawk_bridge2',
        output='screen'
    )

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
        output='screen',
        #emulate_tty=True

    )

    return LaunchDescription([
        # 0. saniyede pixhawk_bridge başlatılıyor
        TimerAction(
            period=0.0,
            actions=[pixhawk_bridge]
        ),
        # 2. saniyede thruster_mixer başlatılıyor
        TimerAction(
            period=2.0,
            actions=[thruster_mixer]
        ),
        # 4. saniyede blind_push_action başlatılıyor
        TimerAction(
            period=4.0,
            actions=[blind_push_action]
        ),
        # 6. saniyede yawer başlatılıyor
        TimerAction(
            period=6.0,
            actions=[yawer]
        ),
        # 8. saniyede bittiğini gösteren log yazdırılıyor
        TimerAction(
            period=8.0,
            actions=[LogInfo(msg="===================================\nYAW(ER) TURN LAUNCH COMPLETED SUCCESSFULLY\n===================================")]
        )
    ])
