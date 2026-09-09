import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction, LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    description_dir = get_package_share_directory('auv_description')
    hardware_dir = get_package_share_directory('auv_hardware')

    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(description_dir, 'launch', 'gazebo.launch.py')
        ),
        launch_arguments={
            'use_sim_time': 'true',
            'rviz': 'false',        #RVIZ CLOESD
        }.items()
    )

    thruster_mixer = Node(
        package='auv_control',
        executable='thruster_mixer',
        name='thruster_mixer',
        output='screen',
        parameters=[{'use_sim_time': True}],
    )

    visual_servoing = Node(
        package='auv_control',
        executable='visual_servoing_action',
        name='visual_servoing_action',
        output='screen',
        parameters=[{'use_sim_time': True}],
    )

    keypoint_detector = Node(
        package='auv_vision',
        executable='object_keypoint_detector',
        name='object_keypoint_detector',
        output='screen',
        parameters=[{'use_sim_time': True}],
    )

    yawer = Node(
        package='auv_control',
        executable='yawer',
        name='yawer',
        output='screen',
        parameters=[{'use_sim_time': True}],
    )

    pixhawk_bridge = Node(
        package='auv_hardware',
        executable='pixhawk_bridge2',
        name='pixhawk_bridge_node',
        output='screen',
        parameters=[{'use_sim_time': True}],
    )

    # Simulated DVL, defined once in auv_hardware so running it alone and
    # running it inside the full sim cannot drift apart. It is a plain ROS
    # sensor - it does not feed the autopilot.
    sim_dvl_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(hardware_dir, 'launch', 'sim_dvl.launch.py')
        ),
        launch_arguments={'use_sim_time': 'true'}.items()
    )

    return LaunchDescription([
        # t=0s: Gazebo
        LogInfo(msg='[system_sim]1 '),
        gazebo_launch,
        pixhawk_bridge,
        sim_dvl_launch,

        # t=8s:
        TimerAction(
            period=8.0,
            actions=[
                LogInfo(msg='[system_sim]2'),
                thruster_mixer,
                visual_servoing,
                keypoint_detector,
                yawer,
            ]
        ),

        TimerAction(
            period=12.0,
            actions=[LogInfo(msg='[system_sim]3 ')]
        ),
    ])
