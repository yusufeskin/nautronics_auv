import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import TimerAction, LogInfo
from launch_ros.actions import Node

def generate_launch_description():
    
    config_dir = os.path.join(
        get_package_share_directory('auv_bringup'),
        'config',
        'front_camera_params.yaml'
    )

    gscam_node = Node(
        package='gscam2',
        executable='gscam_main',
        name='front_camera_node',
        namespace='front_camera',
        output='screen',
        parameters=[config_dir]
    )

    keypoint_detector = Node(
        package='auv_vision',
        executable='object_keypoint_detector',
        name='object_keypoint_detector',
        output='screen',
    )

    return LaunchDescription([
        LogInfo(msg='[vision_pipeline] 1. Kamera baslatiliyor (YAML Config ile)...'),
        gscam_node,

        TimerAction(
            period=3.0,
            actions=[
                LogInfo(msg='[vision_pipeline] 2. Keypoint dedektoru baslatiliyor...'),
                keypoint_detector,
            ]
        ),
    ])