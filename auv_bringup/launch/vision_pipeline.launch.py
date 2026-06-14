import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction, LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    realsense_dir = get_package_share_directory('realsense2_camera')
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(realsense_dir, 'launch', 'rs_launch.py')
        ),
        launch_arguments={
            'depth_module.profile': '640x480x15',   # derinlik: 640x480, 15fps
            'rgb_camera.profile':   '640x480x15',   # renkli: 640x480, 15fps
            'enable_pointcloud':    'false',         # nokta bulutu kapalı
            'align_depth.enable':   'true',          # derinliği renkli kamerayla hizala
        }.items()
    )

    keypoint_detector = Node(
        package='auv_vision',
        executable='object_keypoint_detector',
        name='object_keypoint_detector',
        output='screen',
    )

    return LaunchDescription([
        LogInfo(msg='[vision_pipeline]1'),
        realsense_launch,

        TimerAction(
            period=3.0,
            actions=[
                LogInfo(msg='[vision_pipeline]2'),
                keypoint_detector,
            ]
        ),
    ])
