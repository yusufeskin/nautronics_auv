import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction, LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    realsense_dir   = get_package_share_directory('realsense2_camera')
    auv_hardware_dir = get_package_share_directory('auv_hardware')

    bno055_config = os.path.join(auv_hardware_dir, 'config', 'bno055_params_i2c.yaml')

    #wxternal
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(realsense_dir, 'launch', 'rs_launch.py')
        ),
        launch_arguments={
            'depth_module.profile': '640x480x15',
            'rgb_camera.profile':   '640x480x15',
            'enable_pointcloud':    'false',
            'align_depth.enable':   'true',
        }.items()
    )

    pixhawk_bridge = Node(
        package='auv_hardware',
        executable='pixhawk_bridge',
        name='pixhawk_bridge_node',
        output='screen',
    )

    # external
    bno055 = Node(
        package='bno055',
        executable='bno055',
        name='bno055',
        parameters=[bno055_config],
        remappings=[
            ('/bno055/imu',          '/imu/data'),
            ('/bno055/calib_status', '/imu/calib_status'),
        ]
    )

    thruster_mixer = Node(
        package='auv_control',
        executable='thruster_mixer',
        name='thruster_mixer',
        output='screen',
    )


    visual_servoing = Node(
        package='auv_control',
        executable='visual_servoing_action',
        name='visual_servoing_action',
        output='screen',
    )

    keypoint_detector = Node(
        package='auv_vision',
        executable='object_keypoint_detector',
        name='object_keypoint_detector',
        output='screen',
    )

    yawer = Node(
        package='auv_control',
        executable='yawer',
        name='yawer',
        output='screen',
    )

    return LaunchDescription([
        # t=0s: 
        LogInfo(msg='[system_real]1'),
        realsense_launch,
        pixhawk_bridge,
        bno055,

        # t=6s:
        TimerAction(
            period=6.0,
            actions=[
                LogInfo(msg='[system_real]2'),
                thruster_mixer,
                visual_servoing,
                keypoint_detector,
                yawer,
            ]
        ),

        TimerAction(
            period=10.0,
            actions=[LogInfo(msg='[system_real]3 ')]
        ),
    ])
