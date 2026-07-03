import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction, LogInfo, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    realsense_dir   = get_package_share_directory('realsense2_camera')
    auv_hardware_dir = get_package_share_directory('auv_hardware')
    auv_bringup_dir = get_package_share_directory('auv_bringup')
    tracker_config_path = os.path.join(auv_bringup_dir, 'config', 'botsort.yaml')

    #wxternal
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(realsense_dir, 'launch', 'rs_launch.py')
        ),
        launch_arguments={
            'rgb_camera.color_profile': '640x480x30',
            'depth_module.depth_profile': '640x480x30',
            # 'rgb_camera.enable_auto_exposure': 'false',
            # 'depth_module.enable_auto_exposure': 'false',
            # 'enable_pointcloud': 'false',
            # 'align_depth.enable': 'true',
        }.items()
    )

    pixhawk_bridge = Node(
        package='auv_hardware',
        executable='pixhawk_bridge2',
        name='pixhawk_bridge_node2',
        output='screen',
    )

    foxglove_launch = ExecuteProcess(
        cmd=['ros2', 'launch', 'foxglove_bridge', 'foxglove_bridge_launch.xml', 'port:=8765'],
        output='screen'
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

    yolo_node = Node(
        package='auv_vision',            
        executable='yolo_keypoint_lifecycle', 
        name='universal_yolo_node', 
        output='screen',
        emulate_tty=True,         
        parameters=[
            {
                'model_name': 'torpedo_last.pt', 
                'model_type': 'keypoint',    
                'image_topic': '/camera/camera/color/image_raw',      
                'ema_alpha': 0.70,                 
                'distance_gate_threshold': 40.0,   
                'miss_frames_limit': 15,
                # 'tracker_type': tracker_config_path         
            }
        ]
    )

    pnp_solver = Node(
        package='auv_vision',
        executable='pnp_solver',
        name='pnp_solver',
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
        # realsense_launch,
        pixhawk_bridge,
        foxglove_launch,

        # t=6s:
        TimerAction(
            period=6.0,
            actions=[
                LogInfo(msg='[system_real]2'),
                thruster_mixer,
                visual_servoing,
                # yolo_node,
                # pnp_solver,
                yawer,
            ]
        )
    ])
