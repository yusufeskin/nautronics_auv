import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    
    realsense_dir = get_package_share_directory('realsense2_camera')
    
    realsense_service = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(realsense_dir, 'launch', 'rs_launch.py')
        ),
        launch_arguments={
            'depth_module.profile': '640x480x15',
            'rgb_camera.profile': '640x480x15',
            'enable_pointcloud': 'false',
            'align_depth.enable': 'true'
        }.items()
    )

    pwm_router_node = Node(
        package='auv_hardware',
        executable='pwm_router_node',
        name='pwm_router_node',
        output='screen',
        emulate_tty=True
    )

    return LaunchDescription([
        realsense_service,
        pwm_router_node
    ])