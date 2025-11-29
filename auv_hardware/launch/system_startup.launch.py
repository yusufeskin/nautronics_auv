import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    
    realsense_dir = get_package_share_directory('realsense2_camera')
    auv_hardware_dir = get_package_share_directory('auv_hardware')
    bno055_config = os.path.join(
        auv_hardware_dir,
        'config',
        'bno055_params_i2c.yaml'
    )

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
    )

    pixhawk_baro_reader = Node(
        package='auv_hardware',
        executable='pixhawk_baro_reader',
        name='pixhawk_baro_reader',
        output='screen',
    )

    bno055_node = Node(
        package='bno055',
        executable='bno055',
        parameters=[bno055_config], #its from our package not default one (the names are the same do not get confuse)
        remappings=[
            ('/bno055/imu', '/imu/data'),
            ('/bno055/calib_status', '/imu/calib_status')
        ]
    )

    return LaunchDescription([
        realsense_service,
        pwm_router_node,
        pixhawk_baro_reader,
        bno055_node
    ])