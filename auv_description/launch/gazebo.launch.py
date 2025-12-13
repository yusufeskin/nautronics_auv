import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import Command, LaunchConfiguration
from launch.conditions import IfCondition

def generate_launch_description():
    pkg_name = 'auv_description'
    pkg_share = get_package_share_directory(pkg_name)
    
    # 1. Dosya Yolları
    xacro_file = os.path.join(pkg_share, 'models', 'prototype_vehicle', 'prototype.urdf.xacro')
    world_file = os.path.join(pkg_share, 'worlds', 'deneme.world')
    bridge_config_path = os.path.join(pkg_share, 'config', 'bridge.yaml')
    rviz_config_path = os.path.join(pkg_share, 'config', 'rviz_config.rviz')
    # 2. Robot Tanımı (Xacro -> URDF)
    robot_desc_content = Command(['xacro ', xacro_file])


    rviz_arg = DeclareLaunchArgument(
        'rviz', 
        default_value='true',
        description='RViz2 ifcond'
    )

    sim_time_arg = DeclareLaunchArgument(
        'use_sim_time', 
        default_value='true',
        description='Gazebo time'
    )

    use_rviz = LaunchConfiguration('rviz')
    use_sim_time = LaunchConfiguration('use_sim_time')

    # 3. Robot State Publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_desc_content,
            'use_sim_time': True
        }]
    )

    ros_gz_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='ros_gz_bridge',
        output='screen',
        parameters=[{
            'config_file': bridge_config_path,
        }]
    )


    # 5. Gazebo Simülasyonu
    # Not: GZ_SIM_RESOURCE_PATH artık setup.bash tarafından otomatik ayarlanıyor!
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': f'-r {world_file}'}.items(),
    )

    # 6. Spawn (Robotu Yarat)
    spawn_entity = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=['-topic', 'robot_description', '-name', 'prototype_vehicle', '-z', '-3.0'], # Biraz aşağıda doğsun
        output='screen'
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config_path],
        condition=IfCondition(use_rviz)
    )

    return LaunchDescription([
        rviz_arg,      
        sim_time_arg,
        robot_state_publisher,
        ros_gz_bridge,
        gz_sim,
        spawn_entity,
        rviz_node
    ])