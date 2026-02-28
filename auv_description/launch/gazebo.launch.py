import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import Command, LaunchConfiguration
from launch.conditions import IfCondition
from launch.actions import SetEnvironmentVariable

def generate_launch_description():
    pkg_name = 'auv_description'
    pkg_share = get_package_share_directory(pkg_name)
    
    # 1. Dosya Yolları
    xacro_file = os.path.join(pkg_share, 'models', 'prototype_vehicle', 'prototype.urdf.xacro')
    world_file = os.path.join(pkg_share, 'worlds', 'pool.world')
    bridge_config_path = os.path.join(pkg_share, 'config', 'bridge.yaml')
    rviz_config_path = os.path.join(pkg_share, 'config', 'rviz_config.rviz')
    torpedo_xacro_file = os.path.join(pkg_share, 'models', 'prototype_vehicle', 'macros', 'torpedo.xacro')
    torpedo_desc_content = Command(['xacro ', torpedo_xacro_file])
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
        arguments=['-topic', 'robot_description', '-name', 'prototype_vehicle', '-z', '-1.0', '-x', '-9', '-y', '5'], # Biraz aşağıda doğsun
        output='screen'
    )

    spawn_torpedo = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-string', torpedo_desc_content, 
            '-name', 'torpedo_model', # Xacro'daki child_model ismiyle BİREBİR aynı olmalı
            '-z', '-1.2', '-x', '-9', '-y', '5' # AUV'nin 20 cm altında doğuyor
        ],
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
        spawn_torpedo,
        # rviz_node
    ])
