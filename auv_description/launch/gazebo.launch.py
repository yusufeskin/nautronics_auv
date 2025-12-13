import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import Command

def generate_launch_description():
    pkg_name = 'auv_description'
    pkg_share = get_package_share_directory(pkg_name)
    xacro_file = os.path.join(pkg_share, 'models', 'prototype_vehicle', 'prototype.urdf.xacro')
    install_dir = os.path.join(pkg_share, '..')
    world_file_path = os.path.join(pkg_share, 'worlds', 'deneme.world')
    gz_resource_path = SetEnvironmentVariable(
        name='GZ_SIM_RESOURCE_PATH',
        value=install_dir
    )

    robot_desc_content = Command(['xacro ', xacro_file])

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

    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            '/model/prototype_vehicle/joint_state@sensor_msgs/msg/JointState[gz.msgs.Model'
        ],
        output='screen',
        remappings=[
            ('/model/prototype_vehicle/joint_state', '/joint_states')
        ]
    )

    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': f'-r {world_file_path}'}.items(),
    )

    spawn_entity = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=['-topic', 'robot_description', '-name', 'prototype_vehicle'],
        output='screen'
    )

    return LaunchDescription([
        gz_resource_path,
        robot_state_publisher,
        bridge,
        gz_sim,
        spawn_entity,
    ])