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

    # 1. Dosya Yolları
    xacro_file = os.path.join(pkg_share, 'models', 'prototype_vehicle', 'prototype.urdf.xacro')
    world_file = os.path.join(pkg_share, 'worlds', 'deneme.world')

    # 2. Robot Tanımı (Xacro -> URDF)
    robot_desc_content = Command(['xacro ', xacro_file])

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

    # 4. KÖPRÜ (BRIDGE) - Thruster ve Sensor Verileri İçin
    bridge_arguments = [
        '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
        '/model/prototype_vehicle/joint_state@sensor_msgs/msg/JointState[gz.msgs.Model',
    ]

    # Thruster Komut Köprüleri (8 Adet)
    for i in range(1, 9):
        gz_topic = f'/model/prototype/joint/thruster{i}_joint/cmd_thrust'
        bridge_arguments.append(f'{gz_topic}@std_msgs/msg/Float64]gz.msgs.Double')

    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=bridge_arguments,
        output='screen',
        remappings=[
            ('/model/prototype_vehicle/joint_state', '/joint_states'),
        ] + [(f'/model/prototype/joint/thruster{i}_joint/cmd_thrust', f'/thruster/id{i}/cmd') for i in range(1, 9)]
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

    return LaunchDescription([
        robot_state_publisher,
        bridge,
        gz_sim,
        spawn_entity,
    ])