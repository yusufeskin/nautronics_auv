"""Layer 0 + Layer 1 against the simulator.

Runs the same gates, with the same tuning, as the real vehicle - only the raw
topics differ and a synthetic pressure source stands in for the barometer the
Gazebo model does not carry. If a change works here and not in the pool, the
difference is in the sensors, not in the stack.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    sensors_share = get_package_share_directory('auv_sensors')
    core_share = get_package_share_directory('auv_core')

    sensors_config = os.path.join(sensors_share, 'config', 'sensors.yaml')
    vehicle_config = os.path.join(core_share, 'config', 'vehicle.yaml')

    ground_truth_topic = LaunchConfiguration('ground_truth_topic')

    frames = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(core_share, 'launch', 'frames.launch.py')
        ),
        launch_arguments={
            'vehicle_config': vehicle_config,
            'use_sim_time': 'true',
        }.items(),
    )

    gates = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(sensors_share, 'launch', 'sensors.launch.py')
        ),
        launch_arguments={
            'sensors_config': sensors_config,
            'vehicle_config': vehicle_config,
            'use_sim_time': 'true',
            # Published by auv_hardware/sim_dvl_bridge, which reshapes the
            # gz DVL message into the same marine_acoustic_msgs/Dvl the real
            # A50 driver emits.
            'dvl_raw_topic': '/sim/dvl/raw',
            # The vio_imu in prototype.urdf.xacro, mounted with no rotation
            # so it is already FLU-aligned. Not /model/prototype_vehicle/imu,
            # which is the flight IMU ArduPilot binds to and is mounted
            # rolled by pi.
            'imu_raw_topic': '/imu0',
            'pressure_raw_topic': '/sim_pressure_source/raw',
        }.items(),
    )

    pressure = Node(
        package='auv_sensors',
        executable='sim_pressure_source',
        name='sim_pressure_source',
        output='screen',
        parameters=[
            sensors_config,
            {'use_sim_time': True, 'ground_truth_topic': ground_truth_topic},
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'ground_truth_topic',
            default_value='/odom',
            description='Gazebo OdometryPublisher output, used only to synthesise pressure.',
        ),
        frames,
        gates,
        pressure,
    ])
