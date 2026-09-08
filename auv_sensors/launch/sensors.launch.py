"""Bring up Layer 1: one gate per sensor.

Raw topics are launch arguments rather than hard-coded, because the whole
point of the gate layer is that the stack above it cannot tell a simulated
sensor from a real one. sensors_sim.launch.py passes simulator topics; the
defaults here are the real vehicle's.

heading_gate is off by default. It needs an absolute heading source that the
vehicle does not have yet; without one the estimator integrates the gyro and
yaw drifts. Turn it on with use_heading:=true once the hardware is fitted.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    sensors_share = get_package_share_directory('auv_sensors')
    core_share = get_package_share_directory('auv_core')

    default_sensors_config = os.path.join(sensors_share, 'config', 'sensors.yaml')
    default_vehicle_config = os.path.join(core_share, 'config', 'vehicle.yaml')

    sensors_config = LaunchConfiguration('sensors_config')
    vehicle_config = LaunchConfiguration('vehicle_config')
    use_sim_time = LaunchConfiguration('use_sim_time')
    dvl_raw_topic = LaunchConfiguration('dvl_raw_topic')
    imu_raw_topic = LaunchConfiguration('imu_raw_topic')
    pressure_raw_topic = LaunchConfiguration('pressure_raw_topic')
    use_heading = LaunchConfiguration('use_heading')
    heading_raw_topic = LaunchConfiguration('heading_raw_topic')
    magnetic_field_topic = LaunchConfiguration('magnetic_field_topic')

    common = [sensors_config, {'use_sim_time': use_sim_time, 'vehicle_config': vehicle_config}]

    return LaunchDescription([
        DeclareLaunchArgument('sensors_config', default_value=default_sensors_config),
        DeclareLaunchArgument('vehicle_config', default_value=default_vehicle_config),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument(
            'dvl_raw_topic',
            default_value='/waterlinked_dvl_driver/velocity_report',
            description='Raw marine_acoustic_msgs/Dvl stream.',
        ),
        DeclareLaunchArgument(
            'imu_raw_topic',
            default_value='/imu/raw',
            description='Raw sensor_msgs/Imu stream from the IMU driver.',
        ),
        DeclareLaunchArgument(
            'pressure_raw_topic',
            default_value='/pressure/raw',
            description='Raw sensor_msgs/FluidPressure stream from the depth sensor.',
        ),
        DeclareLaunchArgument(
            'use_heading',
            default_value='false',
            description='Run heading_gate. Needs an absolute heading source on the vehicle.',
        ),
        DeclareLaunchArgument(
            'heading_raw_topic',
            default_value='/heading/raw',
            description='sensor_msgs/Imu carrying an orientation, from the compass or AHRS.',
        ),
        DeclareLaunchArgument(
            'magnetic_field_topic',
            default_value='/heading/magnetic_field',
            description='Optional sensor_msgs/MagneticField, for the field magnitude check.',
        ),

        Node(
            package='auv_sensors',
            executable='dvl_gate',
            name='dvl_gate',
            output='screen',
            parameters=common,
            remappings=[('~/raw', dvl_raw_topic)],
        ),
        Node(
            package='auv_sensors',
            executable='imu_gate',
            name='imu_gate',
            output='screen',
            parameters=common,
            remappings=[('~/raw', imu_raw_topic)],
        ),
        Node(
            package='auv_sensors',
            executable='depth_gate',
            name='depth_gate',
            output='screen',
            parameters=common,
            remappings=[('~/raw', pressure_raw_topic)],
        ),
        Node(
            package='auv_sensors',
            executable='heading_gate',
            name='heading_gate',
            output='screen',
            condition=IfCondition(use_heading),
            parameters=common,
            remappings=[
                ('~/raw', heading_raw_topic),
                # The cross-check reference is imu_gate's validated,
                # bias-corrected gyro, not the raw driver stream.
                ('~/gyro', '/imu_gate/data'),
                ('~/magnetic_field', magnetic_field_topic),
            ],
        ),
    ])
