"""Real DVL: Water Linked A50 via waterlinked_dvl_driver.

Hardware counterpart of sim_dvl.launch.py - both produce the same three
topics, so everything downstream is identical in sim and on the vehicle:

    /waterlinked_dvl_driver/velocity_report        marine_acoustic_msgs/Dvl
    /waterlinked_dvl_driver/odom                   nav_msgs/Odometry
    /waterlinked_dvl_driver/dead_reckoning_report  geometry_msgs/PoseWithCovarianceStamped

Reuses the driver's own launch file (it also drives the lifecycle node through
configure -> activate), but hands it our config/waterlinked_dvl.yaml instead
of the driver's dvl.yaml - see that file for why.

    ros2 launch auv_hardware real_dvl.launch.py
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    driver_dir = get_package_share_directory('waterlinked_dvl_driver')
    hardware_dir = get_package_share_directory('auv_hardware')

    dvl_driver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(driver_dir, 'launch', 'dvl.launch.py')
        ),
        launch_arguments={
            'parameters_file': os.path.join(hardware_dir, 'config', 'waterlinked_dvl.yaml'),
        }.items(),
    )

    return LaunchDescription([dvl_driver])
