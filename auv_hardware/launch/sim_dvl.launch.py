"""Simulated DVL: Gazebo sensor -> the real driver's ROS interface.

Stands in for the WaterLinked A50 plus waterlinked_dvl_driver. The DVL is a
plain ROS sensor here, independent of the flight controller - nothing in this
chain talks to the Pixhawk:

    gz /model/prototype_vehicle/dvl
      -> sim_dvl_bridge   -> /sim/dvl/raw
      -> sim_dvl_adapter  -> /waterlinked_dvl_driver/velocity_report
                             /waterlinked_dvl_driver/odom
                             /waterlinked_dvl_driver/dead_reckoning_report

Those are the same three topics the real driver publishes, so swapping this
launch file for waterlinked_dvl_driver's own dvl.launch.py is the only
difference between sim and hardware.

Needs Gazebo running first - sim_dvl_bridge subscribes to a Gazebo Transport
topic, and publishes nothing until the sensor appears.

    ros2 launch auv_hardware sim_dvl.launch.py
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Gazebo saatini kullan (/clock)'
    )

    use_sim_time = LaunchConfiguration('use_sim_time')

    # gz.msgs.DVLVelocityTracking has no ros_gz_bridge factory, so the DVL
    # cannot go in auv_description/config/bridge.yaml - this node subscribes to
    # the Gazebo topic directly and republishes marine_acoustic_msgs/Dvl.
    sim_dvl_bridge = Node(
        package='auv_hardware',
        executable='sim_dvl_bridge',
        name='sim_dvl_bridge',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
    )

    # Renames the raw stream onto the three topics the real WaterLinked A50
    # driver publishes, so mission code cannot tell sim from hardware.
    sim_dvl_adapter = Node(
        package='auv_hardware',
        executable='sim_dvl_adapter',
        name='sim_dvl_adapter',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
    )

    return LaunchDescription([
        use_sim_time_arg,
        sim_dvl_bridge,
        sim_dvl_adapter,
    ])
