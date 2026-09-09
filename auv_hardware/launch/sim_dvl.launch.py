"""Simulated DVL chain: Gazebo sensor -> ROS -> ArduSub's EKF.

Starts the three nodes that carry DVL data out of Gazebo and into the
autopilot, so they need not be run one per terminal:

    gz /model/prototype_vehicle/dvl
      -> sim_dvl_bridge   -> /sim/dvl/raw
      -> sim_dvl_adapter  -> /waterlinked_dvl_driver/{velocity_report,odom}
      -> pixhawk_bridge2  -> VISION_POSITION_DELTA (MAVLink) -> ArduSub EKF3

Run AFTER Gazebo and SITL are already up - pixhawk_bridge2 blocks in
wait_heartbeat() until the autopilot answers on mavlink_url.

    ros2 launch auv_hardware sim_dvl.launch.py

SITL must be told to open the port this connects to; sim_vehicle.py only
opens 14550 on its own, so its command line needs
    --out=udp:127.0.0.1:14551
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

    mavlink_url_arg = DeclareLaunchArgument(
        'mavlink_url',
        default_value='udp:127.0.0.1:14551',
        description='pixhawk_bridge2 MAVLink adresi (SITL icin sim_vehicle.py '
                    '--out=udp:127.0.0.1:14551 gerekir)'
    )

    use_sim_time = LaunchConfiguration('use_sim_time')
    mavlink_url = LaunchConfiguration('mavlink_url')

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

    # Renames the raw stream onto the topics the real WaterLinked A50 driver
    # publishes, so mission code cannot tell sim from hardware.
    sim_dvl_adapter = Node(
        package='auv_hardware',
        executable='sim_dvl_adapter',
        name='sim_dvl_adapter',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
    )

    # Carries DVL velocity to the autopilot as VISION_POSITION_DELTA via its
    # DvlOdomHandler, and is the same node used against the real Pixhawk.
    pixhawk_bridge = Node(
        package='auv_hardware',
        executable='pixhawk_bridge2',
        name='pixhawk_bridge_node',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'mavlink_url': mavlink_url,
        }],
    )

    return LaunchDescription([
        use_sim_time_arg,
        mavlink_url_arg,
        sim_dvl_bridge,
        sim_dvl_adapter,
        pixhawk_bridge,
    ])
