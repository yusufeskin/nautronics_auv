"""Bring up Layer 4: the outer control loop and the boundary to ArduSub.

auv_hardware/pixhawk_bridge2 must already be running with
command_mode:=rc_override: it owns the MAVLink link and forwards /pilot_demand
to the autopilot as RC_CHANNELS_OVERRIDE.

Nothing moves on launch. The vehicle is not armed, the flight mode is not
changed, and both nodes start with their output gated off. Four deliberate
steps, in this order:

    ros2 service call /change_mode  auv_interfaces/srv/SetVehicleMode "{mode_name: 'ALT_HOLD'}"
    ros2 service call /arm          std_srvs/srv/SetBool "{data: true}"
    ros2 service call /pilot_demand/enable    std_srvs/srv/SetBool "{data: true}"
    ros2 service call /pose_controller/enable std_srvs/srv/SetBool "{data: true}"

Enabling the controller holds the vehicle where it is. Then send a target:

    ros2 topic pub --once /pose_controller/target_pose geometry_msgs/msg/PoseStamped \
      "{header: {frame_id: odom}, pose: {position: {x: 2.0, y: 0.0, z: -1.5}}}"

or hold wherever it is now:

    ros2 service call /pose_controller/hold std_srvs/srv/Trigger
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory('auv_actuation')
    demand_config = os.path.join(share, 'config', 'pilot_demand.yaml')
    controller_config = os.path.join(share, 'config', 'pose_controller.yaml')

    use_sim_time = LaunchConfiguration('use_sim_time')
    overlay = LaunchConfiguration('actuation_overlay')

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument(
            'actuation_overlay',
            default_value=os.path.join(share, 'config', 'actuation_empty.yaml'),
            description='Ek parametre dosyasi (orn. pilot_demand_sim.yaml).',
        ),

        # Publishes auv_core_msgs/PilotDemand. It does not talk to the
        # autopilot - auv_hardware/pixhawk_bridge2 owns the MAVLink link and
        # subscribes to /pilot_demand, so that bridge must be running too.
        Node(
            package='auv_actuation',
            executable='pilot_demand',
            name='pilot_demand',
            output='screen',
            parameters=[demand_config, overlay, {'use_sim_time': use_sim_time}],
            remappings=[('~/demand', '/pilot_demand')],
        ),
        Node(
            package='auv_actuation',
            executable='pose_controller',
            name='pose_controller',
            output='screen',
            parameters=[controller_config, overlay, {'use_sim_time': use_sim_time}],
            remappings=[
                ('~/odom', '/odom/filtered'),
                ('~/nav_status', '/nav_status'),
                ('~/cmd_axis', '/pilot_demand/cmd_axis'),
            ],
        ),
    ])
