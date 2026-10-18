"""Tek giris noktasi: butun otonomi yigini, sim veya gercek arac.

    ros2 launch auv_bringup autonomy.launch.py sim:=true    # simulasyon
    ros2 launch auv_bringup autonomy.launch.py              # gercek arac

Aradaki fark sadece sensor kaynaklari ve birkac parametre. Katman 0'dan
yukarisi iki durumda da AYNI kod, ayni esikler, ayni akis - simulasyonda
calisan bir sey havuzda calismiyorsa fark sensorlerdedir, yiginda degil.

Otopilot (ArduSub) ayri bir surectir ve bu launch onu baslatmaz:

  simulasyonda   sim_vehicle.py -v ArduSub -f JSON --model JSON --console
                 (Gazebo ayaga kalktiktan SONRA)
  gercek arac    zaten Pixhawk uzerinde calisiyor

Otopilot olmadan da sim calisir: sensorler akar, kestirimci pose uretir,
sadece arac hareket etmez ve mutlak yon kaynagi olmaz. Kontrol denemek icin
autopilot:=true ver.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, IncludeLaunchDescription, LogInfo, TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    core = get_package_share_directory('auv_core')
    sensors = get_package_share_directory('auv_sensors')
    estimation = get_package_share_directory('auv_estimation')
    actuation = get_package_share_directory('auv_actuation')
    description = get_package_share_directory('auv_description')

    sim = LaunchConfiguration('sim')
    autopilot = LaunchConfiguration('autopilot')
    use_heading = LaunchConfiguration('use_heading')
    world = LaunchConfiguration('world')
    rviz = LaunchConfiguration('rviz')

    # `sim` drives use_sim_time everywhere, so no node can end up on the wrong
    # clock - the failure that gives you staleness checks which never fire.
    use_sim_time = sim

    # Frames the simulator's robot_state_publisher already emits from the URDF.
    # Publishing them again from vehicle.yaml would mean two sources for one
    # transform, and a silent disagreement the day somebody edits one.
    sim_skip = 'dvl_link,front_camera_link,bottom_camera_link,back_camera_link,ping1d_sonar_link'
    skip_frames = PythonExpression(["'", sim_skip, "' if '", sim, "' == 'true' else ''"])

    # --- sensor sources: the only real difference between sim and vehicle ---
    dvl_topic = PythonExpression(
        ["'/sim/dvl/raw' if '", sim, "' == 'true' "
         "else '/waterlinked_dvl_driver/velocity_report'"]
    )
    imu_topic = PythonExpression(["'/imu0' if '", sim, "' == 'true' else '/imu/raw'"])
    pressure_topic = PythonExpression(
        ["'/sim_pressure_source/raw' if '", sim, "' == 'true' else '/pressure/raw'"]
    )
    sensors_overlay = PythonExpression(
        ["'", os.path.join(sensors, 'config', 'sensors_sim.yaml'), "' if '", sim, "' == 'true' "
         "else '", os.path.join(sensors, 'config', 'sensors_empty.yaml'), "'"]
    )
    actuation_overlay = PythonExpression(
        ["'", os.path.join(actuation, 'config', 'pilot_demand_sim.yaml'), "' if '", sim,
         "' == 'true' else '", os.path.join(actuation, 'config', 'actuation_empty.yaml'), "'"]
    )

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(description, 'launch', 'gazebo.launch.py')),
        launch_arguments={
            'use_sim_time': 'true',
            'rviz': rviz,
            'world': world,
            # NOT bridge.yaml: that one bridges Gazebo's ground-truth
            # odom -> base_link onto /tf, which robot_localization also
            # publishes. Two publishers on one transform present as jitter,
            # not as an error.
            'bridge_config': os.path.join(description, 'config', 'bridge_newarch.yaml'),
        }.items(),
        condition=IfCondition(sim),
    )

    # Reshapes the gz DVL message into the same marine_acoustic_msgs/Dvl the
    # real Water Linked driver emits, so dvl_gate cannot tell the difference.
    sim_dvl = Node(
        package='auv_hardware', executable='sim_dvl_bridge', name='sim_dvl_bridge',
        output='screen', parameters=[{'use_sim_time': True}],
        condition=IfCondition(sim),
    )

    # The Gazebo model carries no barometer; this synthesises one from ground
    # truth so depth_gate runs the same code in both worlds.
    sim_pressure = Node(
        package='auv_sensors', executable='sim_pressure_source', name='sim_pressure_source',
        output='screen',
        parameters=[os.path.join(sensors, 'config', 'sensors.yaml'),
                    {'use_sim_time': True, 'ground_truth_topic': '/ground_truth/odom'}],
        condition=IfCondition(sim),
    )

    # --- from here down, sim and vehicle run identical code ---
    frames = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(core, 'launch', 'frames.launch.py')),
        launch_arguments={'use_sim_time': use_sim_time, 'skip_frames': skip_frames}.items(),
    )

    gates = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(sensors, 'launch', 'sensors.launch.py')),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'sensors_overlay': sensors_overlay,
            'dvl_raw_topic': dvl_topic,
            'imu_raw_topic': imu_topic,
            'pressure_raw_topic': pressure_topic,
            'use_heading': use_heading,
            # The autopilot's own fused attitude, republished in ENU/FLU by
            # pixhawk_bridge2. heading_gate cross-checks it against the
            # external IMU's gyro before the estimator may use it.
            'heading_raw_topic': '/autopilot/attitude',
        }.items(),
    )

    estimator = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(estimation, 'launch', 'estimation.launch.py')),
        launch_arguments={'use_sim_time': use_sim_time}.items(),
    )

    # --- autopilot side ---
    # Sole owner of the MAVLink link. Blocks on wait_heartbeat(), so it is
    # started late in simulation to give SITL a chance to come up.
    # rc_override: ArduSub gets no DVL and no targets, only /pilot_demand as
    # RC channels - the legacy command paths stay closed.
    pixhawk = Node(
        package='auv_hardware', executable='pixhawk_bridge2', name='pixhawk_bridge_node',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time, 'command_mode': 'rc_override'}],
        condition=IfCondition(autopilot),
    )

    control = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(actuation, 'launch', 'actuation.launch.py')),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'actuation_overlay': actuation_overlay,
        }.items(),
        condition=IfCondition(autopilot),
    )

    return LaunchDescription([
        DeclareLaunchArgument('sim', default_value='false',
                              description='true: Gazebo ve sahte sensor kaynaklari. false: gercek arac.'),
        DeclareLaunchArgument('autopilot', default_value='false',
                              description='pixhawk_bridge2 + kontrol. ArduSub (veya SITL) calisiyor olmali.'),
        DeclareLaunchArgument('use_heading', default_value='false',
                              description='Mutlak yon kaynagini kullan. autopilot:=true gerektirir.'),
        DeclareLaunchArgument('world', default_value='teknofest_pool.world'),
        DeclareLaunchArgument('rviz', default_value='false'),

        LogInfo(msg=['[autonomy] sim=', sim, '  autopilot=', autopilot, '  heading=', use_heading]),

        gazebo,
        sim_dvl,
        sim_pressure,
        frames,
        gates,
        estimator,

        # Gazebo needs a moment before SITL has anything to talk to, and
        # pixhawk_bridge2 blocks until it sees a heartbeat.
        TimerAction(period=8.0, actions=[
            LogInfo(msg='[autonomy] otopilot koprusu baslatiliyor'),
            pixhawk,
            control,
        ]),
    ])
