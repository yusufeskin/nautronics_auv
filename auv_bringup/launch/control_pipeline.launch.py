import os
from launch import LaunchDescription
from launch.actions import LogInfo
from launch_ros.actions import Node

def generate_launch_description():
    """
    Sadece kontrol node'larını ve Pixhawk Bridge'i başlatır.
    Kamera, IMU (BNO055) veya Gazebo YOKTUR.
    Amaç: Havuzda (veya masada) yapay zeka olmadan sadece motorların 
    ve mikserin (PID, thruster_mixer) fiziksel olarak doğru çalışıp çalışmadığını test etmektir.
    Kullanım: ros2 launch auv_bringup control_pipeline.launch.py
    """

    # Pixhawk Bridge — PWM değerlerini (ve diğer komutları) donanıma iletir
    pixhawk_bridge = Node(
        package='auv_hardware',
        executable='pixhawk_bridge',
        name='pixhawk_bridge_node',
        output='screen',
    )

    # Thruster Mixer — /cmd_vel'den gelen hız komutlarını thruster'ların PWM değerlerine çevirir
    thruster_mixer = Node(
        package='auv_control',
        executable='thruster_mixer',
        name='thruster_mixer',
        output='screen',
    )

    return LaunchDescription([
        LogInfo(msg='[control_pipeline] Sadece motor kontrolu (Mixer + Pixhawk) baslatiliyor...'),
        pixhawk_bridge,
        thruster_mixer,
    ])
