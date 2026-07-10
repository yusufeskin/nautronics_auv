import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import TimerAction, LogInfo, RegisterEventHandler, EmitEvent
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch_ros.actions import Node

def generate_launch_description():
    blind_push = Node(
        package='auv_control',
        executable='blind_push_action',
        name='blind_push_action',
        output='screen',
    )

    yawer = Node(
        package='auv_control',
        executable='yawer',
        name='yawer',
        output='screen',
    )

    visual_servoing = Node(
        package='auv_control',
        executable='visual_servoing_action',
        name='visual_servoing_action',
        output='screen',
    )

    thruster_mixer = Node(
        package='auv_control',
        executable='thruster_mixer',
        name='thruster_mixer',
        output='screen',
    )

    pixhawk_bridge = Node(
        package='auv_hardware',
        executable='pixhawk_bridge2',
        name='pixhawk_bridge_node2',
        output='screen',
    )

    yolo_node = Node(
        package='auv_vision',            
        executable='yolo_keypoint_lifecycle', 
        name='universal_yolo_node', 
        output='screen',
        emulate_tty=True,         
        parameters=[
            {
                'model_name': 'gate.pt', 
                'model_type': 'keypoint',    
                'image_topic': '/camera/camera/color/image_raw',      
                'ema_alpha': 0.70,                 
                'distance_gate_threshold': 40.0,   
                'miss_frames_limit': 15,
            }
        ]
    )
    
    gate_bt_node = Node(
        package='auv_bt',
        executable='gate_behaviour_tree',
        name='gate_behaviour_tree_node',
        output='screen',
        emulate_tty=True
    )

    return LaunchDescription([
        LogInfo(msg='[gate_bt] Starting blind_push at t=0s'),
        blind_push,

        TimerAction(
            period=1.0,
            actions=[
                LogInfo(msg='[gate_bt] Starting yawer at t=1s'),
                yawer
            ]
        ),
        
        TimerAction(
            period=2.0,
            actions=[
                LogInfo(msg='[gate_bt] Starting visual_servoing at t=2s'),
                visual_servoing
            ]
        ),
        
        TimerAction(
            period=3.0,
            actions=[
                LogInfo(msg='[gate_bt] Starting thruster_mixer at t=3s'),
                thruster_mixer
            ]
        ),
        
        TimerAction(
            period=4.0,
            actions=[
                LogInfo(msg='[gate_bt] Starting pixhawk_bridge2 at t=4s'),
                pixhawk_bridge
            ]
        ),
        
        TimerAction(
            period=5.0,
            actions=[
                LogInfo(msg='[gate_bt] Starting yolo_keypoint_lifecycle at t=5s'),
                yolo_node
            ]
        ),
        
        TimerAction(
            period=8.0,
            actions=[
                LogInfo(msg='[gate_bt] Starting gate_behaviour_tree at t=8s'),
                gate_bt_node
            ]
        ),
        
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=gate_bt_node,
                on_exit=[EmitEvent(event=Shutdown())],
            )
        )
    ])
