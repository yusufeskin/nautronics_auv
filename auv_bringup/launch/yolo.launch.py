import os
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    
    yolo_node = Node(
        package='auv_vision',            
        executable='yolo_keypoint_lifecycle', 
        name='universal_yolo_node', 
        output='screen',
        emulate_tty=True,         
        parameters=[
            {
                'model_name': 'slalom.pt', 
                'model_type': 'bbox',    
                'image_topic': '/camera/front',      
                'ema_alpha': 0.70,                 
                'distance_gate_threshold': 40.0,   
                'miss_frames_limit': 15            
            }
        ]
    )
    
    bbox_pnp_solver = Node(
        package='auv_vision',
        executable='bbox_pnp_solver',
        name='bbox_pnp_solver_node',
        output='screen',
        emulate_tty=True,
        parameters=[
            {
                # Intrinsics for the same camera yolo_node reads above
                # (/camera/front); auv_description publishes them alongside
                # the image. The old '/camera/camera_info' never existed.
                'info_topic': '/camera/front/camera_info'
            }
        ]
    
    )


    return LaunchDescription([
        yolo_node,
        bbox_pnp_solver
    ])