from launch import LaunchDescription
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

def generate_launch_description():
    container = ComposableNodeContainer(
        name='auv_perception_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[
            # 1. Sahte Kamera Düğümü (gscam2)
            ComposableNode(
                package='gscam2',
                plugin='gscam2::GSCamNode',
                name='gscam_publisher',
                parameters=[{
                    'gscam_config': 'videotestsrc pattern=snow ! video/x-raw,width=1920,height=1080 ! videoconvert',
                    'use_intra_process_comms': True
                }]
            ),

            ComposableNode(
                package='auv_perception_cpp',
                plugin='auv_perception_cpp::MultiObjectTrtNode',
                name='tensorrt_detector',
                parameters=[{
                    'use_intra_process_comms': True
                }]
            )
        ],
        output='screen',
    )

    return LaunchDescription([container])