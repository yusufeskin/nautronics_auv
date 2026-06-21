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
            # ====================================================================
            # Kamera Bağlantısı
            # Eğer sahte karlı ekran istiyorsan: 'videotestsrc pattern=snow ! video/x-raw,width=1920,height=1080 ! videoconvert'
            # Standart USB Web Kamera (/dev/video0) için: 'v4l2src device=/dev/video0 ! videoconvert'
            # Blue Robotics (UDP) için: 'udpsrc port=5600 ! application/x-rtp, payload=96 ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert'
            # ====================================================================
            ComposableNode(
                package='gscam2',
                plugin='gscam2::GSCamNode',
                name='gscam_publisher',
                parameters=[{
                    # Varsayılan olarak USB kamera (/dev/video0)
                    'gscam_config': 'v4l2src device=/dev/video0 ! videoconvert',
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