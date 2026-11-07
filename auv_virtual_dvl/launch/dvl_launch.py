from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        
        Node(
            package='auv_virtual_dvl',
            executable='pwm_publisher',
            name='pwm_publisher',
            output='screen'
        ),
        
        
        Node(
            package='auv_virtual_dvl',
            executable='metrekayma',
            name='metrekayma',
            output='screen'
        ),

        
       
    ])