import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo
import yaml
import os
from ament_index_python.packages import get_package_share_directory

class CameraInfoPublisher(Node):
    def __init__(self):
        super().__init__('camera_info_publisher')
        
        self.publisher_ = self.create_publisher(CameraInfo, '/camera/color/camera_info', 10)
        
        self.subscription = self.create_subscription(
            CameraInfo,
            '/camera/color/camera_info_default',
            self.info_callback,
            10)

        bringup_share_dir = get_package_share_directory('auv_bringup')
        yaml_path = os.path.join(bringup_share_dir, 'config', 'camera_parameters.yaml')
        
        self.get_logger().info(f"Kalibrasyon dosyası okunuyor: {yaml_path}")
        with open(yaml_path, 'r') as f:
            self.calib_data = yaml.safe_load(f)

    def info_callback(self, msg):
        msg.width = self.calib_data['image_width']
        msg.height = self.calib_data['image_height']
        msg.k = self.calib_data['camera_matrix']['data']
        msg.d = self.calib_data['distortion_coefficients']['data']
        msg.r = self.calib_data['rectification_matrix']['data']
        msg.p = self.calib_data['projection_matrix']['data']
        msg.distortion_model = self.calib_data['distortion_model']

        self.publisher_.publish(msg)

def main():
    rclpy.init()
    node = CameraInfoPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()