import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import time

OUTPUT_DIR_NAME = "gate_dataset"

class ImageCollector(Node):
    def __init__(self):
        super().__init__("image_collector_node")

        # Change '/camera/front' to your actual camera topic if different
        self.subscription = self.create_subscription(
            Image, 
            "/camera/front", 
            self.camera_callback, 
            10)

        self.bridge = CvBridge()

        # nautronics_auv//gate_dataset
        ws_root = os.path.abspath(__file__)
        for i in range(7): ws_root = os.path.dirname(ws_root)
        self.output_dir = os.path.join(ws_root, OUTPUT_DIR_NAME)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.img_count = 0
        self.last_capture_time = time.time()
        self.capture_interval = 0.5  # Seconds

        self.get_logger().info('Image Collector started. Saving images to: ' + self.output_dir)

    def camera_callback(self, msg):
        current_time = time.time()

        # Capture frame based on the defined interval
        if current_time - self.last_capture_time >= self.capture_interval:
            try:
                # Convert ROS Image message to OpenCV BGR format
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

                # Generate unique filename
                filename = f'frame_{self.img_count:04d}.jpg'
                file_path = os.path.join(self.output_dir, filename)

                # Save the image
                cv2.imwrite(file_path, cv_image)

                self.get_logger().info(f'Saved: {filename}')
                self.img_count += 1
                self.last_capture_time = current_time

            except Exception as e:
                self.get_logger().error(f'Failed to save image: {str(e)}')


def main(args=None):
    rclpy.init(args=args)
    node = ImageCollector()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Image Collector stopped by user.')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main() 