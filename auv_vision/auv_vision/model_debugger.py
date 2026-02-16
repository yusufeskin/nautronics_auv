import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import os

MODEL_NAME = "gate"
INPUT_TOPIC = "/camera/front"
OUTPUT_TOPIC = "/auv_vision/model_debug"

class ModelDebugger(Node):
    def __init__(self):
        super().__init__("model_debugger_node")
        ws_root = os.path.abspath(__file__) # File is running from builded ros2 nodes folder
        for i in range(7): ws_root = os.path.dirname(ws_root)

        self.model_path = os.path.join(ws_root, f"src/nautronics_auv/auv_vision/model/{MODEL_NAME}.pt")
        self.camera_topic = INPUT_TOPIC
        self.conf_threshold = 0.5 # %50 confidence
        self.bridge = CvBridge()
        self.publisher_ = self.create_publisher(Image, OUTPUT_TOPIC, 10)
        self.subscription = self.create_subscription(Image, self.camera_topic, self.image_callback, 10)
        
        self.get_logger().info(f"Loading model: {self.model_path}...")
        try:
            self.model = YOLO(self.model_path)
            self.get_logger().info("Model loaded successfully")
        except Exception as e:
            self.get_logger().error(f"Failed loading model! Error: {e}")
            return

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            results = self.model(frame, conf=self.conf_threshold, verbose=False)
            annotated_frame = results[0].plot() # Draw bounding box, score and keypoints
            out_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding="bgr8")
            self.publisher_.publish(out_msg)

        except Exception as e:
            self.get_logger().error(f"Error in processing image! Error: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = ModelDebugger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Test finished, shutdowning the node...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()