import rclpy
import cv2
import numpy as np
import os
import gc
from rclpy.lifecycle import LifecycleNode, LifecycleState, TransitionCallbackReturn
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
from ultralytics import YOLO
from ament_index_python.packages import get_package_share_directory 
from auv_interfaces.msg import DetectionArray 
from utils.debug_helper import draw_debug, build_compressed_msg
from utils.yolo_helper import get_detections

class UniversalYoloLifecycleNode(LifecycleNode):
    def __init__(self):
        super().__init__('universal_yolo_node')
        self.get_logger().info('YOLO Lifecycle node başlatıldı.')    
        self.declare_parameter('model_name', 'best.engine')
        self.declare_parameter('model_type', 'bbox')
        self.model = None
        self.bridge = CvBridge()
        self.class_names = {}
        
        self.image_sub = None
        self.target_publisher = None
        self.debug_publisher = None

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info('Yapılandırılıyor (Configuring)...')
        self.target_publisher = self.create_lifecycle_publisher(
            DetectionArray, '/yolo_detections', qos_profile=qos_profile_sensor_data)
        self.debug_publisher = self.create_lifecycle_publisher(
            CompressedImage, '/yolo_debug_image/compressed', qos_profile=qos_profile_sensor_data)
        return TransitionCallbackReturn.SUCCESS
    

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info('Aktive ediliyor (Activating)...')
        
        pkg_share_dir = get_package_share_directory('auv_vision')
        model_name = self.get_parameter('model_name').get_parameter_value().string_value
        model_path = os.path.join(pkg_share_dir, 'model', model_name)
        self.model_type = self.get_parameter('model_type').get_parameter_value().string_value

        
        try:
            self.model = YOLO(model_path)
            self.class_names = self.model.names
        except Exception as e:
            self.get_logger().error(f"Model yüklenemedi: {e}")
            return TransitionCallbackReturn.ERROR

        self.get_logger().info('TensorRT motoru ısıtılıyor...')
        dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model(dummy_image, verbose=False, conf=0.5)

        self.image_sub = self.create_subscription(
            Image, '/front_camera/image_raw', self.image_callback, qos_profile_sensor_data)

        return super().on_activate(state)

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info('Deaktive ediliyor (Deactivating)...')
        self.destroy_subscription(self.image_sub)
        self.image_sub = None
        
        del self.model
        del self.class_names
        self.model = None
        gc.collect()

        return super().on_deactivate(state)

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.destroy_publisher(self.target_publisher)
        self.destroy_publisher(self.debug_publisher)
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        return TransitionCallbackReturn.SUCCESS

    def image_callback(self, msg):
        if self.model is None:
            return

        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        frame = cv2.resize(frame, (640, 640))
        results = self.model(frame, verbose=False, conf=0.5)


        detections_msg = get_detections(results[0], msg.header, self.class_names, self.model_type)
        self.target_publisher.publish(detections_msg)

        debug_frame = draw_debug(frame, results, detections_msg, self.model_type)
        debug_msg = build_compressed_msg(debug_frame, msg.header)
        if debug_msg:
            self.debug_publisher.publish(debug_msg)
def main(args=None):
    rclpy.init(args=args)
    node = UniversalYoloLifecycleNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()

if __name__ == '__main__':
    main()