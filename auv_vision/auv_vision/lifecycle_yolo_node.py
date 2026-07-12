#!/usr/bin/env python3
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
from utils.ema_filter import EMAFilter
from utils.track_manager import TrackManager

class UniversalYoloLifecycleNode(LifecycleNode):
    def __init__(self):
        super().__init__('universal_yolo_node')
        self.get_logger().info('YOLO Lifecycle node başlatıldı.')    
        self.declare_parameter('model_name', 'july11_gate.pt')
        self.declare_parameter('model_type', 'keypoint')
        self.declare_parameter('image_topic', '/image_raw')
        #ema parameters
        self.declare_parameter('ema_alpha', 0.60)
        self.declare_parameter('distance_gate_threshold', 35.0)
        #ema and track
        self.declare_parameter('miss_frames_limit', 30)
        #track parameters
        self.declare_parameter('min_hits', 5)
        self.declare_parameter('tracker_config', 'botsort.yaml')
        
        self.model = None
        self.bridge = CvBridge()
        self.class_names = {}
        
        self.ema_filter = None
        self.track_manager = None
        
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
        self.image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        yolo_task = 'pose' if self.model_type == 'keypoint' else 'detect'
        tracker_file = self.get_parameter('tracker_config').get_parameter_value().string_value
        self.tracker_path = os.path.join(pkg_share_dir, 'config', tracker_file)

        try:
            self.model = YOLO(model_path, task=yolo_task)
            self.class_names = self.model.names
        except Exception as e:
            self.get_logger().error(f"Model yüklenemedi: {e}")
            return TransitionCallbackReturn.ERROR
        self.tracker = self.get_parameter('tracker_config').get_parameter_value().string_value
        tracker_path = os.path.join(pkg_share_dir, 'config', self.tracker)

        self.get_logger().info('TensorRT motoru ısıtılıyor...')
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        self.model(dummy_image, verbose=False, conf=0.5)

        ema_alpha = self.get_parameter('ema_alpha').get_parameter_value().double_value
        distance_gate = self.get_parameter('distance_gate_threshold').get_parameter_value().double_value
        miss_frames_limit = self.get_parameter('miss_frames_limit').get_parameter_value().integer_value
        min_hits = self.get_parameter('min_hits').get_parameter_value().integer_value
        
        self.ema_filter = EMAFilter(ema_alpha, distance_gate, miss_frames_limit)
        self.track_manager = TrackManager(min_hits, miss_frames_limit)

        self.image_sub = self.create_subscription(
            Image, self.image_topic, self.image_callback, qos_profile_sensor_data)

        return super().on_activate(state)

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info('Deaktive ediliyor (Deactivating)...')
        self.destroy_subscription(self.image_sub)
        self.image_sub = None
        
        self.ema_filter = None
        self.track_manager = None
        
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
        # frame = cv2.resize(frame, (640, 640))
        results = self.model.track(
            frame, 
            persist=True, 
            tracker=self.tracker_path, 
            verbose=False, 
            conf=0.5
        )

        raw_detections_msg = get_detections(results[0], msg.header, self.class_names, self.model_type)

        if self.track_manager:
            raw_detections_msg = self.track_manager.process_tracks(raw_detections_msg, self.model)
        
        # if self.model_type == 'keypoint' and self.ema_filter:
        #     raw_detections_msg = self.ema_filter.apply(raw_detections_msg, logger=self.get_logger())
        
        self.target_publisher.publish(raw_detections_msg)

        debug_frame = draw_debug(frame, results, raw_detections_msg, self.model_type)
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