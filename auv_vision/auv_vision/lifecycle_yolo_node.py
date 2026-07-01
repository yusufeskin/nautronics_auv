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

class UniversalYoloLifecycleNode(LifecycleNode):
    def __init__(self):
        super().__init__('universal_yolo_node')
        self.get_logger().info('YOLO Lifecycle node başlatıldı.')    
        self.declare_parameter('model_name', 'baris.engine')
        self.declare_parameter('model_type', 'keypoint')
        self.declare_parameter('image_topic', '/camera/camera/color/image_raw')
        #parameters
        self.declare_parameter('ema_alpha', 0.60)
        self.declare_parameter('distance_gate_threshold', 35.0)
        self.declare_parameter('miss_frames_limit', 10)
        
        self.model = None
        self.bridge = CvBridge()
        self.class_names = {}
        self.keypoint_history: dict = {}
        
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
        
        try:
            self.model = YOLO(model_path, task=yolo_task)
            self.class_names = self.model.names
        except Exception as e:
            self.get_logger().error(f"Model yüklenemedi: {e}")
            return TransitionCallbackReturn.ERROR

        self.get_logger().info('TensorRT motoru ısıtılıyor...')
        dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model(dummy_image, verbose=False, conf=0.5)

        self.image_sub = self.create_subscription(
            Image, self.image_topic, self.image_callback, qos_profile_sensor_data)

        return super().on_activate(state)

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info('Deaktive ediliyor (Deactivating)...')
        self.destroy_subscription(self.image_sub)
        self.image_sub = None
        self.keypoint_history.clear()
        
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

    def apply_ema_filter(self, detections_msg):
        if self.model_type != 'keypoint':
            return

        alpha      = self.get_parameter('ema_alpha').get_parameter_value().double_value
        gate_px    = self.get_parameter('distance_gate_threshold').get_parameter_value().double_value
        miss_limit = self.get_parameter('miss_frames_limit').get_parameter_value().integer_value

        seen_ids = set()

        for det in detections_msg.detections:
            cls_id = det.class_id
            seen_ids.add(cls_id)

            raw_pts = np.array(
                [[det.keypoints[i].x, det.keypoints[i].y] for i in range(4)],
                dtype=np.float32
            )

            if cls_id in self.keypoint_history:
                prev_pts = self.keypoint_history[cls_id]['pts']

                max_dist = np.max(np.linalg.norm(raw_pts - prev_pts, axis=1))
                if max_dist > gate_px:
                    self.keypoint_history[cls_id]['miss'] += 1
                    
                    if self.keypoint_history[cls_id]['miss'] >= miss_limit:
                        self.get_logger().info(f'[EMA] cls={cls_id} uzun sure uzak kaldi. Yeni konuma kilitleniyor.')
                        smoothed = raw_pts
                        self.keypoint_history[cls_id] = {'pts': smoothed.copy(), 'miss': 0}
                    else:
                        smoothed = prev_pts
                        self.get_logger().debug(
                            f'[EMA] cls={cls_id} gated (jump={max_dist:.1f}px > {gate_px}px)'
                        )
                else:
                    smoothed = alpha * raw_pts + (1.0 - alpha) * prev_pts
                    self.keypoint_history[cls_id]['pts']  = smoothed
                    self.keypoint_history[cls_id]['miss'] = 0
            else:
                smoothed = raw_pts
                self.keypoint_history[cls_id] = {'pts': smoothed.copy(), 'miss': 0}

            for i in range(4):
                det.keypoints[i].x = float(smoothed[i][0])
                det.keypoints[i].y = float(smoothed[i][1])

        # Increment miss counter for classes absent this frame
        for cls_id in list(self.keypoint_history.keys()):
            if cls_id not in seen_ids:
                self.keypoint_history[cls_id]['miss'] += 1
                if self.keypoint_history[cls_id]['miss'] >= miss_limit:
                    self.get_logger().info(
                        f'[EMA] cls={cls_id} evicted after {miss_limit} missed frames.'
                    )
                    del self.keypoint_history[cls_id]

    def image_callback(self, msg):
        if self.model is None:
            return

        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        frame = cv2.resize(frame, (640, 640))
        results = self.model(frame, verbose=False, conf=0.5)

        detections_msg = get_detections(results[0], msg.header, self.class_names, self.model_type)
        # self.apply_ema_filter(detections_msg)
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