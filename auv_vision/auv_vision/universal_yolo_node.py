#!/usr/bin/env python3
import os
import gc
import traceback

import rclpy
import numpy as np
from rclpy.lifecycle import LifecycleNode, LifecycleState, TransitionCallbackReturn
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
from ultralytics import YOLO
from ament_index_python.packages import get_package_share_directory, PackageNotFoundError
from auv_interfaces.msg import DetectionArray 
from my_py_pkg.debug_helper import draw_debug, build_compressed_msg
from my_py_pkg.yolo_helper import get_detections


class UniversalYoloLifecycleNode(LifecycleNode):
    def __init__(self):
        super().__init__('universal_yolo_node')
        self.get_logger().info('YOLO Lifecycle node başlatıldı.')

        self.declare_parameter('model_name', '/home/zeynep/Desktop/resimler/runs/pose/gate_pose_model/weights/best.pt')
        self.declare_parameter('model_type', 'keypoint')
        self.declare_parameter('image_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('tracker_config', 'botsort.yaml')


        self.model = None
        self.model_type = 'keypoint'
        self.tracker_path = 'botsort.yaml'
        self.bridge = CvBridge()
        self.class_names = {}

        self.image_sub = None
        self.target_publisher = None
        self.debug_publisher = None




    
    def _share_dir(self):
        pkg = self.get_parameter('package_name').get_parameter_value().string_value
        try:
            return get_package_share_directory(pkg)
        except PackageNotFoundError:
            self.get_logger().warn(f"'{pkg}' paketi bulunamadı (share dizini yok).")
            return None

    def _resolve_model_path(self, model_name: str, share_dir) -> str:
        model_name = os.path.expanduser(model_name)
        if os.path.isabs(model_name):
            path = model_name
        elif share_dir is not None:
            path = os.path.join(share_dir, 'model', model_name)
        else:
            path = model_name
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Model dosyası bulunamadı: {path}")
        return path

    def _resolve_tracker_path(self, tracker_file: str, share_dir) -> str:
        tracker_file = os.path.expanduser(tracker_file)
        if os.path.isabs(tracker_file) and os.path.isfile(tracker_file):
            return tracker_file
        if share_dir is not None:
            candidate = os.path.join(share_dir, 'config', tracker_file)
            if os.path.isfile(candidate):
                return candidate

        self.get_logger().warn(
            f"Tracker config paket içinde bulunamadı, Ultralytics varsayılanı kullanılacak: {tracker_file}")
        return tracker_file






    # LİFECYCLE BAŞLANGIÇ

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info('Yapılandırılıyor (Configuring)...')
        det_topic = self.get_parameter('detections_topic').get_parameter_value().string_value
        self.target_publisher = self.create_lifecycle_publisher(
            DetectionArray, det_topic, qos_profile=qos_profile_sensor_data)
        self.debug_publisher = self.create_lifecycle_publisher(
            CompressedImage, '/yolo_debug_image/compressed', qos_profile=qos_profile_sensor_data)
        return TransitionCallbackReturn.SUCCESS


    

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info('Aktive ediliyor (Activating)...')
        try:
            share_dir = self._share_dir()

            model_name = self.get_parameter('model_name').get_parameter_value().string_value
            self.model_type = self.get_parameter('model_type').get_parameter_value().string_value
            image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
            tracker_file = self.get_parameter('tracker_config').get_parameter_value().string_value

            model_path = self._resolve_model_path(model_name, share_dir)
            self.tracker_path = self._resolve_tracker_path(tracker_file, share_dir)
            yolo_task = 'pose' if self.model_type == 'keypoint' else (
                'obb' if self.model_type == 'obb' else 'detect')

            self.get_logger().info(f'Model yükleniyor: {model_path} (task={yolo_task})')
            self.model = YOLO(model_path, task=yolo_task)
            self.class_names = self.model.names

            self.get_logger().info('Model ısıtılıyor...')
            dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
            self.model(dummy_image, verbose=False, conf=0.5)

            self.image_sub = self.create_subscription(
                Image, image_topic, self.image_callback, qos_profile_sensor_data)
        except Exception as e:
            self.get_logger().error(f'Aktivasyon başarısız: {e}\n{traceback.format_exc()}')
            self.model = None

            return TransitionCallbackReturn.ERROR

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

    # LİFECYCLE BİTİŞ




    
    def image_callback(self, msg):
        model = self.model
        if model is None:
            return
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            results = model.track(
                frame,
                persist=True,
                tracker=self.tracker_path,
                verbose=False,
                conf=0.5,
            )

            detections_msg = get_detections(results[0], msg.header, self.class_names, self.model_type)
            self.target_publisher.publish(detections_msg)

            debug_frame = draw_debug(frame, results, detections_msg, self.model_type)
            debug_msg = build_compressed_msg(debug_frame, msg.header)
            if debug_msg:
                self.debug_publisher.publish(debug_msg)
        except Exception as e:
            self.get_logger().error(f'image_callback hatası: {e}\n{traceback.format_exc()}')


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

