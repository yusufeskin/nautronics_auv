import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
import os
from ament_index_python.packages import get_package_share_directory
from auv_interfaces.msg import DetectedObject, DetectionArray
from geometry_msgs.msg import Point
from .object_config import OBJECT_REGISTRY
from scipy.spatial.transform import Rotation as R


class MultiObjectPnPNode(Node):
    def __init__(self):
        super().__init__('multi_object_pnp_node')
        self.get_logger().info('object_keypoint_detector başladı.')
        pkg_share_dir = get_package_share_directory('auv_vision')

        model_path = os.path.join(pkg_share_dir, 'model', 'best.engine')
        self.model = YOLO(model_path, task='pose')

        self.camera_matrix = None
        self.dist_coeffs = None
        # solvePnP için scale edilmiş camera matrix (640x640)
        self.camera_matrix_scaled = None

        self.object_library = {}
        self.load_object_config()

        self.create_subscription(CameraInfo, '/front_camera/camera_info', self.camera_info_callback, 10)
        self.create_subscription(Image, '/front_camera/image_raw', self.image_callback, 10)

        self.target_publisher = self.create_publisher(DetectionArray, '/yolo_detections', 10)
        self.debug_publisher = self.create_publisher(CompressedImage, '/yolo_debug_image/compressed', 10)

        self.bridge = CvBridge()

        # TensorRT warmup
        self.get_logger().info('tensorrt warmup başlıyor...')
        dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model(dummy_image, verbose=False, conf=0.5)
        self.get_logger().info('model ready.')

    def camera_info_callback(self, msg):
        if self.camera_matrix is not None:
            return

        self.camera_matrix = np.array(msg.k, dtype=np.float64).reshape((3, 3))
        self.dist_coeffs = np.array(msg.d, dtype=np.float64)

        # Kamera matrisini 640x640'a scale et
        # Orijinal görüntü boyutları camera_info'dan alınır
        orig_w = msg.width   # 1280
        orig_h = msg.height  # 720
        scale_x = 640.0 / orig_w
        scale_y = 640.0 / orig_h

        self.camera_matrix_scaled = self.camera_matrix.copy()
        self.camera_matrix_scaled[0, 0] *= scale_x  # fx
        self.camera_matrix_scaled[1, 1] *= scale_y  # fy
        self.camera_matrix_scaled[0, 2] *= scale_x  # cx
        self.camera_matrix_scaled[1, 2] *= scale_y  # cy

        self.get_logger().info(
            f'CameraInfo alındı. Orijinal: {orig_w}x{orig_h}, '
            f'scale: ({scale_x:.3f}, {scale_y:.3f})'
        )

    def load_object_config(self):
        for cls_id, props in OBJECT_REGISTRY.items():
            name = props['name']
            points_3d = np.array(props['points_3d'], dtype=np.float32)
            self.object_library[cls_id] = {
                'points': points_3d,
                'name': name
            }

    def image_callback(self, msg):
        if self.camera_matrix is None or self.dist_coeffs is None:
            self.get_logger().warn('CameraInfo bekleniyor, görüntü atlandı...')
            return

        # Orijinal frame'i al, 640x640'a resize et
        frame_orig = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        frame = cv2.resize(frame_orig, (640, 640))
        height, width = frame.shape[:2]  # 640, 640

        results = self.model(frame, verbose=False, conf=0.5)
        r = results[0]

        det_array = DetectionArray()
        det_array.header = msg.header
        det_array.detections = []

        boxes = r.boxes
        kpts_batch = r.keypoints.xy.cpu().numpy()

        text_y_offset = 30

        for box, kpts in zip(boxes, kpts_batch):
            cls_id = int(box.cls[0])

            obj_msg = DetectedObject()
            obj_msg.class_id = cls_id
            obj_msg.class_name = self.object_library[cls_id]['name']
            obj_msg.confidence = float(box.conf[0])

            # Box koordinatları zaten 640x640 frame'e göre
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, obj_msg.class_name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            keypoints_2d = []
            for index in range(min(len(kpts), 4)):
                x_val = float(kpts[index][0])
                y_val = float(kpts[index][1])

                obj_msg.keypoints[index].x = x_val
                obj_msg.keypoints[index].y = y_val
                obj_msg.keypoints[index].z = 0.0
                keypoints_2d.append([x_val, y_val])

                cv2.circle(frame, (int(x_val), int(y_val)), 5, (0, 255, 0), -1)
                cv2.putText(frame, str(index), (int(x_val) + 5, int(y_val) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            object_3d_points = self.object_library[cls_id]['points']
            image_2d_points = np.array(keypoints_2d, dtype=np.float32)

            if len(image_2d_points) != len(object_3d_points) or len(image_2d_points) < 4:
                continue

            try:
                success, rvec, tvec = cv2.solvePnP(
                    object_3d_points,
                    image_2d_points,
                    self.camera_matrix_scaled,  # scale edilmiş matrix kullan
                    self.dist_coeffs,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
                if success:
                    obj_msg.distance = float(tvec[2][0])
                    rmat, _ = cv2.Rodrigues(rvec)
                    rot = R.from_matrix(rmat)
                    euler = rot.as_euler('xyz', degrees=False)
                    yaw = euler[2]
                    obj_msg.yaw_angle = float(yaw)

                    yaw_deg = np.degrees(yaw)
                    info_text = f"{obj_msg.class_name} | Dist: {obj_msg.distance:.2f}m | Yaw: {yaw_deg:.1f}deg"

                    (text_w, text_h), _ = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    text_x = width - text_w - 15

                    cv2.rectangle(frame,
                                  (text_x - 5, text_y_offset - text_h - 5),
                                  (text_x + text_w + 5, text_y_offset + 5),
                                  (0, 0, 0), -1)
                    cv2.putText(frame, info_text, (text_x, text_y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                    text_y_offset += text_h + 15
                else:
                    obj_msg.distance = -1.0
                    obj_msg.yaw_angle = 0.0
            except Exception as e:
                self.get_logger().warn(f'solvePnP hatası: {e}')
                obj_msg.distance = -1.0
                obj_msg.yaw_angle = 0.0

            det_array.detections.append(obj_msg)

        self.target_publisher.publish(det_array)

        # 640x640 frame'i direkt encode et
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
        success, encoded_image = cv2.imencode('.jpg', frame, encode_param)

        if success:
            debug_msg = CompressedImage()
            debug_msg.header = msg.header
            debug_msg.format = "jpeg"
            debug_msg.data = np.array(encoded_image).tobytes()
            self.debug_publisher.publish(debug_msg)


def main(args=None):
    rclpy.init(args=args)
    node = MultiObjectPnPNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()