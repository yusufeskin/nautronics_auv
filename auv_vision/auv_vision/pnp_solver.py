#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from sensor_msgs.msg import CameraInfo, Image, CompressedImage
from auv_interfaces.msg import DetectionArray
from scipy.spatial.transform import Rotation as R
from .object_config import OBJECT_REGISTRY
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge
from utils.debug_helper import draw_debug_pnp, build_compressed_msg


class PnPSolverNode(Node):
    def __init__(self):
        super().__init__('pnp_solver_node')
        self.get_logger().info('PnP Solver başlatıldı | Algoritma: ITERATIVE')

        self.camera_matrix = None
        self.dist_coeffs = None
        
        self.latest_image = None
        self.bridge = CvBridge()

        self.object_library = {}
        self.load_object_config()

        self.create_subscription(CameraInfo, '/camera/camera/color/camera_info', self.camera_info_cb, 10)
        self.create_subscription(Image, '/camera/camera/color/image_raw', self.image_cb, qos_profile=qos_profile_sensor_data)
        self.create_subscription(DetectionArray, '/yolo_detections', self.yolo_cb, qos_profile=qos_profile_sensor_data)
        
        self.pose_publisher = self.create_publisher(DetectionArray, '/object_3d_poses', 10)
        self.debug_publisher = self.create_publisher(CompressedImage, '/pnp_debug_image/compressed', qos_profile=qos_profile_sensor_data)

    def image_cb(self, msg):
        self.latest_image = msg

    def load_object_config(self):
        for cls_id, props in OBJECT_REGISTRY.items():
            self.object_library[cls_id] = np.array(props['points_3d'], dtype=np.float32)

    def camera_info_cb(self, msg):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k, dtype=np.float64).reshape((3, 3))
            self.dist_coeffs = np.array(msg.d, dtype=np.float64)

            # scale_x = 640.0 / msg.width
            # scale_y = 640.0 / msg.height
            # self.camera_matrix[0, 0] *= scale_x  # fx
            # self.camera_matrix[1, 1] *= scale_y  # fy
            # self.camera_matrix[0, 2] *= scale_x  # cx
            # self.camera_matrix[1, 2] *= scale_y  # cy

            self.get_logger().info(
                f'Kamera matrisi alındı ve scale edildi: {msg.width}x{msg.height} -> 640x640'
            )

    def yolo_cb(self, msg: DetectionArray):
        if self.camera_matrix is None or self.dist_coeffs is None:
            self.get_logger().warn('Kamera Info bekleniyor, PnP atlandı.', throttle_duration_sec=2.0)
            return

        for det in msg.detections:
            cls_id = det.class_id

            if not det.class_name:
                det.class_name = OBJECT_REGISTRY.get(cls_id, {}).get('name', 'unknown')

            keypoints_2d = [[kp.x, kp.y] for kp in det.keypoints[:4]]
            image_2d_points = np.array(keypoints_2d, dtype=np.float32)
            object_3d_points = self.object_library.get(cls_id)

            if (
                object_3d_points is not None
                and len(image_2d_points) == 4
                and len(image_2d_points) == len(object_3d_points)
            ):
                try:
                    success, rvec, tvec = cv2.solvePnP(
                        object_3d_points,
                        image_2d_points,
                        self.camera_matrix,
                        self.dist_coeffs,
                        flags=cv2.SOLVEPNP_ITERATIVE
                    )

                    if success:
                        det.distance = float(tvec[2][0])
                        rmat, _ = cv2.Rodrigues(rvec)
                        yaw = float(R.from_matrix(rmat).as_euler('xyz', degrees=True)[1])
                        det.yaw_angle = yaw
                    else:
                        det.distance = -1.0
                        det.yaw_angle = 0.0

                except Exception as e:
                    self.get_logger().error(f'PnP Hatası (Class {cls_id}): {e}', throttle_duration_sec=1.0)
                    det.distance = -1.0
                    det.yaw_angle = 0.0
            else:
                det.distance = -1.0
                det.yaw_angle = 0.0

        self.pose_publisher.publish(msg)

        if self.latest_image is not None:
            try:
                frame = self.bridge.imgmsg_to_cv2(self.latest_image, "bgr8")
                debug_frame = draw_debug_pnp(frame, None, msg, 'keypoint')
                debug_msg = build_compressed_msg(debug_frame, self.latest_image.header)
                if debug_msg:
                    self.debug_publisher.publish(debug_msg)
            except Exception as e:
                self.get_logger().error(f'Debug image yayınlama hatası: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = PnPSolverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()