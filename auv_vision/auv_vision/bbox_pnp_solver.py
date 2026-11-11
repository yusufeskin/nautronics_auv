#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from sensor_msgs.msg import CameraInfo
from auv_interfaces.msg import DetectionArray
from scipy.spatial.transform import Rotation as R
from .object_config import OBJECT_REGISTRY 

class BBoxPnPSolverNode(Node):
    def __init__(self):
        super().__init__('bbox_pnp_solver_node')
        self.get_logger().info('BBox PnP Solver Node başlatıldı.')
        
        self.declare_parameter('info_topic', '/camera/camera_info')
        self.info_topic = self.get_parameter('info_topic').get_parameter_value().string_value
        self.camera_matrix = None
        self.dist_coeffs = None
        self.object_library = {}
        self.load_object_config()
        
        self.create_subscription(CameraInfo, self.info_topic, self.camera_info_cb, 10)
        self.create_subscription(DetectionArray, '/yolo_detections', self.yolo_cb, 10)
        
        self.pose_publisher = self.create_publisher(DetectionArray, '/object_3d_poses_from_bbox', 10)

    def load_object_config(self):
        for cls_id, props in OBJECT_REGISTRY.items():
            self.object_library[cls_id] = np.array(props['points_3d'], dtype=np.float32)

    def camera_info_cb(self, msg):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k, dtype=np.float64).reshape((3, 3))
            self.dist_coeffs = np.array(msg.d, dtype=np.float64)

            scale_x = 640.0 / msg.width
            scale_y = 640.0 / msg.height

            self.camera_matrix[0, 0] *= scale_x  # fx
            self.camera_matrix[1, 1] *= scale_y  # fy
            self.camera_matrix[0, 2] *= scale_x  # cx
            self.camera_matrix[1, 2] *= scale_y  # cy

            self.get_logger().info(f'Kamera matrisi ayarlandı ve 640x640 için ölçeklendi.')

    def yolo_cb(self, msg: DetectionArray):
        if self.camera_matrix is None or self.dist_coeffs is None:
            self.get_logger().warn('Kamera Info bekleniyor, BBox PnP atlandı.', throttle_duration_sec=2.0)
            return

        for det in msg.detections:
            cls_id = det.class_id
            
            if det.bbox_width <= 0.0 or det.bbox_height <= 0.0:
                continue

            if not det.class_name:
                det.class_name = OBJECT_REGISTRY.get(cls_id, {}).get('name', 'unknown')

            cx = det.bbox_center_x
            cy = det.bbox_center_y
            w = det.bbox_width
            h = det.bbox_height

            image_2d_points = np.array([
                [cx - w / 2.0, cy - h / 2.0],
                [cx + w / 2.0, cy - h / 2.0],
                [cx + w / 2.0, cy + h / 2.0],
                [cx - w / 2.0, cy + h / 2.0]
            ], dtype=np.float32)
            
            object_3d_points = self.object_library.get(cls_id)

            if object_3d_points is not None and len(object_3d_points) == 4:
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
                        yaw = float(R.from_matrix(rmat).as_euler('xyz', degrees=False)[2])
                        det.yaw_angle = yaw
                    else: 
                        det.distance = -1.0
                        det.yaw_angle = 0.0
                        
                except Exception as e:
                    self.get_logger().error(f"BBox PnP Hatası (Class {cls_id}): {e}", throttle_duration_sec=1.0)
                    det.distance = -1.0
                    det.yaw_angle = 0.0
            else:
                det.distance = -1.0
                det.yaw_angle = 0.0

        self.pose_publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = BBoxPnPSolverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()

if __name__ == '__main__':
    main()