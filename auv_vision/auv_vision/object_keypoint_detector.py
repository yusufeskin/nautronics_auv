import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
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
        
        pkg_share_dir = get_package_share_directory('auv_vision')
        model_path = os.path.join(pkg_share_dir, 'model', 'Multimodel.pt')
        self.model = YOLO(model_path)

        self.cu = 320.0
        self.cv = 240.0
        self.fx = 556.0
        self.fy = 556.0
        
        self.camera_matrix = np.array([
            [self.fx, 0, self.cu],
            [0, self.fy, self.cv],
            [0, 0, 1]
        ], dtype=np.float32)
        
        self.dist_coeffs = np.zeros((4,1))
        self.object_library = {}
        self.load_object_config()

        self.create_subscription(Image, '/camera/front', self.image_callback, 10) 
        self.target_publisher = self.create_publisher(DetectionArray, '/yolo_detections', 10)
        self.bridge = CvBridge()

    def load_object_config(self):
        for cls_id, props in OBJECT_REGISTRY.items():
            w = props['width']
            h = props['height']
            name = props['name']
            
            points_3d = np.array([
                [0, 0, 0],
                [w, 0, 0],
                [w, h, 0],
                [0, h, 0]
            ], dtype=np.float32)
            
            self.object_library[cls_id] = {
                'points': points_3d,
                'name': name
            }

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        results = self.model(frame, verbose=False, conf=0.5)
        r = results[0] 
        det_array = DetectionArray()
        det_array.header = msg.header
        det_array.detections = []
        boxes = r.boxes
        kpts_batch = r.keypoints.xy.cpu().numpy() 

        for box, kpts in zip(boxes, kpts_batch):
            cls_id = int(box.cls[0])

            obj_msg = DetectedObject()
            obj_msg.class_id = cls_id
            obj_msg.class_name = self.object_library[cls_id]['name']
            obj_msg.confidence = float(box.conf[0])
            
            keypoints_2d = []
            # index = keypoint index
            for index in range(min(len(kpts), 4)):
                point = Point(x=float(kpts[index][0]), y=float(kpts[index][1]), z=0.0)
                obj_msg.keypoints[index].x = point.x
                obj_msg.keypoints[index].y = point.y
                obj_msg.keypoints[index].z = 0.0
                keypoints_2d.append([float(kpts[index][0]), float(kpts[index][1])])

            object_3d_points = self.object_library[cls_id]['points']
            image_2d_points = np.array(keypoints_2d, dtype=np.float32)

            # distance -1 = no detection
            try:
                success, rvec, tvec = cv2.solvePnP(
                    object_3d_points,
                    image_2d_points,
                    self.camera_matrix, 
                    self.dist_coeffs, 
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
                if success: 
                    obj_msg.distance = float(tvec[2][0])
                    rmat, _ = cv2.Rodrigues(rvec)
                    #if i face with an issue i will check below(note for myself) note2: i faced and i changed now its working(2 hours later)
                    rot = R.from_matrix(rmat)
                    euler = rot.as_euler('xyz', degrees=False)
                    yaw = euler[2]
                    obj_msg.yaw_angle = float(yaw)
                else: 
                    obj_msg.distance = -1.0
                    obj_msg.yaw_angle = 0.0
            except Exception:
                obj_msg.distance = -1.0
                obj_msg.yaw_angle = 0.0

            det_array.detections.append(obj_msg)

        self.target_publisher.publish(det_array)
            

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