import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
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
        model_path = os.path.join(pkg_share_dir, 'model', 'best.pt')
        self.model = YOLO(model_path)
        self.camera_matrix = None
        self.dist_coeffs = None


        self.object_library = {}
        self.load_object_config()

        self.create_subscription(CameraInfo, '/front_camera/camera_info', self.camera_info_callback, 10)
        self.create_subscription(Image, '/front_camera/image_raw', self.image_callback, 10)
        self.target_publisher = self.create_publisher(DetectionArray, '/yolo_detections', 10)
        self.debug_publisher = self.create_publisher(Image, '/yolo_debug_image', 10)
        self.bridge = CvBridge()

    def camera_info_callback(self, msg):
        if self.camera_matrix is not None:
            return
            
        self.camera_matrix = np.array(msg.k, dtype=np.float64).reshape((3, 3))
        self.dist_coeffs = np.array(msg.d, dtype=np.float64)
        self.get_logger().info('camerainfo geldi.')    

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
            self.get_logger().warn('CameraInfo bekleniyor, goruntu atlandi...')
            return
        # self.get_logger().warn('goruntu aldım...')  # for debugging
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
            if len(image_2d_points) != len(object_3d_points) or len(image_2d_points) < 4:
                continue

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
        
        #ros2 run rqt_image_view rqt_image_view to see what our model is recognizing
        debug_frame = r.plot()
        debug_msg = self.bridge.cv2_to_imgmsg(debug_frame, "bgr8")
        debug_msg.header = msg.header
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