import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from auv_interfaces.msg import DetectedPipe
from geometry_msgs.msg import Point
from ament_index_python.packages import get_package_share_directory
import os
from ultralytics import YOLO
import numpy as np
import cv2

class PipeDetector(Node):
    def __init__(self):
        super().__init__('pipe_detector')
        self.subscription = self.create_subscription(Image, '/camera/front', self.image_callback, 10)
        self.bridge = CvBridge()
        self.pipe_publisher = self.create_publisher(DetectedPipe, '/pipe_detections', 10)
        
        pkg_share_dir = get_package_share_directory('auv_vision')
        model_path = os.path.join(pkg_share_dir, 'model', 'pipe.pt')
        self.model = YOLO(model_path)
        self.class_names = {0: "red_pipe", 1: "white_pipe"}
        
        self.real_pipe_length = 2.0
        self.cu = 320.0
        self.cv = 240.0
        self.fx = 556.0
        self.fy = 556.0
        
        self.camera_matrix = np.array([
            [self.fx, 0, self.cu],
            [0, self.fy, self.cv],
            [0, 0, 1]
        ], dtype=np.float32)
        self.dist_coeffs = np.zeros((5,), dtype=np.float32) 

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        results = self.model(frame, verbose=False, conf=0.1)
        r = results[0] 

        if r.boxes is None or len(r.boxes) == 0:
            return
        
        kpts_batch = r.keypoints.xy.cpu().numpy() 
        boxes = r.boxes
        
        for box, kpts in zip(boxes, kpts_batch):
            cls_id = int(box.cls[0])
            det_pipe = DetectedPipe()
            det_pipe.class_id = cls_id
            det_pipe.class_name = self.class_names.get(cls_id, "unknown")
            det_pipe.confidence = float(box.conf[0])

            for index in range(min(len(kpts), 2)):
                det_pipe.keypoints[index].x = float(kpts[index][0])
                det_pipe.keypoints[index].y = float(kpts[index][1])
                det_pipe.keypoints[index].z = 0.0

            try:
                if len(kpts) >= 2:
                    pts_distorted = np.array([
                            [[float(kpts[0][0]), float(kpts[0][1])]],
                            [[float(kpts[1][0]), float(kpts[1][1])]]
                        ], dtype=np.float32)

                        
                    pts_undistorted = cv2.undistortPoints(
                            pts_distorted, 
                            self.camera_matrix, 
                            self.dist_coeffs, 
                            P=self.camera_matrix
                        )

                    
                    x1_u, y1_u = pts_undistorted[0][0]
                    x2_u, y2_u = pts_undistorted[1][0]
                    pixel_distance = np.sqrt((x2_u - x1_u)**2 + (y2_u - y1_u)**2)

                
                    if pixel_distance > 0: 
                            distance = (self.real_pipe_length * self.fx) / pixel_distance
                            det_pipe.distance = float(distance)
                    else: 
                            det_pipe.distance = -1.0
                else:
                     det_pipe.distance = -1.0
            except Exception as e:
                self.get_logger().error(f"err: {e}")
                det_pipe.distance = -1.0
            self.pipe_publisher.publish(det_pipe)
            
def main(args=None):
    rclpy.init(args=args)
    node = PipeDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()