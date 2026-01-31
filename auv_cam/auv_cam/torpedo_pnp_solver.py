import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Vector3 
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
from scipy.spatial.transform import Rotation as R
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import math
import os
from ament_index_python.packages import get_package_share_directory 

# Custom Message Import
from auv_interfaces.msg import TorpedoTarget 

class TorpedoPnPNode(Node):
    def __init__(self):
        super().__init__('torpedo_pnp_node')

        # --- SETTINGS ---
        self.target_width = 0.6   # Meters (Real world width)
        self.target_height = 0.6  # Meters (Real world height)
        self.norm_focal_length = 0.866 # Simulation focal length multiplier
        
        # --- MODEL LOADING ---
        # Dynamically find the model path using ROS 2 package share directory
        pkg_share_dir = get_package_share_directory('auv_cam')
        model_path = os.path.join(pkg_share_dir, 'model', 'best.pt')
        
        self.get_logger().info(f"Model Path Found: {model_path}")
        self.model = YOLO(model_path)
        
        # --- QOS SETTINGS ---
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # --- SUBSCRIBERS & PUBLISHERS ---
        self.img_topic = '/camera/front'
        self.subscription = self.create_subscription(
            Image, self.img_topic, self.image_callback, qos_profile) 

        self.target_publisher = self.create_publisher(TorpedoTarget, '/auv/torpedo_data', 10)

        self.bridge = CvBridge()
        self.camera_matrix = None
        self.dist_coeffs = np.zeros((4,1))
        
        # --- 3D OBJECT POINTS ---
        # Defining corners of the target relative to top-left (0,0,0)
        w = self.target_width 
        h = self.target_height 
        self.object_points = np.array([
            [0, 0, 0], [w, 0, 0], [w, h, 0], [0, h, 0]  
        ], dtype=np.float32)

        self.get_logger().info("PnP Node Ready.")

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            height, width, _ = frame.shape
        except Exception:
            return

        # Initialize Camera Matrix (if not done)
        if self.camera_matrix is None:
            focal_length_pixel = self.norm_focal_length * width
            center_x = width / 2.0
            center_y = height / 2.0
            self.camera_matrix = np.array([
                [focal_length_pixel, 0, center_x],
                [0, focal_length_pixel, center_y],
                [0, 0, 1]
            ], dtype=np.float32)

        # YOLO Inference
        results = self.model(frame, verbose=False)

        if len(results) > 0 and len(results[0].boxes) > 0:
            box = results[0].boxes[0]
            xyxy = box.xyxy[0].cpu().numpy()
            x_min, y_min, x_max, y_max = xyxy
            
            # Calculate Center Pixel
            pixel_center_x = (x_min + x_max) / 2.0
            pixel_center_y = (y_min + y_max) / 2.0

            # 2D Image Points
            image_points = np.array([
                [x_min, y_min], [x_max, y_min], 
                [x_max, y_max], [x_min, y_max]  
            ], dtype=np.float32)

            # Solve PnP
            success, rvec, tvec = cv2.solvePnP(
                self.object_points, image_points, self.camera_matrix, self.dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
            )

            if success:
                self.publish_data(tvec, rvec, pixel_center_x, pixel_center_y, detected=True)
        
        else:
            # Publish zeros if not detected
             self.publish_data(np.zeros((3,1)), np.zeros((3,1)), 0, 0, detected=False)

    def publish_data(self, tvec, rvec, pix_x, pix_y, detected):
        msg = TorpedoTarget()
        
        if detected:
            # --- MATH / TRANSFORMATIONS ---
            
            # 1. Offset Calculation (Shift from corner to center)
            rotation_matrix, _ = cv2.Rodrigues(rvec)
            center_offset_object = np.array([[self.target_width/2.0], [self.target_height/2.0], [0.0]])
            center_offset_cam = np.dot(rotation_matrix, center_offset_object)
            tvec_center = tvec + center_offset_cam
            
            # 2. Axis Conversion (Camera Frame -> AUV Frame)
            # AUV: X=Forward, Y=Left, Z=Up
            auv_x = float(tvec_center[2][0])  # Camera Z -> AUV X
            auv_y = float(-tvec_center[0][0]) # Camera -X -> AUV Y
            auv_z = float(-tvec_center[1][0]) # Camera -Y -> AUV Z
            
            # 3. Rotation Calculation (Euler Angles)
            r = R.from_matrix(rotation_matrix)
            euler = r.as_euler('xyz', degrees=False) 

            # --- POPULATE MESSAGE ---

            # 1. DISTANCE
            msg.distance = float(math.sqrt(auv_x**2 + auv_y**2 + auv_z**2))

            # 2. VECTORS
            msg.position_vec = Vector3(x=auv_x, y=auv_y, z=auv_z)
            msg.orientation_vec = Vector3(x=float(euler[0]), y=float(euler[1]), z=float(euler[2]))
            msg.pixel_vec = Vector3(x=float(pix_x), y=float(pix_y), z=0.0)
            
            # Optional Logging
            # self.get_logger().info(
            #     f"DISTANCE: {msg.distance:.2f}m | PIXEL: {pix_x:.0f},{pix_y:.0f}"
            # )
        else:
            # If target not found, publish zeros
            msg.distance = 0.0
            msg.position_vec = Vector3(x=0.0, y=0.0, z=0.0)
            msg.orientation_vec = Vector3(x=0.0, y=0.0, z=0.0)
            msg.pixel_vec = Vector3(x=0.0, y=0.0, z=0.0)

        self.target_publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = TorpedoPnPNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()