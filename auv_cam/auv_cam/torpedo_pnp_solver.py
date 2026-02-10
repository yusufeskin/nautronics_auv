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
        self.target_width = 0.6   # Meters
        self.target_height = 0.6  # Meters
        self.norm_focal_length = 0.866 
        
        # --- MODEL LOADING ---
        pkg_share_dir = get_package_share_directory('auv_cam')
        model_path = os.path.join(pkg_share_dir, 'model', 'best.pt')
        self.get_logger().info(f"Model Path: {model_path}")
        self.model = YOLO(model_path)
        
        # --- QOS ---
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.img_topic = '/camera/front'
        self.subscription = self.create_subscription(
            Image, self.img_topic, self.image_callback, qos_profile) 

        self.target_publisher = self.create_publisher(TorpedoTarget, '/auv/torpedo_data', 10)

        self.bridge = CvBridge()
        self.camera_matrix = None
        self.dist_coeffs = np.zeros((4,1))
        
        # --- 3D OBJECT POINTS (Local Frame) ---
        # ORDER MATTERS: Must match the order of keypoints from YOLO model!
        # Assuming YOLO detects: Top-Left -> Top-Right -> Bottom-Right -> Bottom-Left
        w = self.target_width 
        h = self.target_height 
        self.object_points = np.array([
            [0, 0, 0],   # 0: Top-Left
            [w, 0, 0],   # 1: Top-Right
            [w, h, 0],   # 2: Bottom-Right
            [0, h, 0]    # 3: Bottom-Left
        ], dtype=np.float32)

        self.get_logger().info("PnP Node (Keypoint Mode) Ready.")

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            height, width, _ = frame.shape
        except Exception:
            return

        if self.camera_matrix is None:
            focal_length_pixel = self.norm_focal_length * width
            center_x = width / 2.0
            center_y = height / 2.0
            self.camera_matrix = np.array([
                [focal_length_pixel, 0, center_x],
                [0, focal_length_pixel, center_y],
                [0, 0, 1]
            ], dtype=np.float32)

        results = self.model(frame, verbose=False)

        # Check if we have detections AND keypoints
        if len(results) > 0 and results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
            
            # --- KEYPOINT EXTRACTION (THE FIX) ---
            # Get the keypoints of the first detected object
            # shape: (Num_Keypoints, 2) -> e.g., (4, 2)
            kpts = results[0].keypoints.xy[0].cpu().numpy()
            
            # Check if we detected enough points (we need 4 for the rectangle)
            if len(kpts) == 4:
                image_points = np.array(kpts, dtype=np.float32)

                # Calculate Center Pixel (Average of 4 corners is more accurate for rotated objects)
                pixel_center_x = np.mean(image_points[:, 0])
                pixel_center_y = np.mean(image_points[:, 1])

                # Solve PnP using REAL KEYPOINTS
                success, rvec, tvec = cv2.solvePnP(
                    self.object_points, image_points, self.camera_matrix, self.dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
                )

                if success:
                    self.publish_data(tvec, rvec, image_points, pixel_center_x, pixel_center_y, detected=True)
            else:
                # Detected but not 4 points (maybe occlusion?)
                self.publish_data(None, None, None, 0, 0, detected=False)
        
        else:
             self.publish_data(None, None, None, 0, 0, detected=False)

    def publish_data(self, tvec, rvec, image_points, center_x, center_y, detected):
        msg = TorpedoTarget()
        
        if detected:
            # --- LEGACY CALCULATIONS (CENTER 3D) ---
            rotation_matrix, _ = cv2.Rodrigues(rvec)
            
            # Center Calculation (Offset from Top-Left to Center)
            center_offset_object = np.array([[self.target_width/2.0], [self.target_height/2.0], [0.0]])
            center_offset_cam = np.dot(rotation_matrix, center_offset_object)
            tvec_center = tvec + center_offset_cam
            
            # AUV Frame Conversion
            auv_x = float(tvec_center[2][0])
            auv_y = float(-tvec_center[0][0])
            auv_z = float(-tvec_center[1][0])
            
            # Orientation
            r = R.from_matrix(rotation_matrix)
            euler = r.as_euler('xyz', degrees=False) 

            # -- FILL LEGACY FIELDS --
            msg.distance = float(math.sqrt(auv_x**2 + auv_y**2 + auv_z**2))
            msg.position_vec = Vector3(x=auv_x, y=auv_y, z=auv_z)
            msg.orientation_vec = Vector3(x=float(euler[0]), y=float(euler[1]), z=float(euler[2]))
            msg.pixel_vec = Vector3(x=float(center_x), y=float(center_y), z=0.0)

            # -- FILL PIXEL CORNERS (FROM KEYPOINTS) --
            # Assuming standard order: TL, TR, BR, BL
            msg.pixel_top_left = Vector3(x=float(image_points[0][0]), y=float(image_points[0][1]), z=0.0)
            msg.pixel_top_right = Vector3(x=float(image_points[1][0]), y=float(image_points[1][1]), z=0.0)
            msg.pixel_bottom_right = Vector3(x=float(image_points[2][0]), y=float(image_points[2][1]), z=0.0)
            msg.pixel_bottom_left = Vector3(x=float(image_points[3][0]), y=float(image_points[3][1]), z=0.0)

        else:
            # RESET ALL FIELDS
            zero_vec = Vector3(x=0.0, y=0.0, z=0.0)
            
            # Legacy
            msg.distance = 0.0
            msg.position_vec = zero_vec
            msg.orientation_vec = zero_vec
            msg.pixel_vec = zero_vec
            
            # Pixel Corners
            msg.pixel_top_left = zero_vec
            msg.pixel_top_right = zero_vec
            msg.pixel_bottom_right = zero_vec
            msg.pixel_bottom_left = zero_vec

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