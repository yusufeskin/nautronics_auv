import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import numpy as np
import os
import math
from rclpy.qos import qos_profile_sensor_data

MODEL_NAME = "pool"

class PnPDebugger(Node):
    def __init__(self):
        super().__init__("pnp_debugger_node")

        # --- 1. Model Path Configuration ---
        ws_root = os.path.abspath(__file__)
        for i in range(7): ws_root = os.path.dirname(ws_root)
        self.model_path = os.path.join(ws_root, f"src/auv_vision/model/{MODEL_NAME}.onnx")
        
        W_HALF = 0.1
        H_HALF = 0.1

        # Order: 0=Top Left, 1=Top Right, 2=Bottom Right, 3=Bottom Left
        self.object_points = np.array([
            [-W_HALF, -H_HALF, 0.0],  # 0: Top Left (sol üst)
            [ W_HALF, -H_HALF, 0.0],  # 1: Top Right (sağ üst)
            [ W_HALF,  H_HALF, 0.0],  # 2: Bottom Right (sağ alt)
            [-W_HALF,  H_HALF, 0.0],  # 3: Bottom Left (sol alt)
        ], dtype=np.float32)

        self.bridge = CvBridge()
        self.model = YOLO(self.model_path, task="pose")
        self.camera_matrix = None
        self.dist_coeffs = None
        
        
        self.create_subscription(
            CameraInfo, 
            "/camera_info", 
            self.camera_info_callback, 
            qos_profile_sensor_data
        )
        
        self.create_subscription(
            Image, 
            "/image_raw", 
            self.image_callback, 
            qos_profile_sensor_data
        )
        
        self.debug_pub = self.create_publisher(Image, "/auv_vision/pnp_debug", 10)
        
        self.get_logger().info("PnP Debugger Started (Standard solvePnP Active).")
        
        self.get_logger().info("PnP Debugger Started (Standard solvePnP Active).")

    def camera_info_callback(self, msg):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k).reshape((3, 3))
            self.dist_coeffs = np.array(msg.d)

    def image_callback(self, msg):
        if self.camera_matrix is None: return
        
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            
            # 1. YOLO Inference
            results = self.model(frame, verbose=False, conf=0.5, imgsz=640, device='cpu', half=True)
            
            if len(results) > 0 and results[0].keypoints is not None and len(results[0].keypoints) > 0:
                kp_xy = results[0].keypoints.xy.cpu().numpy()[0]
                kp_conf = results[0].keypoints.conf.cpu().numpy()[0]
                
                image_points = []
                object_points_filtered = []
                
                # 2. Keypoint Matching
                for i in range(len(kp_xy)):
                    # Prevent index errors if model predicts more points than defined
                    if i >= len(self.object_points): break

                    # Confidence check
                    if kp_conf[i] > 0.5 and kp_xy[i][0] > 1:
                        image_points.append(kp_xy[i])
                        object_points_filtered.append(self.object_points[i])
                        
                        # Draw Debug Keypoints
                        cv2.circle(frame, (int(kp_xy[i][0]), int(kp_xy[i][1])), 5, (0, 255, 0), -1)
                        cv2.putText(frame, str(i), (int(kp_xy[i][0]), int(kp_xy[i][1])), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                img_pts_np = np.array(image_points, dtype=np.float32)
                obj_pts_np = np.array(object_points_filtered, dtype=np.float32)

                # 3. Solve PnP (Minimum 4 points required)
                if len(img_pts_np) >= 4:
                    success, rvec, tvec = cv2.solvePnP(
                        obj_pts_np, 
                        img_pts_np, 
                        self.camera_matrix, 
                        self.dist_coeffs,
                        flags=cv2.SOLVEPNP_ITERATIVE
                    )
                    
                    if success:
                        # --- Calculate Data ---
                        dist = tvec[2][0]
                        x_offset = tvec[0][0]
                        
                        rmat, _ = cv2.Rodrigues(rvec)
                        yaw_deg = math.degrees(math.atan2(rmat[0][2], rmat[2][2]))
                        
                        # --- Visualization ---
                        cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs, rvec, tvec, 0.5)
                        
                        cv2.putText(frame, f"DIST: {dist:.2f}m", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                        cv2.putText(frame, f"X-OFF: {x_offset:.2f}m", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                        cv2.putText(frame, f"YAW: {yaw_deg:.1f} deg", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                    else:
                        cv2.putText(frame, "PnP FAILED", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, f"NOT ENOUGH POINTS ({len(img_pts_np)}/4)", (20, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

            else:
                cv2.putText(frame, "NO GATE DETECTED", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            debug_frame = cv2.resize(frame, (640, 480))
            self.debug_pub.publish(self.bridge.cv2_to_imgmsg(debug_frame, encoding="bgr8"))
            self.get_logger().error(f"Error:")

        except Exception as e:
            self.get_logger().error(f"Error: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = PnPDebugger()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()