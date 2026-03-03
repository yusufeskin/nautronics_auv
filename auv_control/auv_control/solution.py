import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import math


dead_band = 0.1

class OpticalFlowStationKeeping(Node):
    def __init__(self):
        super().__init__('optical_flow_node')
        
        self.get_logger().info("Optical Flow started")

        self.subscription = self.create_subscription(
            Image,
            "/camera/bottom", 
            self.image_callback,
            10)
        
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.bridge = CvBridge()
        self.lk_params = dict(winSize=(21, 21),
                              maxLevel=4,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

        self.feature_params = dict(maxCorners=300,
                                   qualityLevel=0.3,
                                   minDistance=6,
                                   blockSize=7)

        self.old_gray = None
        self.p0 = None
        self.total_sway_pixels = 0.0
        self.total_surge_pixels = 0.0

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge Hatası: {e}")
            return  

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.old_gray is None or self.p0 is None or len(self.p0) < 10:
            self.old_gray = frame_gray.copy()
            self.p0 = cv2.goodFeaturesToTrack(self.old_gray, mask=None, **self.feature_params)
            self.get_logger().info("koseler bulundu")
            #aracın şasisikamerada gozukuyosa maskeleyk
            if self.p0 is None:
                self.get_logger().warn("kose yok")
            return


        p1, st, err = cv2.calcOpticalFlowPyrLK(self.old_gray, frame_gray, self.p0, None, **self.lk_params)

        if p1 is not None and st is not None:

            good_new = p1[st == 1]
            good_old = self.p0[st == 1]

            if len(good_old) >= 3 and len(good_new) >= 3:
                transform_matrix, inliers = cv2.estimateAffinePartial2D(good_old, good_new)
                
                if transform_matrix is not None:

                    dx = transform_matrix[0, 2] 
                    dy = transform_matrix[1, 2]

                    self.total_sway_pixels += dx
                    self.total_surge_pixels += dy

                    Kp_sway = -0.002
                    Kd_sway = -0.03
                    Kp_surge = -0.002
                    Kd_surge = -0.03
                    
                    twist_msg = Twist()
                  
                    twist_msg.linear.y = float((self.total_sway_pixels * Kp_sway) + (dx * Kd_sway))     
                    twist_msg.linear.x = float((self.total_surge_pixels * Kp_surge) + (dy * Kd_surge)) 
                    MAX_SPEED = 0.15
                    twist_msg.linear.y = max(min(twist_msg.linear.y, MAX_SPEED), -MAX_SPEED)
                    twist_msg.linear.x = max(min(twist_msg.linear.x, MAX_SPEED), -MAX_SPEED)
                    
                    if abs(self.total_sway_pixels) < 1.5 and dx == 0 :
                        twist_msg.linear.y = 0.0

                    if abs(self.total_surge_pixels) < 1.5 and dy == 0:
                        twist_msg.linear.x = 0.0

                    self.publisher_.publish(twist_msg)

                    self.get_logger().info(f"total hata sway: {self.total_sway_pixels:.2f} total hata surge:{self.total_surge_pixels:.2f}  swayhata {dx:.2f} surgehata {dy:.2f} | Verilen Hız: {twist_msg.linear.y:.4f}{twist_msg.linear.x:.4f}")
                    

            #update
            self.old_gray = frame_gray.copy()
            self.p0 = good_new.reshape(-1, 1, 2)

def main(args=None):
    rclpy.init(args=args)
    node = OpticalFlowStationKeeping()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()