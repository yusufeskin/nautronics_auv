#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Vector3Stamped
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

class OpticalFlowVelocityEstimator(Node):
    def __init__(self):
        super().__init__('optical_flow_velocity_node')
        
        self.get_logger().info("Optik Akış Hız Kestirim Düğümü Başlatıldı...")

        self.subscription = self.create_subscription(
            Image,
            "/camera/bottom", 
            self.image_callback,
            10)
        
        self.vel_pub = self.create_publisher(Vector3Stamped, '/optical_flow/velocity', 10)
        
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
        self.t_prev = None

        ##################################################
        self.camera_distance_fr_floor = 2.0 
        self.fx = 300.0  
        self.fy = 300.0  

    def image_callback(self, msg):
        t_sec = msg.header.stamp.sec
        t_nanosec = msg.header.stamp.nanosec
        t_now = t_sec + (t_nanosec / 1e9)

        if self.t_prev is None:
            self.t_prev = t_now
            delta_t = 0.0 
        else:
            delta_t = t_now - self.t_prev
            self.t_prev = t_now

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge Hatası: {e}")
            return  

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.old_gray is None or self.p0 is None or len(self.p0) < 10:
            self.old_gray = frame_gray.copy()
            self.p0 = cv2.goodFeaturesToTrack(self.old_gray, mask=None, **self.feature_params)
            
            if self.p0 is None:
                self.get_logger().warn("Takip edilecek köşe bulunamadı, zemin çok pürüzsüz veya karanlık!")
            else:
                self.get_logger().info("Yeni köşeler yakalandı, takibe başlanıyor.")
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

                    if delta_t > 0.0:
                        v_x = (dx * self.camera_distance_fr_floor) / (self.fx * delta_t)
                        v_y = (dy * self.camera_distance_fr_floor) / (self.fy * delta_t)
                        
                        opt_vel_msg = Vector3Stamped()
                        opt_vel_msg.header.stamp = msg.header.stamp 
                        opt_vel_msg.header.frame_id = "camera_bottom_link"
                        
                        opt_vel_msg.vector.x = float(v_x)
                        opt_vel_msg.vector.y = float(v_y)
                        opt_vel_msg.vector.z = 0.0 

                        self.vel_pub.publish(opt_vel_msg)

                        self.get_logger().info(f"V_x: {v_x:.3f} m/s | V_y: {v_y:.3f} m/s | dt: {delta_t:.3f}")

            self.old_gray = frame_gray.copy()
            self.p0 = good_new.reshape(-1, 1, 2)

def main(args=None):
    rclpy.init(args=args)
    node = OpticalFlowVelocityEstimator()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()