#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point  # <--- YENİ: Koordinat mesajı
from cv_bridge import CvBridge
import cv2
import numpy as np

class GateDetectorNode(Node):
    def __init__(self):
        super().__init__('gate_detector_node')
        
        # --- AYARLAR ---
        self.GATE_REAL_WIDTH = 1.5  # Metre
        self.GATE_REAL_HEIGHT = 1.0 # Metre
        
        self.bridge = CvBridge()
        
        # 1. Kamera Görüntüsünü ALAN (Abone)
        self.subscription = self.create_subscription(
            Image,
            '/camera/front', 
            self.image_callback,
            10)
            
        # 2. İşlenmiş Görüntüyü GÖNDEREN (Debug için)
        self.debug_pub = self.create_publisher(Image, 'debug/gate_image', 10)
        
        # 3. Kapı Verilerini YAYINLAYAN (YENİ - Telsiz)
        # Mesaj Tipi: Point (x, y, z)
        # x = Görüntüdeki Yatay Merkez
        # y = Görüntüdeki Dikey Merkez
        # z = Mesafe (Metre)
        self.gate_data_pub = self.create_publisher(Point, '/auv/gate_data', 10)
        
        self.get_logger().info("Gate Detector Node Started (Publishing Data)")

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.process_image(cv_image)
        except Exception as e:
            self.get_logger().error(f"Error: {e}")

    def process_image(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Renk Ayarları (Senin bulduğun)
        lower_black = np.array([0, 0, 0])  
        upper_black = np.array([50, 50, 126])
        
        mask = cv2.inRange(hsv, lower_black, upper_black)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Görsel Çizimler
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            center_x = x + w // 2
            center_y = y + h // 2
            cv2.circle(frame, (center_x, center_y), 5, (255, 0, 0), -1)

            # --- MESAFE HESABI (PnP) ---
            object_points = np.array([
                [-self.GATE_REAL_WIDTH/2, -self.GATE_REAL_HEIGHT/2, 0],
                [ self.GATE_REAL_WIDTH/2, -self.GATE_REAL_HEIGHT/2, 0],
                [ self.GATE_REAL_WIDTH/2,  self.GATE_REAL_HEIGHT/2, 0],
                [-self.GATE_REAL_WIDTH/2,  self.GATE_REAL_HEIGHT/2, 0]
            ], dtype=np.float32)

            image_points = np.array([
                [x, y], [x+w, y], [x+w, y+h], [x, y+h]
            ], dtype=np.float32)

            img_h, img_w, _ = frame.shape
            camera_matrix = np.array([
                [img_w, 0, img_w/2], [0, img_w, img_h/2], [0, 0, 1]
            ], dtype=np.float32)
            
            success, _, translation_vec = cv2.solvePnP(
                object_points, image_points, camera_matrix, np.zeros((4,1))
            )
            
            if success:
                distance = translation_vec[2][0]
                
                # --- YENİ KISIM: VERİYİ YAYINLA ---
                gate_msg = Point()
                gate_msg.x = float(center_x)  # Kapı sağda mı solda mı?
                gate_msg.y = float(center_y)  # Kapı yukarıda mı aşağıda mı?
                gate_msg.z = float(distance)  # Kapı ne kadar uzakta?
                
                self.gate_data_pub.publish(gate_msg)
                
                # Log (Yine de görelim)
                self.get_logger().info(f"Yayinlaniyor -> Mesafe: {distance:.2f}m | X: {center_x}")

        out_msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
        self.debug_pub.publish(out_msg)

def main(args=None):
    rclpy.init(args=args)
    node = GateDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()