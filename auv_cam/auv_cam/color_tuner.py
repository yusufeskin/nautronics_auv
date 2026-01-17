#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

def nothing(x):
    pass

class ColorTuner(Node):
    def __init__(self):
        super().__init__('color_tuner')
        
        # Topic ismini buraya yaz (senin bulduğun isim)
        self.subscription = self.create_subscription(
            Image,
            '/camera/front', 
            self.image_callback,
            10)
        
        self.bridge = CvBridge()
        
        # Pencere ve Trackbar'ları oluştur
        cv2.namedWindow('Calibration')
        
        # H: Hue (Renk Özü) 0-179
        # S: Saturation (Doygunluk) 0-255
        # V: Value (Parlaklık) 0-255
        
        # Başlangıç değerleri (Lower)
        cv2.createTrackbar('L_H', 'Calibration', 0, 179, nothing)
        cv2.createTrackbar('L_S', 'Calibration', 0, 255, nothing)
        cv2.createTrackbar('L_V', 'Calibration', 0, 255, nothing)
        
        # Bitiş değerleri (Upper)
        cv2.createTrackbar('U_H', 'Calibration', 179, 179, nothing)
        cv2.createTrackbar('U_S', 'Calibration', 255, 255, nothing)
        cv2.createTrackbar('U_V', 'Calibration', 255, 255, nothing)

        self.get_logger().info("Renk Ayar Aracı Başladı! Slider'larla oyna...")

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Trackbar pozisyonlarını oku
            l_h = cv2.getTrackbarPos('L_H', 'Calibration')
            l_s = cv2.getTrackbarPos('L_S', 'Calibration')
            l_v = cv2.getTrackbarPos('L_V', 'Calibration')
            
            u_h = cv2.getTrackbarPos('U_H', 'Calibration')
            u_s = cv2.getTrackbarPos('U_S', 'Calibration')
            u_v = cv2.getTrackbarPos('U_V', 'Calibration')
            
            lower_color = np.array([l_h, l_s, l_v])
            upper_color = np.array([u_h, u_s, u_v])
            
            # Maske oluştur
            mask = cv2.inRange(hsv, lower_color, upper_color)
            
            # Sonucu göster (Orijinal + Maske)
            result = cv2.bitwise_and(frame, frame, mask=mask)
            
            # Ekrana sığması için küçültüyoruz
            stacked = np.hstack((cv2.resize(frame, (320, 240)), cv2.resize(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), (320, 240))))
            
            cv2.imshow('Calibration', stacked)
            cv2.waitKey(1)
            
        except Exception as e:
            pass

def main(args=None):
    rclpy.init(args=args)
    node = ColorTuner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()