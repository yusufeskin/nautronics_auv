#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import cv2 as cv
import numpy as np
import glob
import os

class CameraCalibrationNode(Node):
    def __init__(self):
        super().__init__('camera_calibration_node')
        self.get_logger().info('Camera Calibration Node Started...')
        
        # --- SETTINGS ---
        self.pattern_size = (9, 6)
        self.square_size = 0.025 # In meters (2.5 cm)
        
        # Location to read images from (calib_images folder)
        self.img_mask = "calib_images/*.jpg" 
        
        # Start the process automatically
        self.perform_calibration()

    def perform_calibration(self):
        images = glob.glob(self.img_mask)
        
        if not images:
            self.get_logger().warn(f"No images found! Path searched: {os.path.abspath(self.img_mask)}")
            self.get_logger().warn("Please create a 'calib_images' folder and add .jpg images inside.")
            return

        self.get_logger().info(f"{len(images)} images found, starting process...")

        # Arrays required for calibration
        obj_points = [] # 3D points in real world space
        img_points = [] # 2D points in image plane

        # Prepare chessboard corner points (0,0,0), (1,0,0), (2,0,0) ...
        pattern_points = np.zeros((np.prod(self.pattern_size), 3), np.float32)
        pattern_points[:, :2] = np.indices(self.pattern_size).T.reshape(-1, 2)
        pattern_points *= self.square_size

        h, w = 0, 0

        for fn in images:
            img = cv.imread(fn, 0) # Read as grayscale
            if img is None:
                continue
            
            h, w = img.shape[:2]
            found, corners = cv.findChessboardCorners(img, self.pattern_size)

            if found:
                # Refine corner locations (SubPix)
                term = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_COUNT, 30, 0.1)
                cv.cornerSubPix(img, corners, (5, 5), (-1, -1), term)
                
                img_points.append(corners)
                obj_points.append(pattern_points)
                self.get_logger().info(f"{fn} ... OK")
            else:
                self.get_logger().warn(f"{fn} ... Chessboard not found")

        if len(img_points) > 0:
            self.get_logger().info("Calculating calibration...")
            rms, camera_matrix, dist_coefs, rvecs, tvecs = cv.calibrateCamera(obj_points, img_points, (w, h), None, None)

            self.get_logger().info("\n" + "="*30)
            self.get_logger().info(f"RESULTS (RMS Error: {rms:.4f})")
            self.get_logger().info(f"Camera Matrix:\n{camera_matrix}")
            self.get_logger().info(f"Distortion Coefficients:\n{dist_coefs.ravel()}")
            self.get_logger().info("="*30)
        else:
            self.get_logger().error("Not enough valid images found.")

def main(args=None):
    rclpy.init(args=args)
    node = CameraCalibrationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()