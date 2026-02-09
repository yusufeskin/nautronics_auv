import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import numpy as np
from auv_interfaces.msg import TorpedoTarget
from geometry_msgs.msg import Twist


class VisualServoingController(Node):
    def __init__(self):
        super().__init__('visual_servoing')
        self.get_logger().info("başladı")
        self.cu = 320
        self.cv = 240
        self.fx = 556
        self.fy = 556
        self.lambda_gain = 0.2
        self.subscriber = self.create_subscription(TorpedoTarget, '/auv/torpedo_data', self.visual_servoing, 10)
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        self.targets = [
        (237, 127), # top_left
        (433, 128), # top_right
        (429, 322), # bottom_right
        (239, 322)  # bottom_left
    ]
    def visual_servoing(self, msg):
        cmd = Twist()
        self.z = msg.distance
        self.u = msg.pixel_vec.x
        self.v = msg.pixel_vec.y
        yaw_error = msg.orientation_vec.z 
        w_yaw = -self.lambda_gain * yaw_error

        self.current_points = [
        (msg.pixel_top_left.x,     msg.pixel_top_left.y),
        (msg.pixel_top_right.x,    msg.pixel_top_right.y),
        (msg.pixel_bottom_right.x, msg.pixel_bottom_right.y),
        (msg.pixel_bottom_left.x,  msg.pixel_bottom_left.y)
    ]
        self.get_logger().warn(f"{self.z}, {self.u}, {self.v}")

        L_stacked = []
        error_stacked = []

        for i in range(4):
            curr_u, curr_v = self.current_points[i]
            x = (curr_u - self.cu) / self.fx
            y = (curr_v - self.cv) / self.fy
            
            tar_u, tar_v = self.targets[i]
            x_star = (tar_u - self.cu) / self.fx
            y_star = (tar_v - self.cv) / self.fy
            
            L_i = np.array([
                [-1/self.z,    0,     x/self.z,   -(1 + x**2)],
                [ 0,    -1/self.z,    y/self.z,   -x * y]
            ])

            e_i = np.array([[x - x_star], [y - y_star]])
        
            L_stacked.append(L_i)
            error_stacked.append(e_i)

        #vertical stacking
        L_total = np.vstack(L_stacked)
        e_total = np.vstack(error_stacked)
        error_norm = np.linalg.norm(e_total)
        self.get_logger().info(f"{error_norm}")
        if error_norm < 0.05:
            self.stop_robot()
            return
        
        L_v = L_total[:, 0:3]  # 8x3)
        L_w = L_total[:, 3:]   # (8x1)

        try:
            L_v_inv = np.linalg.pinv(L_v)

            # hybrid visual servoing (https://inria.hal.science/inria-00350638v1/document)
            # v = -lambda * L_v_inv * (error - L_w * w_yaw)
            compensated_error = e_total - (L_w * w_yaw)
            linear_velocities = -self.lambda_gain * np.dot(L_v_inv, compensated_error)
            
            v_surge = np.clip(linear_velocities[2], -1, 1)
            v_sway  = np.clip(linear_velocities[0], -1, 1)
            v_heave = np.clip(linear_velocities[1], -1, 1)
            
            v_yaw = np.clip(w_yaw, -1, 1)

            self.get_logger().info(f'Surge: {v_surge}, Sway: {v_sway}, Yaw: {v_yaw}')
            cmd.linear.x = float(v_surge)
            cmd.linear.y = -float(v_sway)
            cmd.linear.z = -float(v_heave)
            cmd.angular.z = float(v_yaw)

            self.publisher.publish(cmd)

        except np.linalg.LinAlgError:
            return        
    def stop_robot(self):
        self.publisher.publish(Twist())



def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(VisualServoingController())
    rclpy.shutdown()

if __name__ == '__main__':
    main()
