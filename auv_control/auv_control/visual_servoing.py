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
    def visual_servoing(self, msg):
        cmd = Twist()
        self.z = msg.distance
        self.u = msg.pixel_vec.x
        self.v = msg.pixel_vec.y
        self.get_logger().warn(f"{self.z}, {self.u}, {self.v}")
        #interaction matrix (L) assumes fx=1 and fy=1, we should normalize our pixel coordinates to give (L) 
        u_centered = self.u - self.cu 
        v_centered = self.v - self.cv
        x_norm = u_centered / self.fx
        y_norm = v_centered / self.fy
        #interaciton matrix
        L = np.array([
            [-1/self.z,   0,      x_norm/self.z,   y_norm],
            [ 0,    -1/self.z,    y_norm/self.z,  -x_norm]
        ])
        #because target point is (0,0) instead of writing ([x_norm - 0], [y_norm - 0]) I wrote directly [x_norm], [y_norm]
        error_vector = np.array([[x_norm], [y_norm]])
        try:
            # Pseudo-Inverse, because its not square matrix
            L_inv = np.linalg.pinv(L)
            # find the velocities
            velocities = -self.lambda_gain * np.dot(L_inv, error_vector)
            self.get_logger().info(f'{velocities}')
            cmd.linear.x = float(velocities[0])
            cmd.linear.y = float(velocities[1])
            cmd.linear.z = float(velocities[2])
            cmd.angular.z = float(velocities[3])
            self.publisher.publish(cmd)
            
            
        except np.linalg.LinAlgError:
            self.get_logger().error("inversion failed!")
            return


def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(VisualServoingController())
    rclpy.shutdown()

if __name__ == '__main__':
    main()
