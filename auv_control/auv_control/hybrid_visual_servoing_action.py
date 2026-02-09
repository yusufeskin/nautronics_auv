import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node
from auv_interfaces.msg import TorpedoTarget
from auv_interfaces.action import IBVS
import numpy as np
from geometry_msgs.msg import Twist
from rclpy.executors import MultiThreadedExecutor

class HybridIBVSNode(Node):

    def __init__(self):
        super().__init__('hybrid_ibvs_action')
        self.cu = 320
        self.cv = 240
        self.fx = 556
        self.fy = 556
        self.lambda_gain = 0.2
        self.error_norm = 0.0

        self.action_server = ActionServer(
            self,
            IBVS,
            'hybrid_ibvs',
            self.execute_callback)
        self.subscription = self.create_subscription(TorpedoTarget, "/auv/torpedo_data", self.msg_callback, 10)
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)



    def msg_callback(self, msg):
        self.current_points = [
        (msg.pixel_top_left.x,     msg.pixel_top_left.y),
        (msg.pixel_top_right.x,    msg.pixel_top_right.y),
        (msg.pixel_bottom_right.x, msg.pixel_bottom_right.y),
        (msg.pixel_bottom_left.x,  msg.pixel_bottom_left.y)
    ]
        self.z = msg.distance
        self.yaw_error = msg.orientation_vec.z 
        self.w_yaw = -self.lambda_gain * self.yaw_error


 

    def execute_callback(self, goal_handle):
        cmd = Twist()
        self.get_logger().info('Executing goal...')
        result = IBVS.Result()
        target_points = goal_handle.request.target_points
        while self.error_norm < 0.5:
            L_stacked = []      
            error_stacked = []
            for i in range(4):
                curr_u, curr_v = self.current_points[i]
                x = (curr_u - self.cu) / self.fx
                y = (curr_v - self.cv) / self.fy
                
                tar_u, tar_v = target_points[i]
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
                compensated_error = e_total - (L_w * self.w_yaw)
                linear_velocities = -self.lambda_gain * np.dot(L_v_inv, compensated_error)
                
                v_surge = np.clip(linear_velocities[2], -1, 1)
                v_sway  = np.clip(linear_velocities[0], -1, 1)
                v_heave = np.clip(linear_velocities[1], -1, 1)
                
                v_yaw = np.clip(self.w_yaw, -1, 1)

                self.get_logger().info(f'Surge: {v_surge}, Sway: {v_sway}, Yaw: {v_yaw}')
                cmd.linear.x = float(v_surge)
                cmd.linear.y = -float(v_sway)
                cmd.linear.z = -float(v_heave)
                cmd.angular.z = float(v_yaw)

                self.publisher.publish(cmd)

            except np.linalg.LinAlgError:
                return        

            feedback = IBVS.Feedback()
            return result


def main(args=None):
    rclpy.init(args=args)

    hybrid_ibvs_action_server = HybridIBVSNode()
    executor = MultiThreadedExecutor()
    executor.add_node(hybrid_ibvs_action_server)
    executor.spin()
    rclpy.shutdown()

    rclpy.spin(hybrid_ibvs_action_server)


if __name__ == '__main__':
    main()