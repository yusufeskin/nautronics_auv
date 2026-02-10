import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from std_msgs.msg import String
from geometry_msgs.msg import Twist
from auv_interfaces.msg import TorpedoTarget
from auv_interfaces.action import ServoToTorpedo 

import numpy as np
import time

class VisualServoingActionServer(Node):
    def __init__(self):
        super().__init__('visual_servoing_action_server')
        self.get_logger().info("Visual Servoing Action Server (Simple) Başladı")

        self.cu = 320
        self.cv = 240
        self.fx = 556
        self.fy = 556
        self.lambda_gain = 0.2

        self.callback_group = ReentrantCallbackGroup()

        self.latest_msg = None 
        self.msg_received = False

        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        #will be changed
        self.subscriber = self.create_subscription(
            TorpedoTarget, 
            '/auv/torpedo_data', 
            self.listener_callback, 
            10,
            callback_group=self.callback_group
        )

        self._action_server = ActionServer(
            self,
            ServoToTorpedo,
            'torpedo_servo',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=self.callback_group
        )

    def goal_callback(self, goal_request):
        if len(goal_request.target_points) != 4:
            self.get_logger().error(f"HATA: 4 nokta gerekli, {len(goal_request.target_points)} geldi.")
            return GoalResponse.REJECT
        self.get_logger().info('Hedef kabul edildi.')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        self.get_logger().info('İptal isteği alındı.')
        return CancelResponse.ACCEPT

    def listener_callback(self, msg):
        self.latest_msg = msg
        self.msg_received = True

    def stop_robot(self):
        self.publisher.publish(Twist())

    async def execute_callback(self, goal_handle):
        self.get_logger().info('Visual Servoing Başlatılıyor...')
        
        feedback_msg = ServoToTorpedo.Feedback()
        result = ServoToTorpedo.Result()

        req_targets = goal_handle.request.target_points
        target_points_xy = [(pt.x, pt.y) for pt in req_targets]

        loop_rate = self.create_rate(10)

        while rclpy.ok():
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.stop_robot()
                self.get_logger().info('Görev iptal edildi.')
                result.success = False
                return result

            if not self.msg_received or self.latest_msg is None:
                self.get_logger().warn("Veri bekleniyor...", throttle_duration_sec=1)
                loop_rate.sleep()
                continue

            msg = self.latest_msg
            z = msg.distance 
            
            current_detected_points = [
                (msg.pixel_top_left.x,     msg.pixel_top_left.y),
                (msg.pixel_top_right.x,    msg.pixel_top_right.y),
                (msg.pixel_bottom_right.x, msg.pixel_bottom_right.y),
                (msg.pixel_bottom_left.x,  msg.pixel_bottom_left.y)
            ]

            yaw_error = msg.orientation_vec.z
            w_yaw_val = -self.lambda_gain * yaw_error

            L_stacked = []
            error_stacked = []

            for i in range(4):
                curr_u, curr_v = current_detected_points[i]
                x = (curr_u - self.cu) / self.fx
                y = (curr_v - self.cv) / self.fy
                
                tar_u, tar_v = target_points_xy[i]
                x_star = (tar_u - self.cu) / self.fx
                y_star = (tar_v - self.cv) / self.fy

                L_i = np.array([
                    [-1/z,    0,      x/z,   -(1 + x**2)],
                    [ 0,     -1/z,    y/z,   -x * y]
                ])

                e_i = np.array([[x - x_star], [y - y_star]])
                
                L_stacked.append(L_i)
                error_stacked.append(e_i)
            #vertical stacking
            L_total = np.vstack(L_stacked)
            e_total = np.vstack(error_stacked)
            error_norm = np.linalg.norm(e_total)

            feedback_msg.current_distance = float(z)
            feedback_msg.current_error = float(error_norm)
            goal_handle.publish_feedback(feedback_msg)

            if error_norm < 0.2:
                self.get_logger().info(f"Hedefe Ulaşıldı! Hata: {error_norm:.4f}")
                self.stop_robot()
                goal_handle.succeed()
                result.success = True
                return result

            try:
                L_v = L_total[:, 0:3] 
                L_w = L_total[:, 3:] 

                L_v_inv = np.linalg.pinv(L_v)
                # hybrid visual servoing (https://inria.hal.science/inria-00350638v1/document)
                # v = -lambda * L_v_inv * (error - L_w * w_yaw)
                compensated_error = e_total - (L_w * w_yaw_val)
                v_linear_raw = -self.lambda_gain * np.dot(L_v_inv, compensated_error)

                v_linear = v_linear_raw.flatten() 

                v_surge = np.clip(v_linear[2], -0.5, 0.5)
                v_sway  = np.clip(v_linear[0], -0.5, 0.5)
                v_heave = np.clip(v_linear[1], -0.5, 0.5)
                v_yaw   = np.clip(w_yaw_val, -0.3, 0.3)

                cmd = Twist()
                cmd.linear.x = float(v_surge)
                cmd.linear.y = -float(v_sway)
                cmd.linear.z = -float(v_heave)
                cmd.angular.z = float(v_yaw)

                self.publisher.publish(cmd)
                
            except Exception as e:
                self.get_logger().error(f"Hesaplama Hatası: {e}")

            loop_rate.sleep()
        
        return result

def main(args=None):
    rclpy.init(args=args)
    node = VisualServoingActionServer()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()