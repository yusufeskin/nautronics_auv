#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import Twist
from auv_interfaces.msg import DetectionArray
from auv_interfaces.action import CenterTarget
from rclpy.qos import qos_profile_sensor_data

class CenterTargetActionServer(Node):
    def __init__(self):
        super().__init__('center_target_action_server')

        self.callback_group = ReentrantCallbackGroup()

        self.latest_msg = None 
        self.msg_received = False
        
        # Publisher for velocities
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Subscriber for YOLO detections
        self.subscriber = self.create_subscription(
            DetectionArray, 
            '/object_3d_poses', 
            self.listener_callback, 
            qos_profile=qos_profile_sensor_data,
            callback_group=self.callback_group
        )

        # The Action Server
        self._action_server = ActionServer(
            self,
            CenterTarget,
            'center_target',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=self.callback_group
        )
        
        self.center_x = 320.0 # Default assuming 640 width
        self.kp_yaw = 0.002
        
        self.get_logger().info('Center Target Action Server started.')

    def goal_callback(self, goal_request):
        self.get_logger().info(f'Goal received: Align to {goal_request.target_class}')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        self.get_logger().info('Cancel request received.')
        return CancelResponse.ACCEPT

    def listener_callback(self, msg):
        self.latest_msg = msg
        self.msg_received = True

    def stop_robot(self):
        self.publisher.publish(Twist())

    async def execute_callback(self, goal_handle):
        self.get_logger().info('Visual Servoing (Center Target) Executing...')
        
        feedback_msg = CenterTarget.Feedback()
        result = CenterTarget.Result()

        target_class = goal_handle.request.target_class
        error_tol_x = goal_handle.request.error_tol_x
        settle_time = goal_handle.request.settle_time

        loop_rate = self.create_rate(10)
        
        time_centered = None

        while rclpy.ok():

            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.stop_robot()
                self.get_logger().info('Goal canceled.')
                result.success = False
                return result

            if not self.msg_received or self.latest_msg is None:
                self.get_logger().warn("Waiting for YOLO data...", throttle_duration_sec=2.0)
                self.stop_robot()
                loop_rate.sleep()
                continue
                
            target_obj = None
            for det in self.latest_msg.detections:
                if det.class_name == target_class:
                    target_obj = det
                    break
                    
            if target_obj is None:
                self.get_logger().warn(f"Target '{target_class}' not in view.", throttle_duration_sec=2.0)
                self.stop_robot()
                time_centered = None
                loop_rate.sleep()
                continue

            err_x = target_obj.bbox_center_x - self.center_x
            
            feedback_msg.current_error_x = float(err_x)
            goal_handle.publish_feedback(feedback_msg)

            if abs(err_x) < error_tol_x:
                if time_centered is None:
                    time_centered = self.get_clock().now()
                    self.get_logger().info('Target centered, settling...')
                    
                self.stop_robot() # Actively damp momentum
                
                elapsed_time = (self.get_clock().now() - time_centered).nanoseconds / 1e9
                if elapsed_time >= settle_time:
                    self.get_logger().info(f"Settled successfully! Error X: {err_x:.2f}")
                    goal_handle.succeed()
                    result.success = True
                    return result
            else:
                time_centered = None
                
                yaw_cmd = -self.kp_yaw * err_x
                max_yaw = 0.3
                yaw_cmd = max(-max_yaw, min(max_yaw, yaw_cmd))
                
                cmd = Twist()
                cmd.angular.z = float(yaw_cmd)
                self.publisher.publish(cmd)

            loop_rate.sleep()
        
        return result

def main(args=None):
    rclpy.init(args=args)
    node = CenterTargetActionServer()
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
