import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import Twist, Vector3
from auv_interfaces.action import YawAndScan
import math
import time

KP = 1.0
TOLERANCE_RAD = 0.01
MIN_SPEED = 0.2

def normalize_angle(angle):
    while angle > math.pi: angle -= 2.0 * math.pi
    while angle < -math.pi: angle += 2.0 * math.pi
    return angle

class ScanActionServer(Node):
    def __init__(self):
        super().__init__("scan_action_server")

        self.callback_group = ReentrantCallbackGroup()
        self._action_server = ActionServer(
            self,
            YawAndScan,
            "yaw_and_scan",
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=self.callback_group
        )
        self.vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.attitude_sub = self.create_subscription(
            Vector3, 
            "/current_attitude", 
            self.attitude_callback, 
            10,
            callback_group=self.callback_group
        )

        self.current_yaw = 0.0
        self.is_attitude_received = False
        self.get_logger().info("action ready and listening /current_attitude")

    def attitude_callback(self, msg):
        # msg.z = yaw in degrees (from pixhawk_bridge attitude_handler)
        self.current_yaw = math.radians(msg.z)
        self.is_attitude_received = True

    def goal_callback(self, goal_request):
        if not self.is_attitude_received:
            self.get_logger().warn("no data from /current_attitude")
            return GoalResponse.REJECT

        if goal_request.max_angular_speed <= 0.0:
            self.get_logger().warn("canceled max angular speed must be positive")
            return GoalResponse.REJECT

        self.get_logger().info(f"accepted: target:{goal_request.target_angle_deg}°")
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        self.get_logger().info("canceled, stopping robot")
        return CancelResponse.ACCEPT

    async def execute_callback(self, goal_handle):
        target_deg_relative = goal_handle.request.target_angle_deg
        max_speed = goal_handle.request.max_angular_speed
        start_yaw = self.current_yaw
        target_yaw_abs = normalize_angle(start_yaw + math.radians(target_deg_relative))
        
        feedback_msg = YawAndScan.Feedback()
        result = YawAndScan.Result()
        cmd = Twist()

        rate = self.create_rate(20) 

        while rclpy.ok():
            if goal_handle.is_cancel_requested:
                self.stop_robot()
                goal_handle.canceled()
                result.success = False
                result.message = "canceled"
                return result

            error = normalize_angle(self.current_yaw - target_yaw_abs)
            turned_amount = normalize_angle(self.current_yaw - start_yaw)
            feedback_msg.current_angle_deg = math.degrees(turned_amount)
            goal_handle.publish_feedback(feedback_msg)

            if abs(error) < TOLERANCE_RAD:
                self.stop_robot()
                self.get_logger().info("success, stopping")
                break

            calculated_speed = error * KP

            if abs(calculated_speed) > max_speed: calculated_speed = math.copysign(max_speed, calculated_speed)
            if abs(error) > TOLERANCE_RAD * 3 and abs(calculated_speed) < MIN_SPEED:
                calculated_speed = math.copysign(MIN_SPEED, calculated_speed)

            cmd.angular.z = calculated_speed
            self.vel_pub.publish(cmd)            
            rate.sleep()
        self.stop_robot()
        goal_handle.succeed()
        
        result.success = True
        result.message = f"{target_deg_relative} yawed that much."
        self.get_logger().info(f"last error {math.degrees(error):.2f}°")
        return result

    def stop_robot(self):
        cmd = Twist()
        cmd.angular.z = 0.0
        self.vel_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    action_server = ScanActionServer()
    executor = MultiThreadedExecutor()
    executor.add_node(action_server)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            executor.shutdown()
        except Exception:
            pass

        stop_cmd = Twist()
        stop_cmd.angular.z = 0.0
        
        action_server.get_logger().info("Motorlar durduruluyor...")
        for _ in range(4):
            action_server.vel_pub.publish(stop_cmd)
            time.sleep(0.1)
            
        action_server.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()