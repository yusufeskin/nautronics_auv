import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import Twist, Vector3
from auv_interfaces.action import Roll
import math
import time

KP = 0.4
TOLERANCE_RAD = 0.1
MIN_SPEED = 0.0

def normalize_angle(angle):
    while angle > math.pi: angle -= 2.0 * math.pi
    while angle < -math.pi: angle += 2.0 * math.pi
    return angle

class RollActionServer(Node):
    def __init__(self):
        super().__init__("roll_action_server")

        self.callback_group = ReentrantCallbackGroup()
        self._action_server = ActionServer(
            self,
            Roll,
            "roll",
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

        self.current_roll = 0.0
        self.is_attitude_received = False
        self.get_logger().info("action ready and listening /current_attitude")

    def attitude_callback(self, msg):
        # msg.x = roll in degrees (from pixhawk_bridge attitude_handler)
        self.current_roll = math.radians(msg.x)
        self.is_attitude_received = True

    def goal_callback(self, goal_request):
        if not self.is_attitude_received:
            self.get_logger().warn("no data from /current_attitude")
            return GoalResponse.REJECT

        if goal_request.angular_speed <= 0.0:
            self.get_logger().warn("canceled max angular speed must be positive")
            return GoalResponse.REJECT

        self.get_logger().info(f"accepted: target:{goal_request.target_angle_deg}°")
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        self.get_logger().info("canceled, stopping robot")
        return CancelResponse.ACCEPT

    async def execute_callback(self, goal_handle):
            target_deg_relative = goal_handle.request.target_angle_deg # Buraya 360.0 gelecek
            target_rad = math.radians(abs(target_deg_relative))
            max_speed = goal_handle.request.angular_speed
            
            direction = 1.0 if target_deg_relative > 0 else -1.0
            
            total_turned_rad = 0.0
            prev_roll = self.current_roll
            
            feedback_msg = Roll.Feedback()
            result = Roll.Result()
            cmd = Twist()

            rate = self.create_rate(20)

            while rclpy.ok():
                if goal_handle.is_cancel_requested:
                    self.stop_robot()
                    goal_handle.canceled()
                    result.success = False
                    return result

                delta_roll = normalize_angle(self.current_roll - prev_roll)
                total_turned_rad += delta_roll
                prev_roll = self.current_roll

                feedback_msg.current_angle_deg = math.degrees(abs(total_turned_rad))
                goal_handle.publish_feedback(feedback_msg)

                error_rad = target_rad - abs(total_turned_rad)
                
                if error_rad < TOLERANCE_RAD:
                    self.stop_robot()
                    self.get_logger().info("360 tamamlandı, durduruluyor.")
                    break

                cmd.angular.x = max_speed * direction
                self.vel_pub.publish(cmd)            
                
                rate.sleep()
                
            self.stop_robot()
            goal_handle.succeed()
            
            result.success = True
            result.message = f"{target_deg_relative} derece dönüldü."
            return result

    def stop_robot(self):
        cmd = Twist()
        cmd.angular.x = 0.0
        self.vel_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    action_server = RollActionServer()
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
        stop_cmd.angular.x = 0.0
        
        action_server.get_logger().info("Motorlar durduruluyor...")
        for _ in range(4):
            action_server.vel_pub.publish(stop_cmd)
            time.sleep(0.1)
            
        action_server.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()
