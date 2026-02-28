import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import Twist
from auv_interfaces.action import BlindPush
import time

class BlindPushActionServer(Node):
    def __init__(self):
        super().__init__('blind_push_action_server')

        # Using ReentrantCallbackGroup to allow concurrent callbacks
        self.callback_group = ReentrantCallbackGroup()

        # Publisher setup (matching your reference)
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        self._action_server = ActionServer(
            self,
            BlindPush,
            'blind_push',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=self.callback_group
        )
        
        self.get_logger().info('Blind Push Action Server is ready.')

    def goal_callback(self, goal_request):
        """Accepts or rejects the goal request."""
        if goal_request.duration <= 0:
            self.get_logger().error('Rejecting goal: Duration must be positive.')
            return GoalResponse.REJECT
        
        self.get_logger().info(f'Goal accepted: Speed={goal_request.speed}, Duration={goal_request.duration}')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        """Handles cancellation requests."""
        self.get_logger().info('Cancel request received.')
        return CancelResponse.ACCEPT

    def stop_robot(self):
        """Publishes zero velocity to stop the robot."""
        stop_cmd = Twist()
        self.publisher.publish(stop_cmd)

    async def execute_callback(self, goal_handle):
        self.get_logger().info('Executing Blind Push...')

        feedback_msg = BlindPush.Feedback()
        result = BlindPush.Result()

        # Get goal parameters
        duration = goal_handle.request.duration
        speed = goal_handle.request.speed

        # Setup loop rate
        loop_rate = self.create_rate(10) # 10 Hz
        start_time = self.get_clock().now()

        # Main Loop
        while rclpy.ok():
            current_time = self.get_clock().now()
            elapsed_time = (current_time - start_time).nanoseconds / 1e9
            remaining = duration - elapsed_time

            # 1. Check for Cancellation
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.stop_robot()
                self.get_logger().info('Goal canceled by client.')
                result.success = False
                return result

            # 2. Check for Completion
            if elapsed_time >= duration:
                self.stop_robot()
                goal_handle.succeed()
                self.get_logger().info('Blind Push Completed Successfully.')
                result.success = True
                return result

            # 3. Execution Logic (Open Loop Movement)
            cmd = Twist()
            cmd.linear.x = float(speed) # Surge
            cmd.linear.y = 0.0          # Sway
            cmd.linear.z = 0.0          # Heave
            cmd.angular.z = 0.0         # Yaw (maintain straight line)

            self.publisher.publish(cmd)

            # 4. Publish Feedback
            feedback_msg.remaining_time = float(remaining)
            goal_handle.publish_feedback(feedback_msg)

            loop_rate.sleep()
        
        # Fallback return (should not be reached usually)
        return result

def main(args=None):
    rclpy.init(args=args)
    node = BlindPushActionServer()
    
    # Using MultiThreadedExecutor as requested
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