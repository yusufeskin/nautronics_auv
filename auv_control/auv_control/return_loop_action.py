import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import Twist
from auv_interfaces.action import ReturnLoop
import time
import math

class ReturnLoopActionServer(Node):
    def __init__(self):
        super().__init__('return_loop_action_server')

        # Using ReentrantCallbackGroup to allow concurrent callbacks
        self.callback_group = ReentrantCallbackGroup()

        # Publisher setup
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        self._action_server = ActionServer(
            self,
            ReturnLoop,
            'return_loop',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=self.callback_group
        )
        
        self.get_logger().info('Return Loop Action Server is ready.')

    def goal_callback(self, goal_request):
        """Accepts or rejects the goal request."""
        if goal_request.duration <= 0:
            self.get_logger().error('Rejecting goal: Duration must be positive.')
            return GoalResponse.REJECT
        
        self.get_logger().info(f'Goal accepted: Radius={goal_request.radius}, Duration={goal_request.duration}')
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
        self.get_logger().info('Executing Return Loop...')

        feedback_msg = ReturnLoop.Feedback()
        result = ReturnLoop.Result()

        # Get goal parameters
        duration = goal_handle.request.duration
        radius = goal_handle.request.radius

        # Calculate max speeds to achieve the circle with linear deceleration
        # Total angle = 2*pi. Integral of w0 * (1 - t/T) from 0 to T is w0*T/2. So w0 = 4*pi/T
        w_max = (4.0 * math.pi) / duration
        if radius < 0:
            w_max = -w_max # Negative radius means turn the other way (right)
            radius = -radius
            
        v_max = radius * abs(w_max)

        # Setup loop rate
        loop_rate = self.create_rate(20) # 20 Hz for smoother curve
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
                self.get_logger().info('Return Loop Completed Successfully.')
                result.success = True
                return result

            # 3. Execution Logic (Open Loop Movement with deceleration)
            progress = elapsed_time / duration
            speed_factor = max(0.0, 1.0 - progress) # linear decrease from 1 to 0
            
            cmd = Twist()
            cmd.linear.x = float(v_max * speed_factor) # Surge
            cmd.linear.y = 0.0          # Sway (keep 0 so nose is aligned with velocity)
            cmd.linear.z = 0.0          # Heave
            cmd.angular.z = float(w_max * speed_factor) # Yaw
            
            self.publisher.publish(cmd)

            # 4. Publish Feedback
            feedback_msg.remaining_time = float(remaining)
            goal_handle.publish_feedback(feedback_msg)

            loop_rate.sleep()
        
        return result

def main(args=None):
    rclpy.init(args=args)
    node = ReturnLoopActionServer()
    
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
