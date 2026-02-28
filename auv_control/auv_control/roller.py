import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu
from auv_interfaces.action import Roll
import math
import time

KP = 1.0
TOLERANCE_RAD = 0.05
MIN_SPEED = 0.15

# ÖNEMLİ: Eğer robot hala sonsuza kadar dönüyorsa, bu değeri -1.0 yap!
# Bu, motorun dönüş yönüyle IMU'nun okuma yönü ters ise onları eşitlemeye yarar.
MOTOR_DIRECTION = -1.0 

def normalize_angle(angle):
    while angle > math.pi: angle -= 2.0 * math.pi
    while angle < -math.pi: angle += 2.0 * math.pi
    return angle

def euler_from_quaternion(x, y, z, w):
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    return math.atan2(t3, t4)

class ScanActionServer(Node):
    def __init__(self):
        super().__init__("scan_action_server")

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
        self.imu_sub = self.create_subscription(
            Imu, 
            "/imu0", 
            self.imu_callback, 
            10,
            callback_group=self.callback_group
        )

        self.current_yaw = 0.0
        self.is_imu_received = False
        self.get_logger().info("Scan Action Server Hazır. /imu topici bekleniyor...")

    def imu_callback(self, msg):
        q = msg.orientation
        self.current_yaw = euler_from_quaternion(q.x, q.y, q.z, q.w)
        self.is_imu_received = True

    def goal_callback(self, goal_request):
        if not self.is_imu_received:
            self.get_logger().warn("REDDEDİLDİ: /imu verisi henüz gelmedi.")
            return GoalResponse.REJECT

        if goal_request.angular_speed <= 0.0:
            self.get_logger().warn("REDDEDİLDİ: Hız 0 veya negatif olamaz.")
            return GoalResponse.REJECT

        self.get_logger().info(f"KABUL EDİLDİ: Hedef {goal_request.target_angle_deg} derece.")
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        self.get_logger().info("İptal isteği alındı. Durduruluyor...")
        self.stop_robot()
        return CancelResponse.ACCEPT

    async def execute_callback(self, goal_handle):
        # Hedefi artık client'tan alıyoruz (örn: 720 veya -720)
        target_deg_relative = goal_handle.request.target_angle_deg
        target_rad = math.radians(target_deg_relative)
        max_speed = goal_handle.request.angular_speed
        
        accumulated_yaw = 0.0
        previous_yaw = self.current_yaw
        
        feedback_msg = Roll.Feedback()
        result = Roll.Result()
        cmd = Twist()

        rate = self.create_rate(20) 

        while rclpy.ok():
            if goal_handle.is_cancel_requested:
                self.stop_robot()
                goal_handle.canceled()
                result.success = False
                result.message = "Operasyon iptal edildi."
                return result

            delta_yaw = normalize_angle(self.current_yaw - previous_yaw)
            accumulated_yaw += delta_yaw
            previous_yaw = self.current_yaw

            error = target_rad - accumulated_yaw

            feedback_msg.current_angle_deg = math.degrees(accumulated_yaw)
            goal_handle.publish_feedback(feedback_msg)

            if abs(error) < TOLERANCE_RAD:
                self.stop_robot()
                self.get_logger().info("Hedefe varıldı!")
                break

            # Motor yönü ile IMU tersliği ihtimaline karşı MOTOR_DIRECTION ile çarpıyoruz
            calculated_speed = error * KP * MOTOR_DIRECTION

            if abs(calculated_speed) > max_speed: 
                calculated_speed = math.copysign(max_speed, calculated_speed)
            if abs(calculated_speed) < MIN_SPEED: 
                calculated_speed = math.copysign(MIN_SPEED, calculated_speed)

            cmd.angular.z = calculated_speed
            self.vel_pub.publish(cmd)            
            rate.sleep()
            
        self.stop_robot()
        goal_handle.succeed()
        
        result.success = True
        result.message = f"{target_deg_relative} derecelik dönüş tamamlandı."
        return result

    def stop_robot(self):
        cmd = Twist()
        cmd.angular.z = 0.0
        # Güvence için üst üste birkaç kez sıfır komutu gönderelim
        for _ in range(3):
            self.vel_pub.publish(cmd)
            time.sleep(0.05)

def main(args=None):
    rclpy.init(args=args)
    action_server = ScanActionServer()
    executor = MultiThreadedExecutor()
    executor.add_node(action_server)

    try:
        executor.spin()
    except KeyboardInterrupt:
        # Ctrl+C basıldığında rclpy kapanmadan ÖNCE motorları durduruyoruz!
        action_server.get_logger().info("\nCtrl+C algılandı, motorlar ACİL durduruluyor...")
        action_server.stop_robot()
    finally:
        try:
            executor.shutdown()
        except Exception:
            pass
        action_server.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()