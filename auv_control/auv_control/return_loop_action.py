import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import Twist, Vector3  # Odometry silindi, Vector3 eklendi
from auv_interfaces.action import ReturnLoop
import time
import math

# Kuaterniyon dönüşüm fonksiyonuna (euler_from_quaternion) artık ihtiyacımız yok, silindi.

class ReturnLoopActionServer(Node):
    def __init__(self):
        super().__init__('return_loop_action_server')

        self.callback_group = ReentrantCallbackGroup()

        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Abonelik tipi Vector3 olarak güncellendi
        self.attitude_sub = self.create_subscription(
            Vector3, 
            '/current_attitude', 
            self.attitude_callback,  # İsim daha anlamlı olması için attitude_callback yapıldı
            10, 
            callback_group=self.callback_group
        )
        self.current_yaw = None

        self._action_server = ActionServer(
            self,
            ReturnLoop,
            'return_loop',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=self.callback_group
        )
        
        self.get_logger().info('Return Loop Server hazır. İnce hizalama aktif.')

    def attitude_callback(self, msg):
        # Yayıncı kodu yaw açısını derece olarak 'z' ekseninde gönderiyor.
        # Action server'ın geri kalanı radyan üzerinden çalıştığı için 
        # değeri radyana çevirerek sisteme kaydediyoruz.
        self.current_yaw = math.radians(msg.z)

    def goal_callback(self, goal_request):
        if goal_request.duration <= 0:
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        return CancelResponse.ACCEPT

    def stop_robot(self):
        self.publisher.publish(Twist())

    async def execute_callback(self, goal_handle):
        feedback_msg = ReturnLoop.Feedback()
        result = ReturnLoop.Result()

        while self.current_yaw is None and rclpy.ok():
            time.sleep(0.1)

        initial_yaw = self.current_yaw
        self.get_logger().info(f'Baslangic acisi kilitlendi: {math.degrees(initial_yaw):.2f} derece')

        duration = goal_handle.request.duration
        radius = goal_handle.request.radius

        w_const = (2.0 * math.pi) / duration
        if radius < 0:
            w_const = -w_const
            radius = -radius
        v_const = radius * abs(w_const)

        loop_rate = self.create_rate(20)
        start_time = self.get_clock().now()
        
        accumulated_yaw = 0.0
        last_yaw = self.current_yaw
        self.get_logger().info('Asama 1: Buyuk donus baslatildi...')
        while rclpy.ok():
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.stop_robot()
                result.success = False
                return result

            delta_yaw = self.current_yaw - last_yaw
            while delta_yaw > math.pi: delta_yaw -= 2.0 * math.pi
            while delta_yaw < -math.pi: delta_yaw += 2.0 * math.pi
            
            accumulated_yaw += abs(delta_yaw)
            last_yaw = self.current_yaw

            if accumulated_yaw >= (2.0 * math.pi):
                self.stop_robot()
                self.get_logger().info('Asama 1 Tamamlandi. Ince ayar hizalamasina geciliyor.')
                break
            
            cmd = Twist()
            cmd.linear.x = float(v_const)
            cmd.angular.z = float(w_const)
            self.publisher.publish(cmd)

            elapsed_time = (self.get_clock().now() - start_time).nanoseconds / 1e9
            feedback_msg.remaining_time = float(max(0.0, duration - elapsed_time))
            goal_handle.publish_feedback(feedback_msg)
            loop_rate.sleep()

        # ==========================================================
        # AŞAMA 2: Başlangıç Açısına İnce Ayar Hizalama (P-Kontrolcü)
        # ==========================================================
        kp_align = 2.0         
        error_threshold = 0.1 
        max_align_speed = 0.4  

        self.get_logger().info('Asama 2: Ilk aciya hizalaniliyor...')
        while rclpy.ok():
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.stop_robot()
                result.success = False
                return result

            yaw_error = initial_yaw - self.current_yaw
            
            while yaw_error > math.pi: yaw_error -= 2.0 * math.pi
            while yaw_error < -math.pi: yaw_error += 2.0 * math.pi

            if abs(yaw_error) <= error_threshold:
                self.stop_robot()
                self.get_logger().info(f'Hizalama basarili. Mevcut sapma: {math.degrees(yaw_error):.2f} derece.')
                break
            align_cmd = Twist()
            align_cmd.linear.x = 0.0
            
            w_cmd = yaw_error * kp_align
            w_cmd = max(-max_align_speed, min(max_align_speed, w_cmd))
            align_cmd.angular.z = float(w_cmd)
            
            self.publisher.publish(align_cmd)
            loop_rate.sleep()

        goal_handle.succeed()
        self.get_logger().info('Return Loop ve Hizalama tamamen bitti. Goreve devam edilebilir.')
        result.success = True
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