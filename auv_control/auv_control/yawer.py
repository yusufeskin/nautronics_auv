import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu
from auv_interfaces.action import YawAndScan
import math
import time

def normalize_angle(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle

def euler_from_quaternion(x, y, z, w):
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    return math.atan2(t3, t4)

class ScanActionServer(Node):
    def __init__(self):
        super().__init__('scan_action_server')

        self.callback_group = ReentrantCallbackGroup()

        self._action_server = ActionServer(
            self,
            YawAndScan,
            'yaw_and_scan',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=self.callback_group
        )

        self.vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.imu_sub = self.create_subscription(
            Imu, 
            '/imu0', 
            self.imu_callback, 
            10,
            callback_group=self.callback_group
        )

        self.current_yaw = 0.0
        self.is_imu_received = False
        
        self.get_logger().info('Scan Action Server Hazır. /imu topici dinleniyor...')

    def imu_callback(self, msg):
        q = msg.orientation
        self.current_yaw = euler_from_quaternion(q.x, q.y, q.z, q.w)
        self.is_imu_received = True

    def goal_callback(self, goal_request):
        if not self.is_imu_received:
            self.get_logger().warn('REDDEDİLDİ: /imu topici veri publishlemiyor')
            return GoalResponse.REJECT

        if goal_request.angular_speed <= 0.0:
            self.get_logger().warn('REDDEDİLDİ: Dönüş hızı 0 veya negatif olamaz.')
            return GoalResponse.REJECT

        self.get_logger().info(f'KABUL EDİLDİ: Hedef {goal_request.target_angle_deg}°')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        self.get_logger().info('İptal isteği alındı. Robot durduruluyor.')
        return CancelResponse.ACCEPT

    async def execute_callback(self, goal_handle):
        target_deg_relative = goal_handle.request.target_angle_deg
        max_speed = goal_handle.request.angular_speed

        start_yaw = self.current_yaw
        target_yaw_abs = normalize_angle(start_yaw + math.radians(target_deg_relative))
        
        feedback_msg = YawAndScan.Feedback()
        result = YawAndScan.Result()
        cmd = Twist()

        TOLERANCE_RAD = 0.01
        Kp = 1  #degistirilebilir
        MIN_SPEED = 0.2 #surtunmeyi yenmesi icin

        rate = self.create_rate(20) 

        while rclpy.ok():
            if goal_handle.is_cancel_requested:
                self.stop_robot()
                goal_handle.canceled()
                result.success = False
                result.message = "Operasyon iptal edildi."
                return result

            error = normalize_angle(-target_yaw_abs + self.current_yaw)

            turned_amount = normalize_angle(self.current_yaw - start_yaw)
            feedback_msg.current_angle_deg = math.degrees(turned_amount)
            goal_handle.publish_feedback(feedback_msg)

            if abs(error) < TOLERANCE_RAD:
                self.stop_robot()
                self.get_logger().info("Hedefe varıldı, duruluyor...")
                break

            calculated_speed = error * Kp

            
            if calculated_speed > max_speed:
                calculated_speed = max_speed
            elif calculated_speed < -max_speed:
                calculated_speed = -max_speed

            
            if abs(calculated_speed) < MIN_SPEED:
                calculated_speed = math.copysign(MIN_SPEED, calculated_speed)

         
            cmd.angular.z = calculated_speed
            self.vel_pub.publish(cmd)
            
            rate.sleep()
        self.stop_robot()
        

        #self.get_logger().info('Dönüş bitti. Kameranın netleşmesi için 1 saniye bekleniyor...')
        #time.sleep(1.0) 
        goal_handle.succeed()
        #CLİENTE ANLIK FOTOGRAF GONDERİLİP HEDEF TANINDI MI DİYE BAKILIP ONA GORE YENİ GOAL GONDERİELBİLİR
        
        
        result.success = True
        result.message = f"{target_deg_relative} derecelik tarama tamamlandı."
        self.get_logger().info(f'Tamamlandı. Son Hata Payı: {math.degrees(error):.2f}°')
        
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
        
        action_server.get_logger().info('Motorlar durduruluyor...')
        for _ in range(4):
            action_server.vel_pub.publish(stop_cmd)
            time.sleep(0.1)
            
        action_server.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
if __name__ == '__main__':
    main()