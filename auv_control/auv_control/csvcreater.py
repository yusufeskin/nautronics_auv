import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Vector3Stamped  # YENİ EKLENDİ
import message_filters
from pymavlink import mavutil
import csv
import os

class UltimateDataCollector(Node):
    def __init__(self):
        super().__init__('ultimate_data_collector')
        
        # ==========================================
        # 1. TEK CSV DOSYASI AYARLARI
        # ==========================================
        self.csv_file = 'tum_sensor_ve_pwm_verileri.csv'
        file_exists = os.path.isfile(self.csv_file)
        
        self.file = open(self.csv_file, mode='a', newline='')
        self.writer = csv.writer(self.file)
        
        # Dosya ilk defa oluşturuluyorsa başlıkları yaz
        if not file_exists:
            header = [
                'timestamp_sec', 'timestamp_nanosec',
                # IMU Verileri
                'imu_ori_x', 'imu_ori_y', 'imu_ori_z', 'imu_ori_w',
                'imu_ang_vel_x', 'imu_ang_vel_y', 'imu_ang_vel_z',
                'imu_lin_acc_x', 'imu_lin_acc_y', 'imu_lin_acc_z',
                # ODOM Verileri
                'odom_pos_x', 'odom_pos_y', 'odom_pos_z',
                'odom_ori_x', 'odom_ori_y', 'odom_ori_z', 'odom_ori_w',
                'odom_lin_vel_x', 'odom_lin_vel_y', 'odom_lin_vel_z',
                'odom_ang_vel_x', 'odom_ang_vel_y', 'odom_ang_vel_z',
                # OPTİK AKIŞ HIZ VERİLERİ (YENİ EKLENDİ)
                'opt_vel_x', 'opt_vel_y', 'opt_vel_z',
                # PWM (Servo) Verileri
                'servo1', 'servo2', 'servo3', 'servo4', 
                'servo5', 'servo6', 'servo7', 'servo8'
            ]
            self.writer.writerow(header)
            self.file.flush()

        # En güncel PWM verilerini tutacağımız liste
        self.latest_pwm = None 

        # ==========================================
        # 2. MAVLINK BAĞLANTISI VE TIMER (PWM İÇİN)
        # ==========================================
        self.connect_mavlink()
        self.pwm_timer = self.create_timer(0.02, self.pwm_timer_callback)

        # ==========================================
        # 3. ZAMAN SENKRONİZASYONLU ROS ABONELİKLERİ
        # ==========================================
        self.imu_sub = message_filters.Subscriber(self, Imu, '/imu0')
        self.odom_sub = message_filters.Subscriber(self, Odometry, '/odom')
        # YENİ SENSÖRÜMÜZÜ EKLİYORUZ
        self.opt_vel_sub = message_filters.Subscriber(self, Vector3Stamped, '/optical_flow/velocity')
        
        # Synchronizer artık 3 farklı topic'i aynı zaman damgasında yakalayacak
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.imu_sub, self.odom_sub, self.opt_vel_sub], 
            queue_size=20, 
            slop=0.1
        )
        self.ts.registerCallback(self.sync_callback)

        self.get_logger().info(f"Tüm veriler (IMU, Odom, Optik Hız, PWM) '{self.csv_file}' dosyasına yazılıyor...")
        self.get_logger().info("Durdurmak için CTRL+C yapabilirsiniz.")

    def connect_mavlink(self):
        try:
            self.connection = mavutil.mavlink_connection('tcp:127.0.0.1:5762')
            self.connection.wait_heartbeat(timeout=5)
            self.get_logger().info('MAVLink bağlantısı başarılı.')
            
            frequency_hz = 50
            self.connection.mav.command_long_send(
                self.connection.target_system,
                self.connection.target_component,
                mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
                0,
                36, # SERVO_OUTPUT_RAW mesaj ID'si
                1e6 / frequency_hz,
                0, 0, 0, 0, 0
            )
        except Exception as e:
            self.get_logger().error(f"MAVLink bağlantı hatası: {e}")

    def pwm_timer_callback(self):
        # MAVLink Buffer (Tampon) Şişmesini Engelleyen Kritik Fix
        latest_msg = None
        while True:
            msg = self.connection.recv_match(type='SERVO_OUTPUT_RAW', blocking=False)
            if not msg:
                break
            latest_msg = msg
            
        if latest_msg:
            self.latest_pwm = [
                float(latest_msg.servo1_raw), float(latest_msg.servo2_raw),
                float(latest_msg.servo3_raw), float(latest_msg.servo4_raw),
                float(latest_msg.servo5_raw), float(latest_msg.servo6_raw),
                float(latest_msg.servo7_raw), float(latest_msg.servo8_raw)
            ]

    # CALLBACK ARTIK 3 DEĞİŞKEN ALIYOR (IMU, ODOM, OPT_VEL)
    def sync_callback(self, imu_msg, odom_msg, opt_vel_msg):
        # Eğer henüz hiç PWM verisi gelmediyse bekle
        self.get_logger().info("BAŞARILI: IMU, Odom ve Kamera zamanları eşleşti!")
        if self.latest_pwm is None:
            self.get_logger().warn("HATA: Sensörler eşleşti ama PWM (Motor) verisi yok! CSV'ye yazılamıyor.")
            return
        self.get_logger().info("HARİKA: Tüm veriler tamam, CSV'ye 1 satır yazılıyor!")
        # Bütün veriler eşleşti! Hepsini tek satıra diziyoruz:
        row = [
            imu_msg.header.stamp.sec,
            imu_msg.header.stamp.nanosec,
            
            # IMU Sütunları
            imu_msg.orientation.x, imu_msg.orientation.y, imu_msg.orientation.z, imu_msg.orientation.w,
            imu_msg.angular_velocity.x, imu_msg.angular_velocity.y, imu_msg.angular_velocity.z,
            imu_msg.linear_acceleration.x, imu_msg.linear_acceleration.y, imu_msg.linear_acceleration.z,
            
            # ODOM Sütunları
            odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y, odom_msg.pose.pose.position.z,
            odom_msg.pose.pose.orientation.x, odom_msg.pose.pose.orientation.y, odom_msg.pose.pose.orientation.z, odom_msg.pose.pose.orientation.w,
            odom_msg.twist.twist.linear.x, odom_msg.twist.twist.linear.y, odom_msg.twist.twist.linear.z,
            odom_msg.twist.twist.angular.x, odom_msg.twist.twist.angular.y, odom_msg.twist.twist.angular.z,
            
            # OPTİK AKIŞ HIZ SÜTUNLARI (YENİ EKLENDİ)
            opt_vel_msg.vector.x, opt_vel_msg.vector.y, opt_vel_msg.vector.z,
            
            # PWM Sütunları
            *self.latest_pwm
        ]
        
        self.writer.writerow(row)

        
    def destroy_node(self):
        if hasattr(self, 'file') and not self.file.closed:
            self.file.close()
            self.get_logger().info('CSV dosyası güvenli bir şekilde kapatıldı.')
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = UltimateDataCollector()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()