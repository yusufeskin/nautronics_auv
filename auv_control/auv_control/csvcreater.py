import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Vector3Stamped
from std_msgs.msg import Float64MultiArray
import csv
import os
import time
class UltimateDataCollector(Node):
    def __init__(self):
        super().__init__('ultimate_data_collector')
        
        # ==========================================
        # 1. DOSYA AYARLARI (Tam yol vermeyi unutma!)
        # ==========================================
        self.csv_file = '/home/murat/nautronics_auv/src/auv_control/auv_control/tum_sensor_ve_pwm_verileri.csv'
        file_exists = os.path.isfile(self.csv_file)
        
        self.file = open(self.csv_file, mode='a', newline='')
        self.writer = csv.writer(self.file)
        
        if not file_exists:
            header = [
                'sys_time_sec', 'sys_time_nanosec',
                # IMU Verileri
                'imu_ori_x', 'imu_ori_y', 'imu_ori_z', 'imu_ori_w',
                'imu_ang_vel_x', 'imu_ang_vel_y', 'imu_ang_vel_z',
                'imu_lin_acc_x', 'imu_lin_acc_y', 'imu_lin_acc_z',
                # ODOM Verileri
                'odom_pos_x', 'odom_pos_y', 'odom_pos_z',
                'odom_ori_x', 'odom_ori_y', 'odom_ori_z', 'odom_ori_w',
                'odom_lin_vel_x', 'odom_lin_vel_y', 'odom_lin_vel_z',
                'odom_ang_vel_x', 'odom_ang_vel_y', 'odom_ang_vel_z',
                # OPTİK AKIŞ (Kamera)
                'opt_vel_x', 'opt_vel_y', 'opt_vel_z', 'opt_flow_valid',
                # PWM Verileri
                'servo1', 'servo2', 'servo3', 'servo4', 
                'servo5', 'servo6', 'servo7', 'servo8'
            ]
            self.writer.writerow(header)
            self.file.flush()

        # ==========================================
        # 2. EN GÜNCEL VERİLERİ TUTACAK DEĞİŞKENLER
        # ==========================================
        self.latest_imu = None
        self.latest_odom = None
        self.latest_pwm = None
        
        # Kamera verisinin kopma ihtimaline karşı zaman tutucu
        self.latest_opt_vel = None
        self.last_opt_time = 0.0 

        # ==========================================
        # 3. BAĞIMSIZ ROS ABONELİKLERİ (Synchronizer İptal!)
        # ==========================================
        self.create_subscription(Imu, '/imu0', self.imu_cb, 10)
        self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        self.create_subscription(Float64MultiArray, '/auv/pwm_data', self.pwm_cb, 10)
        self.create_subscription(Vector3Stamped, '/optical_flow/velocity', self.opt_vel_cb, 10)

        # ==========================================
        # 4. KAYIT TIMER'I (FPS Belirleyici - 30 Hz)
        # ==========================================
        # 1.0 / 30.0 = 0.033 saniyede bir çalışır
        self.record_timer = self.create_timer(0.033, self.record_state_to_csv)

        self.get_logger().info("YENİ MİMARİ: Veri toplayıcı başlatıldı. 30 FPS hızında kaydedilecek...")

    # --- CALLBACK FONKSİYONLARI (Sadece değişkenleri günceller) ---
    def imu_cb(self, msg):
        self.latest_imu = msg

    def odom_cb(self, msg):
        self.latest_odom = msg

    def pwm_cb(self, msg):
        # rc_data_reader'dan gelen Float64MultiArray verisi
        self.latest_pwm = msg.data

    def opt_vel_cb(self, msg):
        self.latest_opt_vel = msg
        self.last_opt_time = self.get_clock().now().nanoseconds / 1e9

    # --- ANA KAYIT FONKSİYONU ---
    def record_state_to_csv(self):
        # Temel veriler (IMU, Odom, PWM) henüz hiç gelmediyse kayda başlama
        if not (self.latest_imu and self.latest_odom and self.latest_pwm):
            return

        now = self.get_clock().now()
        current_time_sec = now.nanoseconds / 1e9

        # OPTICAL FLOW KONTROLÜ (Veri koptuysa Model Eğitimi için 0 yaz)
        opt_x, opt_y, opt_z = 0.0, 0.0, 0.0
        opt_valid = 0

        # Eğer son 0.5 saniyedir kamera verisi gelmediyse sıfırla (kör nokta)
        if self.latest_opt_vel and (current_time_sec - self.last_opt_time) < 0.5:
            opt_x = self.latest_opt_vel.vector.x
            opt_y = self.latest_opt_vel.vector.y
            opt_z = self.latest_opt_vel.vector.z
            opt_valid = 1

        # CSV Satırını Oluştur
        row = [
            now.seconds_nanoseconds()[0],
            now.seconds_nanoseconds()[1],
            
            # IMU
            self.latest_imu.orientation.x, self.latest_imu.orientation.y, self.latest_imu.orientation.z, self.latest_imu.orientation.w,
            self.latest_imu.angular_velocity.x, self.latest_imu.angular_velocity.y, self.latest_imu.angular_velocity.z,
            self.latest_imu.linear_acceleration.x, self.latest_imu.linear_acceleration.y, self.latest_imu.linear_acceleration.z,
            
            # ODOM
            self.latest_odom.pose.pose.position.x, self.latest_odom.pose.pose.position.y, self.latest_odom.pose.pose.position.z,
            self.latest_odom.pose.pose.orientation.x, self.latest_odom.pose.pose.orientation.y, self.latest_odom.pose.pose.orientation.z, self.latest_odom.pose.pose.orientation.w,
            self.latest_odom.twist.twist.linear.x, self.latest_odom.twist.twist.linear.y, self.latest_odom.twist.twist.linear.z,
            self.latest_odom.twist.twist.angular.x, self.latest_odom.twist.twist.angular.y, self.latest_odom.twist.twist.angular.z,
            
            # OPTICAL FLOW
            opt_x, opt_y, opt_z, opt_valid,
            
            # PWM
            *self.latest_pwm
        ]
        
        self.writer.writerow(row)
        self.file.flush() # ANINDA DİSKE YAZ!

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