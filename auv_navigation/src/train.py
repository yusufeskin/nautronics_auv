#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import pandas as pd
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray
from datetime import datetime
import signal
import sys

class RosbagListener(Node):
    def __init__(self):
        super().__init__('rosbag_listener_node')
        
        # Subscriber'lar
        self.create_subscription(Imu, '/imu0', self.imu_callback, 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(Float64MultiArray, '/auv/pwm_data', self.pwm_callback, 10)
        
        self.get_logger().info("✅ Node başladı! Şimdi 'ros2 bag play' yap...")

        # Veri listesi
        self.data_buffer = []
        
        # Son değerler
        self.latest_pwm = [1500.0] * 8
        self.latest_imu = None
        
        # Dosya ismi
        self.save_path = f"egitim_verisi_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        # Sayaç
        self.record_count = 0
        
        # Shutdown kontrolü
        self.is_shutdown = False

    def pwm_callback(self, msg):
        data = list(msg.data)
        if len(data) < 8:
            data += [1500.0] * (8 - len(data))
        self.latest_pwm = data[:8]

    def imu_callback(self, msg):
        self.latest_imu = msg

    def odom_callback(self, msg):
        if self.latest_imu is None:
            return
        
        ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        
        row = [
            ts,
            # IMU
            self.latest_imu.linear_acceleration.x,
            self.latest_imu.linear_acceleration.y,
            self.latest_imu.linear_acceleration.z,
            self.latest_imu.angular_velocity.x,
            self.latest_imu.angular_velocity.y,
            self.latest_imu.angular_velocity.z,
            # Odom
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z,
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.z,
            # PWM
            *self.latest_pwm
        ]
        
        self.data_buffer.append(row)
        self.record_count += 1
        
        if self.record_count % 100 == 0:
            print(f"📊 {self.record_count} kayıt toplandı... | Hız X: {msg.twist.twist.linear.x:.3f}", end='\r')

    def save_data(self):
        if self.is_shutdown:  # İki kere çağrılmasını engelle
            return
        self.is_shutdown = True
        
        if len(self.data_buffer) == 0:
            print("\n⚠️ HİÇ VERİ YOK! Bag oynatıldı mı?")
            return

        print(f"\n💾 {len(self.data_buffer)} satır CSV'ye yazılıyor...")
        
        columns = [
            'timestamp',
            'imu_linear_acc_x', 'imu_linear_acc_y', 'imu_linear_acc_z',
            'imu_angular_vel_x', 'imu_angular_vel_y', 'imu_angular_vel_z',
            'odom_pos_x', 'odom_pos_y', 'odom_pos_z',
            'odom_vel_x', 'odom_vel_y', 'odom_vel_z',
            'pwm_0', 'pwm_1', 'pwm_2', 'pwm_3', 'pwm_4', 'pwm_5', 'pwm_6', 'pwm_7'
        ]
        
        df = pd.DataFrame(self.data_buffer, columns=columns)
        df.to_csv(self.save_path, index=False)
        
        print(f"✅ BAŞARILI: {self.save_path}")
        print(f"📈 Toplam Satır: {len(df)}")
        if len(df) > 0:
            sure = df['timestamp'].max() - df['timestamp'].min()
            print(f"⏱️  Süre: {sure:.2f} saniye")

def main(args=None):
    # ROS2 zaten çalışıyorsa hata vermesin
    if rclpy.ok():
        print("⚠️ ROS2 zaten çalışıyor, önce kapatıyorum...")
        rclpy.shutdown()
    
    rclpy.init(args=args)
    listener = RosbagListener()
    
    # Ctrl+C handler
    def signal_handler(sig, frame):
        print("\n⏹️  Ctrl+C algılandı, kaydediliyor...")
        listener.save_data()
        listener.destroy_node()
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except:
            pass
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        rclpy.spin(listener)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"❌ HATA: {e}")
    finally:
        listener.save_data()
        listener.destroy_node()
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except:
            pass

if __name__ == '__main__':
    main()