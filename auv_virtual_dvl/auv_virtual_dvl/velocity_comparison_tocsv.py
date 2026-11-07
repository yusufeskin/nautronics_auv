import rclpy
from rclpy.node import Node
import csv
import os

# ROS 2 Mesaj Tipleri
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Vector3

class VelocityComparisonLogger(Node):
    def __init__(self):
        super().__init__('velocity_comparison_logger')
        
        # ==========================================
        # 1. DOSYA AYARLARI
        # ==========================================
        # CSV dosyasının kaydedileceği yol (Kendine göre düzenleyebilirsin)
        self.csv_file = os.path.expanduser('~/nautronics_ws/src/nautronics_auv/auv_virtual_dvl/data/realtime_comparison.csv')
        
        # Klasör yoksa oluştur
        os.makedirs(os.path.dirname(self.csv_file), exist_ok=True)
        file_exists = os.path.isfile(self.csv_file)
        
        self.file = open(self.csv_file, mode='a', newline='')
        self.writer = csv.writer(self.file)
        
        # Dosya yeniyse başlıkları (Header) yaz
        if not file_exists:
            header = [
                'zaman_sec', 'zaman_nanosec',
                'gercek_odom_x', 'gercek_odom_y', 'gercek_odom_z',
                'tahmin_tcn_x', 'tahmin_tcn_y', 'tahmin_tcn_z',
                'hata_x', 'hata_y', 'hata_z' # Sonradan analizi kolaylaştırmak için hata payları
            ]
            self.writer.writerow(header)
            self.file.flush()

        # ==========================================
        # 2. EN GÜNCEL VERİLERİ TUTACAK DEĞİŞKENLER
        # ==========================================
        self.latest_odom_vel = None
        self.latest_est_vel = None

        # ==========================================
        # 3. ABONELİKLER (Subscribers)
        # ==========================================
        # Gerçek Odom verisi
        self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        # TCN'in ürettiği tahmin verisi
        self.create_subscription(Vector3, '/auv/estimated_velocity', self.est_vel_cb, 10)

        # ==========================================
        # 4. KAYIT TIMER'I (30 Hz)
        # ==========================================
        self.timer = self.create_timer(0.033, self.record_to_csv)
        self.get_logger().info("Karşılaştırma Kaydedici Başlatıldı. Veriler bekleniyor...")

    # --- CALLBACK FONKSİYONLARI ---
    def odom_cb(self, msg):
        # Odom mesajından sadece lineer hızları (twist) çekiyoruz
        self.latest_odom_vel = msg.twist.twist.linear

    def est_vel_cb(self, msg):
        # TCN'in yayınladığı Vector3 mesajı
        self.latest_est_vel = msg

    # --- KAYIT FONKSİYONU ---
    def record_to_csv(self):
        # İki veri de henüz sisteme düşmediyse bekle
        if not (self.latest_odom_vel and self.latest_est_vel):
            return

        now = self.get_clock().now()
        
        # Değerleri al
        odom_x = self.latest_odom_vel.x
        odom_y = self.latest_odom_vel.y
        odom_z = self.latest_odom_vel.z
        
        est_x = self.latest_est_vel.x
        est_y = self.latest_est_vel.y
        est_z = self.latest_est_vel.z

        # Mutlak hataları anlık olarak hesapla
        err_x = abs(odom_x - est_x)
        err_y = abs(odom_y - est_y)
        err_z = abs(odom_z - est_z)

        # CSV Satırını Oluştur
        row = [
            now.seconds_nanoseconds()[0],
            now.seconds_nanoseconds()[1],
            odom_x, odom_y, odom_z,
            est_x, est_y, est_z,
            err_x, err_y, err_z
        ]
        
        # Virgülden sonra çok uzamaması için yuvarlayarak formatlama yapabiliriz
        row_formatted = [f"{val:.6f}" if isinstance(val, float) else val for val in row]

        self.writer.writerow(row_formatted)
        self.file.flush() # Veri kaybını önlemek için anında diske yaz

    def destroy_node(self):
        # Düğüm kapatılırken dosyayı güvenli bir şekilde kapat
        if hasattr(self, 'file') and not self.file.closed:
            self.file.close()
            self.get_logger().info('Karşılaştırma CSV dosyası güvenli bir şekilde kapatıldı.')
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = VelocityComparisonLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()