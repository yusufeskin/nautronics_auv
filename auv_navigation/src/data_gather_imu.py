import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
import csv
import os

class ImuDataCollector(Node):
    def __init__(self):
        super().__init__('imu_data_collector')
        
        # Dosya adı (Çalıştırdığınız dizine kaydedilir)
        self.csv_file = 'imu_verileri.csv'
        
        # Dosyanın daha önceden var olup olmadığını kontrol et
        file_exists = os.path.isfile(self.csv_file)
        
        # Dosyayı 'a' (append/ekleme) modunda açıyoruz. 
        # Bu sayede kodu her kapattığınızda veya açtığınızda eski verilerin altına yazar.
        self.file = open(self.csv_file, mode='a', newline='')
        self.writer = csv.writer(self.file)
        
        # Eğer dosya ilk defa oluşturuluyorsa en üste başlıkları (header) yaz
        if not file_exists:
            header = [
                'timestamp_sec', 'timestamp_nanosec',
                'orientation_x', 'orientation_y', 'orientation_z', 'orientation_w',
                'angular_velocity_x', 'angular_velocity_y', 'angular_velocity_z',
                'linear_acceleration_x', 'linear_acceleration_y', 'linear_acceleration_z'
            ]
            self.writer.writerow(header)
            self.file.flush() # Başlığı anında diske kaydet

        # /imu0 topiğine abone oluyoruz
        self.subscription = self.create_subscription(
            Imu,
            '/imu0',
            self.listener_callback,
            10 # Kuyruk boyutu
        )
        self.get_logger().info(f"IMU verileri dinleniyor... Kayıt dosyası: '{self.csv_file}'")
        self.get_logger().info("Durdurmak için CTRL+C yapabilirsiniz.")

    def listener_callback(self, msg):
        # Topicten gelen verileri tek bir satır listesi haline getir
        row = [
            msg.header.stamp.sec,
            msg.header.stamp.nanosec,
            msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w,
            msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z,
            msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z
        ]
        # CSV dosyasına yaz
        self.writer.writerow(row)
        
    def destroy_node(self):
        # Düğüm kapanırken dosyayı güvenli bir şekilde kapat ki veri kaybı olmasın
        self.file.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ImuDataCollector()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Kullanıcı tarafından durduruldu. Dosya kaydedildi.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()