import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import csv
import os

class OdomDataCollector(Node):
    def __init__(self):
        super().__init__('odom_data_collector')
        
        # Dosya adı
        self.csv_file = 'odom_verileri.csv'
        
        # Dosya önceden var mı kontrolü
        file_exists = os.path.isfile(self.csv_file)
        
        # Dosyayı 'a' (append) modunda aç
        self.file = open(self.csv_file, mode='a', newline='')
        self.writer = csv.writer(self.file)
        
        # Dosya yeniyse başlıkları yaz
        if not file_exists:
            header = [
                'timestamp_sec', 'timestamp_nanosec',
                'pos_x', 'pos_y', 'pos_z',
                'ori_x', 'ori_y', 'ori_z', 'ori_w',
                'linear_vel_x', 'linear_vel_y', 'linear_vel_z',
                'angular_vel_x', 'angular_vel_y', 'angular_vel_z'
            ]
            self.writer.writerow(header)
            self.file.flush()

        # /odom topiğine abone ol
        self.subscription = self.create_subscription(
            Odometry,
            '/odom',
            self.listener_callback,
            10
        )
        self.get_logger().info(f"Odometry verileri dinleniyor... Kayıt dosyası: '{self.csv_file}'")
        self.get_logger().info("Durdurmak için CTRL+C yapabilirsiniz.")

    def listener_callback(self, msg):
        # Topicten gelen verileri çek ve listeye çevir
        row = [
            msg.header.stamp.sec,
            msg.header.stamp.nanosec,
            msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z,
            msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w,
            msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z,
            msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z
        ]
        
        # CSV dosyasına yaz
        self.writer.writerow(row)
        
    def destroy_node(self):
        # Veri kaybını önlemek için çıkışta dosyayı güvenli kapat
        self.file.close()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = OdomDataCollector()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Kullanıcı tarafından durduruldu. Dosya kaydedildi.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()