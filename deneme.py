import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import glob

class ImageSaverNode(Node):
    def __init__(self):
        super().__init__('image_saver_node')
        
        # Abone olunacak topic adı
        self.topic_name = '/camera/front'
        self.subscription = self.create_subscription(
            Image, 
            self.topic_name, 
            self.image_callback, 
            10
        )
        self.br = CvBridge()
        
        # Kaydedilecek klasörün yolu
        self.save_dir = os.path.expanduser('~/Desktop/auv_dataset')
        
        # Klasör yoksa oluştur
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            self.get_logger().info(f"Dataset klasoru olusturuldu: {self.save_dir}")

        # --- Sayaç Ayarları ---
        # Klasörü tarayıp en son kalınan numarayı bulur ve üstüne ekler
        self.saved_count = self.get_last_frame_number() + 1 
        self.frame_count = 0 # FPS throttle için sayaç
        
        self.get_logger().info(f"Kayıt {self.save_dir} icine yapilacak.")
        self.get_logger().info(f"Ilk fotograf 'frame_{self.saved_count:05d}.jpg' olarak kaydedilecek.")
        self.get_logger().info(f"Node basladi. '{self.topic_name}' dinleniyor (Yaklasik 4.8 FPS ile)...")

    def get_last_frame_number(self):
        """Klasördeki en yüksek frame_XXXXX.jpg numarasını bulur."""
        search_path = os.path.join(self.save_dir, "frame_*.jpg")
        files = glob.glob(search_path)
        
        if not files:
            return 0
            
        max_num = 0
        for f in files:
            try:
                # 'frame_00012.jpg' formatından 12 sayısını çeker
                base = os.path.basename(f)
                num_str = base.split('_')[1].split('.')[0]
                num = int(num_str)
                if num > max_num:
                    max_num = num
            except Exception:
                continue
        return max_num

    def image_callback(self, data):
        self.frame_count += 1
        
        # FPS Düşürme: 48 FPS'ten gelen verinin sadece her 10. karesini işler
        if self.frame_count % 10 != 0:
            return

        try:
            # ROS Image mesajını OpenCV BGR formatına çevir
            current_frame = self.br.imgmsg_to_cv2(data, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Goruntu donusturme hatasi: {e}")
            return

        # Dosya adını oluştur (Örn: frame_00001.jpg)
        filename = f"frame_{self.saved_count:05d}.jpg"
        save_path = os.path.join(self.save_dir, filename)

        # Görüntüyü kaydet
        cv2.imwrite(save_path, current_frame)
        self.get_logger().info(f"Kaydedildi: {filename}")
        
        # Başarıyla kaydedildiği için sayacı 1 artır
        self.saved_count += 1

def main(args=None):
    rclpy.init(args=args)
    node = ImageSaverNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Kullanici tarafindan durduruldu. Cikiliyor...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()