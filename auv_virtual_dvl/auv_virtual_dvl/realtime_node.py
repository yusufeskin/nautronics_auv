import rclpy
from rclpy.node import Node
import torch
import numpy as np
from collections import deque
import joblib
import os

# ROS 2 Mesaj Tipleri
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Vector3Stamped, Vector3
from std_msgs.msg import Float64MultiArray

# Kendi Modelini Import Et
from auv_virtual_dvl.models.regression import AUVVelocityEstimator
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")

class RealTimeVelocityEstimator(Node):
    def __init__(self):
        super().__init__('realtime_velocity_estimator')
        
        # ==========================================
        # 1. MODEL YÜKLEME VE GPU AYARLARI
        # ==========================================
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"Kullanılan Cihaz: {self.device}")

        self.model = AUVVelocityEstimator(
            input_channels=22, 
            num_channels=[64, 64, 128, 128], 
            kernel_size=3, 
            dropout=0.2,
            output_size=3
        )
        
        # Ağırlıkları yükle (Kendi tam yolunu buraya yaz)
        weights_path = os.path.expanduser('~/nautronics_ws/src/nautronics_auv/auv_virtual_dvl/data/weights/best_tcn_model.pth')
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval() # Çıkarım (Inference) moduna al
        self.get_logger().info("Model başarıyla yüklendi ve çıkarım moduna alındı.")

        # ==========================================
        # 2. SCALER YÜKLEME (DATA CLEANING İLE AYNI MANTIK)
        # ==========================================
        scaler_path = os.path.expanduser('~/nautronics_ws/src/nautronics_auv/auv_virtual_dvl/data/pkl_folder/auv_x_scaler.pkl')
        try:
            self.scaler = joblib.load(scaler_path)
            self.get_logger().info("Scaler başarıyla yüklendi.")
        except Exception as e:
            self.get_logger().error(f"Scaler yüklenemedi! Lütfen dosya yolunu kontrol et: {scaler_path}")
            raise e

        # ==========================================
        # 3. KAYAN PENCERE (SLIDING WINDOW) TAMPONU
        # ==========================================
        self.window_size = 30
        self.buffer = deque(maxlen=self.window_size)

        # ==========================================
        # 4. EN GÜNCEL VERİLERİ TUTACAK DEĞİŞKENLER
        # ==========================================
        self.latest_imu = None
        self.latest_pwm = None
        self.latest_opt_vel = None
        
        self.last_opt_time = 0.0 
        self.last_timer_time = self.get_clock().now().nanoseconds / 1e9 # dt hesabı için

        # ==========================================
        # 5. ABONELİKLER (Subscribers)
        # ==========================================
        self.create_subscription(Imu, '/imu0', self.imu_cb, 10)
        self.create_subscription(Float64MultiArray, '/auv/pwm_data', self.pwm_cb, 10)
        self.create_subscription(Vector3Stamped, '/optical_flow/velocity', self.opt_vel_cb, 10)

        # ==========================================
        # 6. YAYINCI (Publisher) - TAHMİN EDİLEN HIZ
        # ==========================================
        self.est_vel_pub = self.create_publisher(Vector3, '/auv/estimated_velocity', 10)

        # ==========================================
        # 7. GERÇEK ZAMANLI ÇIKARIM DÖNGÜSÜ (30 Hz)
        # ==========================================
        self.timer = self.create_timer(0.033, self.inference_step)
        self.get_logger().info("Gerçek Zamanlı Tahmin Düğümü Başlatıldı. Tampon dolması bekleniyor...")

    # --- CALLBACK FONKSİYONLARI ---
    def imu_cb(self, msg):
        self.latest_imu = msg

    def pwm_cb(self, msg):
        self.latest_pwm = msg.data

    def opt_vel_cb(self, msg):
        self.latest_opt_vel = msg
        self.last_opt_time = self.get_clock().now().nanoseconds / 1e9

    # --- ANA ÇIKARIM FONKSİYONU ---
    def inference_step(self):
        # IMU ve PWM hayati önem taşıyor, gelmediyse bekle
        if not (self.latest_imu and self.latest_pwm):
            return

        now_sec = self.get_clock().now().nanoseconds / 1e9
        
        # 1. dt Hesaplanması
        dt = now_sec - self.last_timer_time
        if dt <= 0 or dt > 1.0: 
            dt = 1.0 / 30.0
        self.last_timer_time = now_sec

        # 2. Optical Flow Ön İşlemesi
        opt_x, opt_y = 0.0, 0.0
        opt_valid = 0.0

        if self.latest_opt_vel and (now_sec - self.last_opt_time) < 0.5:
            opt_valid = 1.0
            opt_x = self.latest_opt_vel.vector.x * opt_valid
            opt_y = self.latest_opt_vel.vector.y * opt_valid

        # 3. PWM / Servo Standardizasyonu (-1.0 ile 1.0 arası)
        processed_pwm = []
        for pwm_val in self.latest_pwm[:8]:
            val = pwm_val if pwm_val != 0 else 1500.0
            norm_val = (val - 1500.0) / 400.0
            processed_pwm.append(norm_val)

        # 4. Girdi Vektörünü Oluşturma
        current_features = [
            dt,
            self.latest_imu.orientation.x, self.latest_imu.orientation.y, self.latest_imu.orientation.z, self.latest_imu.orientation.w,
            self.latest_imu.angular_velocity.x, self.latest_imu.angular_velocity.y, self.latest_imu.angular_velocity.z,
            self.latest_imu.linear_acceleration.x, self.latest_imu.linear_acceleration.y, self.latest_imu.linear_acceleration.z,
            opt_x, opt_y, opt_valid,
            *processed_pwm
        ]

        # 5. Vektörü Tampona Ekle
        self.buffer.append(current_features)

        # 6. Tampon Dolduysa Modeli Çalıştır
        if len(self.buffer) == self.window_size:
            self.run_model_inference()

    def run_model_inference(self):
        # 1. Buffer'ı Numpy array'ine çevir -> Şekil: (30, 22)
        x_data = np.array(self.buffer, dtype=np.float32)

        # ==========================================
        # SCALER UYGULAMASI (DATA CLEANING İLE BİREBİR AYNI)
        # Sadece ilk 14 sütun (Sensör verileri) normalize edilecek
        # Son 8 sütun (PWM verileri) dokunulmadan kalacak
        # ==========================================
        sensor_data = x_data[:, :14]
        pwm_data = x_data[:, 14:]
        
        sensor_data_scaled = self.scaler.transform(sensor_data)
        
        # Normalize edilmiş sensör verisi ile saf PWM verisini tekrar birleştir
        x_data_scaled = np.hstack((sensor_data_scaled, pwm_data))

        # 2. Tensöre çevir ve TCN formatına (Batch, Channels, Sequence_Length) getir
        x_tensor = torch.tensor(x_data_scaled, device=self.device, dtype=torch.float32)
        x_tensor = x_tensor.transpose(0, 1).unsqueeze(0) # -> (1, 22, 30)

        # 3. Modelden geçir (Gradient hesaplamasını kapat)
        with torch.no_grad():
            prediction = self.model(x_tensor)
            
            # CPU'ya alıp numpy'a çevir
            pred_np = prediction.squeeze().cpu().numpy() # Şekil: (3,) -> [v_x, v_y, v_z]

        # 4. Sonucu ROS üzerinden yayınla
        est_vel_msg = Vector3()
        est_vel_msg.x = float(pred_np[0])
        est_vel_msg.y = float(pred_np[1])
        est_vel_msg.z = float(pred_np[2])
        self.est_vel_pub.publish(est_vel_msg)

        # Ekrana yazdırarak kontrol et
        self.get_logger().info(f"Tahmini Hız -> X: {pred_np[0]:.3f}, Y: {pred_np[1]:.3f}, Z: {pred_np[2]:.3f} m/s")

def main(args=None):
    rclpy.init(args=args)
    node = RealTimeVelocityEstimator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()