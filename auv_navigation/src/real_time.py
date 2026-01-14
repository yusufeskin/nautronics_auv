#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import torch
import torch.nn as nn
import numpy as np
import joblib
from collections import deque
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from std_msgs.msg import Float64MultiArray

# --- MODEL MİMARİSİ (Eğitimdekiyle BİREBİR AYNI OLMALI) ---
class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, padding, dropout):
        super(TCNBlock, self).__init__()
        self.chomp_size = padding 
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = out[:, :, :-self.chomp_size] 
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.conv2(out)
        out = out[:, :, :-self.chomp_size]
        out = self.relu2(out)
        out = self.dropout2(out)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class AUV_Net(nn.Module):
    def __init__(self, input_size, output_size):
        super(AUV_Net, self).__init__()
        # Hiperparametreler eğitim koduyla aynı girildi
        self.tcn1 = TCNBlock(input_size, 64, 3, dilation=1, padding=2, dropout=0.2)
        self.tcn2 = TCNBlock(64, 64, 3, dilation=2, padding=4, dropout=0.2)
        self.tcn3 = TCNBlock(64, 64, 3, dilation=4, padding=8, dropout=0.2)
        self.fc = nn.Linear(64, output_size)

    def forward(self, x):
        y = self.tcn1(x)
        y = self.tcn2(y)
        y = self.tcn3(y)
        y = y[:, :, -1] 
        out = self.fc(y)
        return out

# --- ROS2 NODE ---
class RealTimePredictor(Node):
    def __init__(self):
        super().__init__('tcn_predictor_node')
        
        # 1. Modeli ve Scaler'ları Yükle
        self.get_logger().info("Model ve Scaler'lar yükleniyor...")
        try:
            self.scaler_x = joblib.load('/home/sye/nautronics_auv/src/auv_navigation/src/scaler_x.pkl')
            self.scaler_y = joblib.load('/home/sye/nautronics_auv/src/auv_navigation/src/scaler_y.pkl')
            
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = AUV_Net(input_size=14, output_size=3).to(self.device)
            self.model.load_state_dict(torch.load('/home/sye/nautronics_auv/src/auv_navigation/src/auv_tcn_model.pth', map_location=self.device))
            self.model.eval() # Inference modu
            self.get_logger().info(f"✅ Model yüklendi! Cihaz: {self.device}")
        except Exception as e:
            self.get_logger().error(f"DOSYA HATASI: {e}")
            self.get_logger().error("Lütfen .pth ve .pkl dosyalarının bu klasörde olduğundan emin ol!")
            exit()

        # 2. Veri Hafızası (Window Size = 50)
        self.window_size = 50
        self.input_buffer = deque(maxlen=self.window_size)
        
        # Son alınan anlık değerler
        self.latest_pwm = [1500.0] * 8
        self.latest_imu_lin = [0.0, 0.0, 9.8]
        self.latest_imu_ang = [0.0, 0.0, 0.0]
        self.latest_odom_vel = [0.0, 0.0, 0.0] # Karşılaştırma için

        # 3. Subscriberlar
        self.create_subscription(Float64MultiArray, '/auv/pwm_data', self.pwm_cb, 10)
        self.create_subscription(Imu, '/imu0', self.imu_cb, 10)
        self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        
        # Publisher (Tahmin edilen hızı yayınlayalım)
        self.pred_pub = self.create_publisher(Float64MultiArray, '/auv/predicted_velocity', 10)

        # 4. Timer (50 Hz - Modelin Kalbi)
        # Veri ne sıklıkta gelirse gelsin, modeli eğittiğimiz frekansta (0.02s) beslemeliyiz.
        self.create_timer(0.02, self.control_loop)
        self.get_logger().info("🚀 Tahmin Motoru Başlatıldı (50 Hz)...")

    def pwm_cb(self, msg):
        data = list(msg.data)
        if len(data) < 8: data += [1500.0] * (8 - len(data))
        self.latest_pwm = data[:8]

    def imu_cb(self, msg):
        self.latest_imu_ang = [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]
        self.latest_imu_lin = [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z]

    def odom_cb(self, msg):
        # Gerçek veriyi sadece karşılaştırma (Validate) için saklıyoruz
        self.latest_odom_vel = [
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.z
        ]

    def control_loop(self):
        # 1. Şu anki Feature Vektörünü Oluştur (14 Elemanlı)
        # Sıralama eğitimdekiyle AYNI olmalı: 8 PWM + 3 LinAcc + 3 AngVel
        current_features = self.latest_pwm + self.latest_imu_lin + self.latest_imu_ang
        
        # 2. Ölçekle (Scale)
        # Scaler 2D array bekler, o yüzden [current_features] şeklinde veriyoruz
        scaled_features = self.scaler_x.transform([current_features])[0]
        
        # 3. Hafızaya At (Sliding Window)
        self.input_buffer.append(scaled_features)

        # 4. Yeterli Veri Biriktiyse Tahmin Yap
        if len(self.input_buffer) == self.window_size:
            # Model girdisi: (Batch=1, Features=14, Sequence=50)
            input_tensor = torch.FloatTensor(list(self.input_buffer)).unsqueeze(0).transpose(1, 2).to(self.device)
            
            with torch.no_grad():
                prediction_scaled = self.model(input_tensor).cpu().numpy()
            
            # Gerçek birime dönüştür (m/s)
            prediction_real = self.scaler_y.inverse_transform(prediction_scaled)[0]
            
            # 5. Sonuçları Yayınla ve Karşılaştır
            self.publish_prediction(prediction_real)
            self.print_comparison(prediction_real)

    def publish_prediction(self, pred):
        msg = Float64MultiArray()
        msg.data = pred.tolist() # [Surge, Sway, Heave]
        self.pred_pub.publish(msg)

    def print_comparison(self, pred):
        # Gerçek vs Tahmin farkını göster
        real = self.latest_odom_vel
        
        diff_x = abs(real[0] - pred[0])
        color = "\033[92m" if diff_x < 0.1 else "\033[91m" # Hata azsa Yeşil, çoksa Kırmızı
        reset = "\033[0m"
        
        print(f"SURGE (X) | Gerçek: {real[0]:.3f} | AI: {color}{pred[0]:.3f}{reset} | Fark: {diff_x:.3f}")
        # print(f"SWAY  (Y) | Gerçek: {real[1]:.3f} | AI: {pred[1]:.3f}") # İstersen aç

def main(args=None):
    rclpy.init(args=args)
    node = RealTimePredictor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()