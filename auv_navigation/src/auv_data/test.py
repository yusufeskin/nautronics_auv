import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import matplotlib.pyplot as plt

# --- AYARLAR ---
CSV_FILE_PATH = '/home/sye/nautronics_auv/src/auv_navigation/src/egitim_verisi_20260109_011437.csv' 
MODEL_PATH = '/home/sye/nautronics_auv/src/auv_tcn_model.pth'
SCALER_X_PATH = 'scaler_x.pkl'
SCALER_Y_PATH = 'scaler_y.pkl'

# Parametreler (Eğitimle aynı)
WINDOW_SIZE = 50
HIDDEN_CHANNELS = 64
KERNEL_SIZE = 3
DROPOUT = 0.2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- MODEL MİMARİSİ ---
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
        self.tcn1 = TCNBlock(input_size, HIDDEN_CHANNELS, KERNEL_SIZE, dilation=1, padding=2, dropout=DROPOUT)
        self.tcn2 = TCNBlock(HIDDEN_CHANNELS, HIDDEN_CHANNELS, KERNEL_SIZE, dilation=2, padding=4, dropout=DROPOUT)
        self.tcn3 = TCNBlock(HIDDEN_CHANNELS, HIDDEN_CHANNELS, KERNEL_SIZE, dilation=4, padding=8, dropout=DROPOUT)
        self.fc = nn.Linear(HIDDEN_CHANNELS, output_size)

    def forward(self, x):
        y = self.tcn1(x)
        y = self.tcn2(y)
        y = self.tcn3(y)
        y = y[:, :, -1] 
        out = self.fc(y)
        return out

def main():
    print("Veriler yükleniyor...")
    df = pd.read_csv(CSV_FILE_PATH)
    
    # --- KRİTİK NOKTA: Sadece son %20'yi alıyoruz ---
    train_size = int(len(df) * 0.8)
    df_test = df[train_size:].reset_index(drop=True) # Görmediği veri
    
    print(f"Toplam Veri: {len(df)}")
    print(f"Test Edilecek (Görülmemiş) Veri Sayısı: {len(df_test)}")

    scaler_x = joblib.load(SCALER_X_PATH)
    scaler_y = joblib.load(SCALER_Y_PATH)
    
    feature_cols = [
        'pwm_0', 'pwm_1', 'pwm_2', 'pwm_3', 'pwm_4', 'pwm_5', 'pwm_6', 'pwm_7',
        'imu_linear_acc_x', 'imu_linear_acc_y', 'imu_linear_acc_z',
        'imu_angular_vel_x', 'imu_angular_vel_y', 'imu_angular_vel_z'
    ]
    target_cols = ['odom_vel_x', 'odom_vel_y', 'odom_vel_z']
    
    X_raw = df_test[feature_cols].values
    y_raw = df_test[target_cols].values
    
    # Dikkat: fit_transform DEĞİL, sadece transform yapıyoruz.
    # Çünkü model eğitilirkenki ölçeği kullanmalı.
    X_scaled = scaler_x.transform(X_raw)
    
    # Modeli Yükle
    input_dim = X_raw.shape[1]
    output_dim = y_raw.shape[1]
    model = AUV_Net(input_dim, output_dim).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    predictions = []
    targets = []
    
    print("Görülmemiş veri üzerinde tahmin yapılıyor...")
    with torch.no_grad():
        for i in range(len(X_scaled) - WINDOW_SIZE):
            window = X_scaled[i : i + WINDOW_SIZE]
            tensor_x = torch.FloatTensor(window).unsqueeze(0).transpose(1, 2).to(device)
            
            pred = model(tensor_x)
            predictions.append(pred.cpu().numpy()[0])
            targets.append(y_raw[i + WINDOW_SIZE])

    # Geri Dönüştür
    pred_array = np.array(predictions)
    pred_real = scaler_y.inverse_transform(pred_array)
    target_array = np.array(targets)

    # Grafik
    plt.figure(figsize=(14, 8))
    
    # Sadece Surge (İleri Hız) örneği
    plt.plot(target_array[:, 0], label='Gerçek (Ground Truth)', color='black', alpha=0.6, linewidth=2)
    plt.plot(pred_real[:, 0], label='Yapay Zeka Tahmini (Unseen Data)', color='red', linestyle='--', linewidth=2)
    
    plt.title("GÖRÜLMEMİŞ VERİ TESTİ: Surge Hızı")
    plt.xlabel("Zaman Adımı")
    plt.ylabel("Hız (m/s)")
    plt.legend()
    plt.grid()
    plt.savefig('gercek_sinav_sonucu.png')
    print("✅ Test bitti! 'gercek_sinav_sonucu.png' dosyasına bak.")
    # plt.show() # SSH kullanıyorsan kapat

if __name__ == "__main__":
    main()