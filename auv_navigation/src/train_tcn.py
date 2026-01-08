import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import joblib 
import matplotlib.pyplot as plt

# --- AYARLAR ---
CSV_FILE_PATH = '/home/sye/nautronics_auv/src/auv_navigation/src/egitim_verisi_20260109_011437.csv' 
MODEL_SAVE_PATH = 'auv_tcn_model.pth'
SCALER_X_PATH = 'scaler_x.pkl'
SCALER_Y_PATH = 'scaler_y.pkl'

# HİPERPARAMETRELER
WINDOW_SIZE = 50   
BATCH_SIZE = 32    
EPOCHS = 100       
LEARNING_RATE = 0.001
HIDDEN_CHANNELS = 64
KERNEL_SIZE = 3
DROPOUT = 0.2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Eğitim şu cihazda yapılacak: {device}")

# --- 1. VERİ HAZIRLIĞI ---
def load_and_process_data(csv_path):
    df = pd.read_csv(csv_path)
    
    # GİRDİLER 
    feature_cols = [
        'pwm_0', 'pwm_1', 'pwm_2', 'pwm_3', 'pwm_4', 'pwm_5', 'pwm_6', 'pwm_7',
        'imu_linear_acc_x', 'imu_linear_acc_y', 'imu_linear_acc_z',
        'imu_angular_vel_x', 'imu_angular_vel_y', 'imu_angular_vel_z'
    ]
    
    # ÇIKTILAR (Hızlar)
    target_cols = [
        'odom_vel_x', 'odom_vel_y', 'odom_vel_z' 
    ]
    
    X = df[feature_cols].values
    y = df[target_cols].values
    
    return X, y

# --- 2. DATASET SINIFI ---
class AUVTimeDataset(Dataset):
    def __init__(self, X, y, window_size):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.window_size = window_size

    def __len__(self):
        return len(self.X) - self.window_size

    def __getitem__(self, idx):
        x_window = self.X[idx : idx + self.window_size]
        y_target = self.y[idx + self.window_size] 
        return x_window.transpose(0, 1), y_target

# --- 3. TCN MODEL MİMARİSİ (DÜZELTİLDİ) ---
class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, padding, dropout):
        super(TCNBlock, self).__init__()
        
        # Padding miktarı kadar sağdan kırpmak için saklıyoruz
        self.chomp_size = padding 
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                               padding=padding, dilation=dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 
                               padding=padding, dilation=dilation)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        # --- DÜZELTME BURADA: Fazla padding'i kesiyoruz (Chomp) ---
        out = out[:, :, :-self.chomp_size] 
        
        out = self.relu1(out)
        out = self.dropout1(out)
        
        out = self.conv2(out)
        # --- DÜZELTME BURADA: İkinci katmanda da kesiyoruz ---
        out = out[:, :, :-self.chomp_size]
        
        out = self.relu2(out)
        out = self.dropout2(out)
        
        res = x if self.downsample is None else self.downsample(x)
        
        # Artık boyutlar eşit, hata vermeyecek
        return self.relu(out + res)

class AUV_Net(nn.Module):
    def __init__(self, input_size, output_size):
        super(AUV_Net, self).__init__()
        # Padding değerlerini dilation * (kernel_size - 1) olarak ayarladık (Causal Convolutions)
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

# --- 4. ANA EĞİTİM DÖNGÜSÜ ---
def main():
    print("Veri yükleniyor...")
    X_raw, y_raw = load_and_process_data(CSV_FILE_PATH)
    
    train_size = int(len(X_raw) * 0.8)
    
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_raw = X_raw[:train_size]
    y_train_raw = y_raw[:train_size]
    X_val_raw = X_raw[train_size:]
    y_val_raw = y_raw[train_size:]
    
    X_train = scaler_x.fit_transform(X_train_raw)
    y_train = scaler_y.fit_transform(y_train_raw)
    X_val = scaler_x.transform(X_val_raw)
    y_val = scaler_y.transform(y_val_raw)
    
    joblib.dump(scaler_x, SCALER_X_PATH)
    joblib.dump(scaler_y, SCALER_Y_PATH)
    print("Scaler'lar kaydedildi.")

    train_dataset = AUVTimeDataset(X_train, y_train, WINDOW_SIZE)
    val_dataset = AUVTimeDataset(X_val, y_val, WINDOW_SIZE)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    input_dim = X_raw.shape[1] 
    output_dim = y_raw.shape[1] 
    
    model = AUV_Net(input_dim, output_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"Eğitim Başlıyor... Toplam Veri: {len(X_raw)}, Input Dim: {input_dim}, Output Dim: {output_dim}")
    
    train_losses = []
    val_losses = []

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {avg_train_loss:.5f}, Val Loss: {avg_val_loss:.5f}")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model kaydedildi: {MODEL_SAVE_PATH}")
    
    # Grafik kaydet (SSH kullandığın için ekrana basmak yerine kaydetmek daha iyi)
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Eğitim Süreci')
    plt.legend()
    plt.savefig('egitim_grafigi.png') # Dosyaya kaydeder
    print("Grafik 'egitim_grafigi.png' olarak kaydedildi.")

if __name__ == "__main__":
    main()