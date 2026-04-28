import torch
from torch.utils.data import DataLoader
from auv_virtual_dvl.scripts.sliding_windows import AUVSlidingWindowDataset
# Önceden yazdığımız AUVSlidingWindowDataset sınıfının burada import edildiğini varsayıyoruz (edildi)

# 1. Hiperparametreleri Belirleme
BATCH_SIZE = 64  # GPU'ya tek seferde gönderilecek pencere sayısı (VRAM'e göre 128 de yapılabilir)
WINDOW_SIZE = 30 # 30 Hz için 1 saniyelik geçmiş

import os
# 2. Dataset'leri Başlatma (Bir önceki adımda böldüğümüz CSV'leri kullanıyoruz)
print("Veri setleri yükleniyor...")
train_dataset = AUVSlidingWindowDataset(os.path.expanduser('~/projects/tcn_velocity_estimator/data/train_data/train_data.csv'), window_size=WINDOW_SIZE)
val_dataset = AUVSlidingWindowDataset(os.path.expanduser('~/projects/tcn_velocity_estimator/data/val_data/val_data.csv'), window_size=WINDOW_SIZE)
test_dataset = AUVSlidingWindowDataset(os.path.expanduser('~/projects/tcn_velocity_estimator/data/test_data/test_data.csv'), window_size=WINDOW_SIZE)

# 3. DataLoader'ları Oluşturma
# num_workers: Veriyi hazırlamak için işlemcinin kaç çekirdeğini kullanacağı (Hızlandırır)
# pin_memory: Verilerin RAM'den ekran kartına (GPU) aktarımını çok daha hızlı hale getirir
train_loader = DataLoader(
    dataset=train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True,          # SADECE TRAIN ESNASINDA pencerelerin sırasını karıştırırız ki ağ ezberlemesin
    num_workers=4,         
    pin_memory=True        
)

val_loader = DataLoader(
    dataset=val_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False,         # Doğrulama setinde zaman akışını bozmuyoruz!
    num_workers=4, 
    pin_memory=True
)

test_loader = DataLoader(
    dataset=test_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False,         # Test setinde de zaman akışını bozmuyoruz!
    num_workers=4, 
    pin_memory=True
)

# 4. Sistemin Nasıl Çalıştığını Test Etme (Anlamak İçin)
print("\nDataLoader Testi:")
for batch_x, batch_y in train_loader:
    print(f"Ağa Girecek Girdi (X) Matrisinin Boyutu: {batch_x.shape}")
    print(f"Ağın Tahmin Edeceği Hedef (Y) Boyutu: {batch_y.shape}")
    break # Sadece ilk paketi (batch) görüp döngüyü kırıyoruz