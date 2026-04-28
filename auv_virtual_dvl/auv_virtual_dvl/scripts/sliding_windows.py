import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class AUVSlidingWindowDataset(Dataset):
    def __init__(self, csv_file, window_size=30, feature_cols=None, target_cols=None):
        """
        csv_file: İşlenmiş verinin yolu
        window_size: Geriye dönük kaç satıra (adıma) bakılacağı (örn: 30 Hz için 1 saniye = 30)
        feature_cols: TCN'e girdi (X) olarak verilecek sütun isimlerinin listesi
        target_cols: TCN'in tahmin etmeye çalışacağı (Y) sütun isimlerinin listesi
        """
        # Veriyi Pandas ile oku
        self.df = pd.read_csv(csv_file)
        self.window_size = window_size
        
        # Eğer sütun belirtilmemişse varsayılanları ayarla (kendi sütun isimlerine göre düzeltebilirsin)
        if feature_cols is None:
            # Hedefler hariç her şey girdi (X) kabul edilir
            self.feature_cols = [col for col in self.df.columns if col not in ['odom_lin_vel_x', 'odom_lin_vel_y', 'odom_lin_vel_z']]
        else:
            self.feature_cols = feature_cols
            
        if target_cols is None:
            self.target_cols = ['odom_lin_vel_x', 'odom_lin_vel_y', 'odom_lin_vel_z'] # Odom hız sütunlarının isimleri
        else:
            self.target_cols = target_cols

        # Pandas DataFrame'i işleme hızını artırmak için Numpy dizisine çeviriyoruz
        self.X_data = self.df[self.feature_cols].values.astype(np.float32)
        self.Y_data = self.df[self.target_cols].values.astype(np.float32)

        # Toplam veri sayısından pencere boyutunu çıkarıyoruz ki dizinin sonuna taşıp hata vermesin
        self.length = len(self.df) - self.window_size

    def __len__(self):
        # PyTorch'a bu veri setinden kaç tane "pencere" çıkarabileceğimizi söylüyoruz
        return self.length

    def __getitem__(self, idx):
        # Kayan Pencere (Sliding Window) mantığının çalıştığı asıl yer burası!
        
        # X: idx'ten başla, idx + window_size'a kadar olan satırları al (Örn: 0. satırdan 30. satıra kadar)
        x_window = self.X_data[idx : idx + self.window_size]
        
        # Y: Pencerenin tam bittiği anın hedefini (gerçek Odom hızını) al
        y_target = self.Y_data[idx + self.window_size - 1]

        # PyTorch Tensörüne dönüştür
        x_tensor = torch.tensor(x_window)
        y_tensor = torch.tensor(y_target)

        # ÇOK ÖNEMLİ TCN DÜZELTMESİ:
        # TCN (Conv1d) girdiyi (Batch, Channels, Sequence_Length) sırasında ister.
        # Bizim verimiz (Sequence_Length, Channels) sırasında. Bunu tersyüz (transpose) etmeliyiz.
        x_tensor = x_tensor.transpose(0, 1)

        return x_tensor, y_tensor