import torch
import torch.nn as nn
from auv_virtual_dvl.models.tcn import TemporalConvNet

class AUVVelocityEstimator(nn.Module):
    def __init__(self, input_channels, num_channels, kernel_size, dropout, output_size=3):
        """
        input_channels: Sensör sayısı (Senin verinde 25)
        num_channels: TCN gizli katmanlarındaki filtre sayısı (örn: [64, 128, 256])
        output_size: Tahmin edilecek hız sayısı (v_x, v_y, v_z için 3)
        """
        super(AUVVelocityEstimator, self).__init__()
        
        # Orijinal LocusLab TCN çekirdeği
        self.tcn = TemporalConvNet(
            num_inputs=input_channels, 
            num_channels=num_channels, 
            kernel_size=kernel_size, 
            dropout=dropout
        )
        
        # Sonuçları [v_x, v_y, v_z] olarak bağlayan son lineer katman
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        # x'in boyutu DataLoader'dan gelen: (Batch, Channels=25, Sequence=30)
        
        # TCN'den geçir (Boyut değişmez, sadece kanallar genişler/daralır)
        y1 = self.tcn(x) 
        
        # Bize sadece en son zaman adımı lazım!
        # y1[:, :, -1] demek: Tüm batchleri al, tüm kanalları al, ama zaman eksenindeki SADECE SON (30.) adımı al
        last_step = y1[:, :, -1]
        
        # 3 adet hız tahmini üret
        out = self.linear(last_step)
        
        return out # Softmax YOK! Doğrudan sayıları döndürüyoruz.