import pandas as pd
import matplotlib.pyplot as plt

# Dosyayı okuyun
df = pd.read_csv('~/nautronics_ws/src/nautronics_auv/auv_virtual_dvl/data/realtime_comparison.csv')

# Zaman bilgisine ihtiyaç olmadan, doğrudan verilerin endeksine (satır sırasına) göre grafik çizimi
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 10))

# --- 1. GRAFİK: X Değerleri ---
axes[0].plot(df.index, df['gercek_odom_x'], label='Gerçek Odom X', color='blue', linewidth=2)
axes[0].plot(df.index, df['tahmin_tcn_x'], label='Tahmin TCN X', color='red', linestyle='--', linewidth=2)
axes[0].set_title('X Ekseni: Gerçek vs Tahmin')
axes[0].set_xlabel('Örneklem (Satır Numarası)')
axes[0].set_ylabel('X Değeri')
axes[0].legend()
axes[0].grid(True, linestyle=':', alpha=0.7)

# --- 2. GRAFİK: Y Değerleri ---
axes[1].plot(df.index, df['gercek_odom_y'], label='Gerçek Odom Y', color='green', linewidth=2)
axes[1].plot(df.index, df['tahmin_tcn_y'], label='Tahmin TCN Y', color='orange', linestyle='--', linewidth=2)
axes[1].set_title('Y Ekseni: Gerçek vs Tahmin')
axes[1].set_xlabel('Örneklem (Satır Numarası)')
axes[1].set_ylabel('Y Değeri')
axes[1].legend()
axes[1].grid(True, linestyle=':', alpha=0.7)

# Düzen ve gösterme
plt.tight_layout()
plt.show()