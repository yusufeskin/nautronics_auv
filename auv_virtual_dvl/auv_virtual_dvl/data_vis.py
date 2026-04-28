import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def gorsellestir_karsilastirma(csv_yolu):
    # Dosya yolunu genişlet
    tam_yol = os.path.expanduser(csv_yolu)
    
    if not os.path.exists(tam_yol):
        print(f"❌ Hata: CSV dosyası bulunamadı! Yol: {tam_yol}")
        return

    print(f"'{tam_yol}' okunuyor...")
    df = pd.read_csv(tam_yol)

    # 1. Zamanı Düzenleme (Sıfırdan Başlayan Göreceli Zaman)
    # Saniye ve nanosaniyeyi birleştirerek tam zamanı buluyoruz
    df['tam_zaman'] = df['zaman_sec'] + (df['zaman_nanosec'] * 1e-9)
    # İlk ölçüm anını 0. saniye kabul ederek göreceli zaman yaratıyoruz
    df['göreceli_zaman'] = df['tam_zaman'] - df['tam_zaman'].iloc[0]

    # 2. Ortalama Hataları (MAE) Hesaplama
    mae_x = df['hata_x'].mean()
    mae_y = df['hata_y'].mean()
    mae_z = df['hata_z'].mean()

    print("\n📊 ORTALAMA MUTLAK HATALAR (MAE):")
    print(f"X Ekseni (İleri/Geri) : {mae_x:.4f} m/s")
    print(f"Y Ekseni (Sağ/Sol)    : {mae_y:.4f} m/s")
    print(f"Z Ekseni (Aşağı/Yukarı): {mae_z:.4f} m/s")

    # 3. Grafik Çizimi
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    fig.suptitle('Gerçek Zamanlı AUV Hız Tahmini: TCN vs Odometri', fontsize=16, fontweight='bold')

    eksen_ayarlari = [
        {'gercek': 'gercek_odom_x', 'tahmin': 'tahmin_tcn_x', 'baslik': f'X Ekseni Hızı (MAE: {mae_x:.4f} m/s)', 'renk': 'blue'},
        {'gercek': 'gercek_odom_y', 'tahmin': 'tahmin_tcn_y', 'baslik': f'Y Ekseni Hızı (MAE: {mae_y:.4f} m/s)', 'renk': 'green'},
        {'gercek': 'gercek_odom_z', 'tahmin': 'tahmin_tcn_z', 'baslik': f'Z Ekseni Hızı (MAE: {mae_z:.4f} m/s)', 'renk': 'red'}
    ]

    zaman = df['göreceli_zaman']

    for i, ayar in enumerate(eksen_ayarlari):
        # Gerçek Odometri (Siyah ve kalın çizgi)
        axes[i].plot(zaman, df[ayar['gercek']], label='Gerçek Odom', color='black', alpha=0.7, linewidth=2)
        
        # TCN Tahmini (Renkli ve kesik çizgi)
        axes[i].plot(zaman, df[ayar['tahmin']], label='TCN Tahmini', color=ayar['renk'], linestyle='--', linewidth=1.5)
        
        axes[i].set_title(ayar['baslik'], fontsize=14)
        axes[i].set_ylabel('Hız (m/s)', fontsize=12)
        axes[i].legend(loc='upper right')
        axes[i].grid(True, alpha=0.4, linestyle=':')

    # En alt grafiğin X ekseni etiketi
    axes[2].set_xlabel('Geçen Süre (Saniye)', fontsize=12)

    plt.tight_layout()
    
    # Grafiği CSV ile aynı klasöre kaydet
    kayit_yolu = os.path.dirname(tam_yol) + '/realtime_grafik.png'
    plt.savefig(kayit_yolu, dpi=300)
    print(f"\n✅ Grafik '{kayit_yolu}' adıyla kaydedildi.")
    
    plt.show()

if __name__ == "__main__":
    # Düğümün (Node) veriyi kaydettiği dosya yolunu buraya veriyoruz
    dosya_yolu = '~/nautronics_ws/src/nautronics_auv/auv_virtual_dvl/data/realtime_comparison.csv'
    gorsellestir_karsilastirma(dosya_yolu)