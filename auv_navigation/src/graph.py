import pandas as pd
import matplotlib.pyplot as plt

# --- BURAYA SENİN CSV DOSYANIN ADINI YAZ ---
csv_file = '/home/sye/nautronics_auv/src/auv_navigation/src/egitim_verisi_20260109_011437.csv'  # Dosya ismini kontrol et!

try:
    df = pd.read_csv(csv_file)
    
    # Zamanı 0'dan başlat
    df['time'] = df['timestamp'] - df['timestamp'].iloc[0]

    plt.figure(figsize=(12, 10))

    # GRAFİK 1: Surge (İleri/Geri) İlişkisi
    plt.subplot(2, 1, 1)
    plt.title("Motor Komutu vs. Gerçek Hız (İleri/Geri)")
    # Genelde PWM 4 veya 5 Surge'dür. (Mavi Çizgi)
    plt.plot(df['time'], df['pwm_4'], color='blue', label='PWM Komutu (Motor)', alpha=0.6)
    plt.ylabel('PWM', color='blue')
    plt.axhline(1500, color='black', linestyle='--', alpha=0.3)
    
    # İkinci eksen (Hız için)
    ax2 = plt.gca().twinx()
    # Hız (Kırmızı Çizgi)
    ax2.plot(df['time'], df['odom_vel_x'], color='red', label='Gerçek Hız (m/s)', linewidth=2)
    ax2.set_ylabel('Hız (m/s)', color='red')
    
    # GRAFİK 2: Yaw (Dönme) İlişkisi
    plt.subplot(2, 1, 2)
    plt.title("Dönme Komutu vs. Dönme Hızı (Yaw)")
    # Genelde PWM 3 Yaw'dır.
    plt.plot(df['time'], df['pwm_3'], color='green', label='Dönme Komutu', alpha=0.6)
    plt.ylabel('PWM', color='green')
    plt.axhline(1500, color='black', linestyle='--', alpha=0.3)

    ax3 = plt.gca().twinx()
    ax3.plot(df['time'], df['odom_vel_z'], color='purple', label='Dönme Hızı (rad/s)', linewidth=2) # Angular Z hızı
    ax3.set_ylabel('Açısal Hız', color='purple')

    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"Hata: {e}")
    print("CSV dosyasının adını koda doğru yazdığından emin ol!")