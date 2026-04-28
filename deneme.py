import gurobipy as gp
from gurobipy import GRB

# ==========================================
# 1. BÖLÜM: KÜMELER VE ÖRNEK VERİ SETİ (DATASET)
# ==========================================

# Kümeler (Sets)
# ==========================================
# GÜNCEL 1. BÖLÜM: KÜMELER VE VERİ SETİ
# ==========================================

# Tüm Araç Tipleri Sisteme Eklendi
merkezler = ['Istanbul', 'Ankara', 'Izmir']
arac_tipleri = ['TIR', 'Kamyon', 'Hafif Kamyon', 'Kamyonet'] # Eksik olanlar eklendi
mulkiyet = ['Kiralik', 'Spot']

# Talep Verisi D_ij (Desi)
talep = {
    ('Istanbul', 'Ankara'): 4500,
    ('Istanbul', 'Izmir'): 2000,
    ('Ankara', 'Istanbul'): 1500,
    ('Ankara', 'Izmir'): 1000,
    ('Izmir', 'Istanbul'): 3000,
    ('Izmir', 'Ankara'): 500
}

# 4 Araç Tipi İçin Kapasiteler Cap_v (Desi) - Örnek Değerler
kapasite = {
    'TIR': 3000, 
    'Kamyon': 1500, 
    'Hafif Kamyon': 1000, 
    'Kamyonet': 500
}

# Transfer Merkezi Elleçleme Kapasiteleri H_i (Desi)
ellecleme_kapasitesi = {'Istanbul': 20000, 'Ankara': 15000, 'Izmir': 10000}

# TIR Yanaşma İzni Dock_i
tir_yanasma = {'Istanbul': 1, 'Ankara': 1, 'Izmir': 0}

# Şirket Envanterindeki Kiralık Araç Sayısı N_v (Tüm tipler eklendi)
envanter = {
    'TIR': 2, 
    'Kamyon': 5, 
    'Hafif Kamyon': 8, 
    'Kamyonet': 10
}

# Rota, Araç ve Mülkiyet Bazlı Sefer Maliyetleri C_ijvk (TL)
maliyet = {}
mesafe_carpani = {
    ('Istanbul', 'Ankara'): 4, ('Ankara', 'Istanbul'): 4,
    ('Istanbul', 'Izmir'): 5, ('Izmir', 'Istanbul'): 5,
    ('Ankara', 'Izmir'): 6, ('Izmir', 'Ankara'): 6
}

for i in merkezler:
    for j in merkezler:
        if i != j:
            for v in arac_tipleri:
                for k in mulkiyet:
                    taban_fiyat = mesafe_carpani[(i,j)] * 1000 # Mesafeye göre baz fiyat
                    
                    # Araç tipine göre maliyet çarpanı
                    if v == 'TIR': 
                        arac_carpani = 2.0
                    elif v == 'Kamyon': 
                        arac_carpani = 1.5
                    elif v == 'Hafif Kamyon': 
                        arac_carpani = 1.0
                    else: # Kamyonet
                        arac_carpani = 0.6 
                        
                    mulkiyet_carpani = 3.0 if k == 'Spot' else 1.0
                    
                    maliyet[i, j, v, k] = taban_fiyat * arac_carpani * mulkiyet_carpani

Big_M = 100000

# ==========================================
# 2. BÖLÜM: GUROBİ MODELİNİN KURULMASI
# ==========================================

# Modeli oluşturuyoruz
m = gp.Model("Middle_Mile_Optimizasyonu")

# --- KARAR DEĞİŞKENLERİ ---
# X_ijvk: Gönderilecek araç sayısı (Tam Sayı - GRB.INTEGER)
X = m.addVars(merkezler, merkezler, arac_tipleri, mulkiyet, vtype=GRB.INTEGER, name="Araç_Sayısı")

# Y_ijvk: Araçlara yüklenecek kargo miktarı (Sürekli Sayı - GRB.CONTINUOUS)
Y = m.addVars(merkezler, merkezler, arac_tipleri, mulkiyet, vtype=GRB.CONTINUOUS, name="Yük_Miktarı")

# --- AMAÇ FONKSİYONU ---
# Toplam taşıma maliyetini minimize et
m.setObjective(
    gp.quicksum(maliyet[i, j, v, k] * X[i, j, v, k] 
                for i in merkezler for j in merkezler if i != j 
                for v in arac_tipleri for k in mulkiyet),
    GRB.MINIMIZE
)

# --- KISITLAR (CONSTRAINTS) ---

for i in merkezler:
    for j in merkezler:
        if i != j:
            # Kısıt 1: Talep Karşılama (Her rotadaki toplam talep, araçlara yüklenen toplam yüke eşit olmalı)
            m.addConstr(
                gp.quicksum(Y[i, j, v, k] for v in arac_tipleri for k in mulkiyet) == talep[(i, j)],
                name=f"Talep_{i}_{j}"
            )
            
            # Kısıt 2: Araç Kapasitesi (Yüklenen miktar, gönderilen araçların toplam kapasitesini aşamaz)
            for v in arac_tipleri:
                for k in mulkiyet:
                    m.addConstr(
                        Y[i, j, v, k] <= kapasite[v] * X[i, j, v, k],
                        name=f"Kapasite_{i}_{j}_{v}_{k}"
                    )

# Kısıt 3: Kiralık Araç Envanteri (Tüm rotalardaki toplam kiralık araç kullanımı envanteri aşamaz)
for v in arac_tipleri:
    m.addConstr(
        gp.quicksum(X[i, j, v, 'Kiralik'] for i in merkezler for j in merkezler if i != j) <= envanter[v],
        name=f"Envanter_{v}"
    )

# Kısıt 4: Transfer Merkezi Elleçleme Kapasitesi (Giren + Çıkan yük)
for i in merkezler:
    # i'den çıkanlar
    cikan_yuk = gp.quicksum(Y[i, j, v, k] for j in merkezler if i != j for v in arac_tipleri for k in mulkiyet)
    # i'ye girenler
    giren_yuk = gp.quicksum(Y[j, i, v, k] for j in merkezler if j != i for v in arac_tipleri for k in mulkiyet)
    
    m.addConstr(cikan_yuk + giren_yuk <= ellecleme_kapasitesi[i], name=f"Ellecleme_{i}")

# Kısıt 5: TIR Yanaşma Durumu (Eğer merkeze TIR yanaşamıyorsa, X değişkenini sıfıra zorla)
for i in merkezler:
    # Merkezden çıkan TIR'lar
    m.addConstr(
        gp.quicksum(X[i, j, 'TIR', k] for j in merkezler if i != j for k in mulkiyet) <= Big_M * tir_yanasma[i],
        name=f"TIR_Cikis_{i}"
    )
    # Merkeze giren TIR'lar
    m.addConstr(
        gp.quicksum(X[j, i, 'TIR', k] for j in merkezler if j != i for k in mulkiyet) <= Big_M * tir_yanasma[i],
        name=f"TIR_Giris_{i}"
    )

# ==========================================
# 3. BÖLÜM: ÇÖZÜM VE ÇIKTILARIN EKRANA YAZDIRILMASI
# ==========================================

# Ekrandaki log karmaşasını gizle, sadece sonuçları göster
m.setParam('OutputFlag', 0)

# Optimizasyonu başlat
m.optimize()

# Sonuçları İnceleme
if m.status == GRB.OPTIMAL:
    print(f"--- OPTİMAL ÇÖZÜM BULUNDU ---")
    print(f"Toplam Minimum Operasyon Maliyeti: {m.objVal:,.2f} TL\n")
    print("Planlanan Seferler ve Araç Yükleri:")
    print("-" * 50)
    
    for i in merkezler:
        for j in merkezler:
            if i != j:
                for v in arac_tipleri:
                    for k in mulkiyet:
                        arac_sayisi = X[i, j, v, k].x
                        if arac_sayisi > 0.5: # 0'dan büyükse (Float hassasiyeti için >0.5)
                            yuk_miktari = Y[i, j, v, k].x
                            doluluk_orani = (yuk_miktari / (arac_sayisi * kapasite[v])) * 100
                            
                            print(f"Rota: {i} -> {j}")
                            print(f"Atanan Araç: {int(arac_sayisi)} adet {k} {v}")
                            print(f"Yüklenen: {yuk_miktari} Desi (Araç Başı Doluluk: %{doluluk_orani:.1f})\n")
else:
    print("Mevcut kısıtlarla uygun bir çözüm bulunamadı (Infeasible).")