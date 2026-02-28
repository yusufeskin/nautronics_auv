import cv2
import numpy as np

# 1. Fotoğrafları yükle (Kendi dosya yollarınla değiştir)
img1 = cv2.imread('/home/murat/Pictures/Screenshots/qwe.png')
img2 = cv2.imread('/home/murat/Pictures/Screenshots/rty.png')

# 2. Gri formata çevir
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
mask = np.ones(img1.shape[:2], dtype="uint8") * 255

# 3. Parametreleri tanımla


feature_params = dict(maxCorners=400, qualityLevel=0.2, minDistance=8, blockSize=7)
lk_params = dict(winSize=(21, 21), maxLevel=3, 
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

# 4. İlk fotoğraftaki iyi noktaları (goodFeaturesToTrack) bul
p0 = cv2.goodFeaturesToTrack(gray1, mask=None, **feature_params)

if p0 is not None:
    # 5. İkinci fotoğraftaki noktaları Lucas-Kanade ile bul
    p1, st, err = cv2.calcOpticalFlowPyrLK(gray1, gray2, p0, None, **lk_params)

    # Durumu 1 olan (başarıyla takip edilen) noktaları filtrele
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    # İkinci fotoğraf üzerinde çizimleri yap
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        
        a, b, c, d = int(a), int(b), int(c), int(d)
        
        # Hareket çizgisi (mavi)
        cv2.line(img2, (a, b), (c, d), (255, 0, 0), 2)
        # Yeni nokta (kırmızı)
        cv2.circle(img2, (a, b), 5, (0, 0, 255), -1)
        # Eski nokta (yeşil)
        cv2.circle(img2, (c, d), 5, (0, 255, 0), -1)

    cv2.imshow('Iki Fotograf Arasindaki Flow', img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("İlk fotoğrafta takip edilecek köşe bulunamadı.")