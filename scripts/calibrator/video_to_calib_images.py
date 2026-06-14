#!/usr/bin/env python3
"""
video_to_calib_images.py
────────────────────────────────────────────────────────────────
MP4 → Kalibrasyon görüntüleri (calib_images/ klasörüne)

Kalite filtreleri:
  1. Bulanıklık filtresi     — Laplacian varyansı düşük kareleri atar
  2. Benzerlik filtresi      — Bir önceki kabul edilen kareye çok benzeyen atlar
  3. ChArUco ön kontrolü     — Yeterli marker içermeyen kareleri atar

Kullanım:
    python video_to_calib_images.py --video output.mp4
    python video_to_calib_images.py --video output.mp4 --target 150 --blur 80
    python video_to_calib_images.py --video output.mp4 --target 200 --skip 3
    

Gereksinimler:
    pip install opencv-contrib-python numpy
"""

import argparse
import sys
from pathlib import Path

import cv2
import cv2.aruco as aruco
import numpy as np


# ─── PARAMETRELER ───────────────────────────────────────────────────────────

ARUCO_DICT  = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
BOARD       = aruco.CharucoBoard((5, 7), 0.04, 0.02, ARUCO_DICT)
MIN_MARKERS = 4  # Kare kabul için minimum ArUco marker sayısı


# ─── KALİTE FONKSİYONLARI ───────────────────────────────────────────────────

def blur_score(gray: np.ndarray) -> float:
    """
    Laplacian varyansı — yüksek = keskin, düşük = bulanık.
    Sualtı için eşik genellikle 50–100 arası iyidir.
    """
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def frame_diff_score(gray1: np.ndarray, gray2: np.ndarray) -> float:
    """
    İki kare arasındaki farkın ortalaması.
    Düşük = çok benzer, yüksek = farklı.
    """
    if gray1 is None or gray2 is None:
        return 999.0
    diff = cv2.absdiff(gray1, gray2)
    return float(diff.mean())


def has_enough_markers(gray: np.ndarray, min_markers: int = MIN_MARKERS) -> tuple[bool, int]:
    """
    Karede yeterli ArUco marker var mı kontrol eder.
    Döndürür: (yeterli_mi, bulunan_marker_sayısı)
    """
    OPENCV_MAJOR = int(cv2.__version__.split('.')[0])
    OPENCV_MINOR = int(cv2.__version__.split('.')[1])
    use_new_api  = (OPENCV_MAJOR > 4) or (OPENCV_MAJOR == 4 and OPENCV_MINOR >= 7)

    if use_new_api:
        params = aruco.DetectorParameters()
        params.adaptiveThreshWinSizeMin  = 5
        params.adaptiveThreshWinSizeMax  = 25
        params.adaptiveThreshWinSizeStep = 4
        detector = aruco.ArucoDetector(ARUCO_DICT, params)
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        corners, ids, _ = aruco.detectMarkers(gray, ARUCO_DICT)

    count = len(ids) if ids is not None else 0
    return count >= min_markers, count


# ─── ANA FONKSİYON ──────────────────────────────────────────────────────────

def extract_frames(
    video_path: str,
    output_dir: str,
    target: int,
    blur_thresh: float,
    diff_thresh: float,
    frame_skip: int,
    check_markers: bool,
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Video açılamadı: {video_path}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_video    = cap.get(cv2.CAP_PROP_FPS)
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[INFO] Video: {video_path}")
    print(f"[INFO] {total_frames} kare, {fps_video:.1f} FPS, {width}x{height}")
    print(f"[INFO] Hedef: {target} kalibrasyon görüntüsü")
    print(f"[INFO] Filtreler → bulanıklık eşiği: {blur_thresh}, fark eşiği: {diff_thresh}")
    print(f"[INFO] Her {frame_skip} karede bir işleniyor\n")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    saved          = 0
    processed      = 0
    skipped_blur   = 0
    skipped_diff   = 0
    skipped_marker = 0
    last_saved_gray = None
    frame_idx       = 0

    while saved < target:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # Her frame_skip karede bir işle
        if frame_idx % frame_skip != 0:
            continue

        processed += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ── 1. Bulanıklık Filtresi ──────────────────────────────────────────
        score = blur_score(gray)
        if score < blur_thresh:
            skipped_blur += 1
            print(f"  [BLUR]   Kare {frame_idx:5d} — skor: {score:.1f} < {blur_thresh}  → atlandı")
            continue

        # ── 2. Benzerlik Filtresi ───────────────────────────────────────────
        diff = frame_diff_score(last_saved_gray, gray)
        if diff < diff_thresh:
            skipped_diff += 1
            # Bu çok sık olacak, sadece her 50'de bir göster
            if skipped_diff % 50 == 1:
                print(f"  [SAME]   Kare {frame_idx:5d} — fark: {diff:.2f} < {diff_thresh}  → atlandı")
            continue

        # ── 3. Marker Kontrolü (opsiyonel) ─────────────────────────────────
        if check_markers:
            ok, n_markers = has_enough_markers(gray)
            if not ok:
                skipped_marker += 1
                print(f"  [NOBOARD] Kare {frame_idx:5d} — {n_markers} marker ({MIN_MARKERS} gerekli) → atlandı")
                continue

        # ── Kaydet ─────────────────────────────────────────────────────────
        saved += 1
        filename = out_path / f"frame_{saved:04d}.png"
        cv2.imwrite(str(filename), frame)
        last_saved_gray = gray

        marker_info = ""
        if check_markers:
            _, n = has_enough_markers(gray)
            marker_info = f", {n} marker"

        print(f"  [SAVED]  {saved:3d}/{target}  kare {frame_idx:5d}  "
              f"bulanıklık:{score:.0f}{marker_info}  →  {filename.name}")

    cap.release()

    # ── Özet ────────────────────────────────────────────────────────────────
    print("\n" + "="*55)
    print(f"  Kaydedilen görüntü   : {saved}")
    print(f"  İşlenen kare         : {processed}")
    print(f"  Bulanık (atlandı)    : {skipped_blur}")
    print(f"  Benzer   (atlandı)   : {skipped_diff}")
    if check_markers:
        print(f"  Marker yok (atlandı) : {skipped_marker}")
    print(f"  Klasör               : {out_path.resolve()}")
    print("="*55)

    if saved < 10:
        print("\n[WARN] Çok az görüntü kaydedildi!")
        print("  → --blur değerini düşürmeyi deneyin (örn. --blur 30)")
        print("  → --diff değerini düşürmeyi deneyin (örn. --diff 5)")
        print("  → --skip değerini düşürmeyi deneyin (örn. --skip 1)")
    elif saved < target:
        print(f"\n[WARN] Hedef {target} görüntüye ulaşılamadı ({saved} kaydedildi).")
        print("  → Video daha kısa olabilir veya filtreler çok sıkı.")
    else:
        print(f"\n[OK] {saved} kalibrasyon görüntüsü hazır!")
        print(f"  Şimdi çalıştırabilirsiniz: python calibrate_charuco.py")


# ─── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Video'dan kalibrasyon görüntüsü çıkarır",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnekler:
  python video_to_calib_images.py --video output.mp4
  python video_to_calib_images.py --video output.mp4 --target 200 --blur 50
  python video_to_calib_images.py --video output.mp4 --no-marker-check
  python video_to_calib_images.py --video output.mp4 --skip 2 --diff 8
Sualtı için önerilen:
  --blur 40-60   (normal ışıkta 80-120)
  --diff 8-12    (yavaş kamera hareketi için düşür)
  --skip 2-5     (30fps videoda her 3. kare = etkin 10fps)
        """
    )
    parser.add_argument("--video",   required=True,    help="Giriş MP4 dosyası")
    parser.add_argument("--output",  default="calib_images", help="Çıktı klasörü (varsayılan: calib_images)")
    parser.add_argument("--target",  type=int,   default=150,  help="Hedef görüntü sayısı (varsayılan: 150)")
    parser.add_argument("--blur",    type=float, default=60.0, help="Min bulanıklık skoru (varsayılan: 60)")
    parser.add_argument("--diff",    type=float, default=10.0, help="Min kare farkı (varsayılan: 10)")
    parser.add_argument("--skip",    type=int,   default=3,    help="Her N. kare işlenir (varsayılan: 3)")
    parser.add_argument("--no-marker-check", action="store_true",
                        help="ArUco marker kontrolünü devre dışı bırak (daha hızlı)")

    args = parser.parse_args()

    extract_frames(
        video_path    = args.video,
        output_dir    = args.output,
        target        = args.target,
        blur_thresh   = args.blur,
        diff_thresh   = args.diff,
        frame_skip    = args.skip,
        check_markers = not args.no_marker_check,
    )


if __name__ == "__main__":
    main()