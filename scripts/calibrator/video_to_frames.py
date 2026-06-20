#!/usr/bin/env python3
"""
extract_all_frames.py
────────────────────────────────────────────────────────────────
MP4 → Tüm kareleri belirtilen FPS hızında (varsayılan 30) dışarı aktarır.
Hiçbir filtre (bulanıklık, ArUco vs.) uygulamaz.

Kullanım:
    python extract_all_frames.py --video output.mp4
    python extract_all_frames.py --video output.mp4 --output frames_klasoru
"""

import argparse
import sys
from pathlib import Path
import cv2

def extract_frames_at_fps(video_path: str, output_dir: str, target_fps: float = 30.0):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[HATA] Video açılamadı: {video_path}")
        sys.exit(1)

    # Videonun temel özelliklerini alıyoruz
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[BİLGİ] Video: {video_path}")
    print(f"[BİLGİ] Orijinal Değerler: {total_frames} kare, {original_fps:.1f} FPS, {width}x{height}")
    print(f"[BİLGİ] Hedef: Çıktı olarak saniyede {target_fps} kare alınacak.\n")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    saved_count = 0
    frame_idx = 0
    
    sec_per_frame = 1.0 / target_fps
    next_save_time = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        if current_time >= next_save_time:
            saved_count += 1
            filename = out_path / f"frame_{saved_count:05d}.png"
            cv2.imwrite(str(filename), frame)
            
            next_save_time += sec_per_frame
            
            if saved_count % 100 == 0:
                print(f"  [KAYDEDİLDİ] {saved_count:5d}. kare çıkarıldı → Zaman: {current_time:.2f} sn")

        frame_idx += 1

    cap.release()

    print("\n" + "="*55)
    print(f"  Toplam Okunan Kare : {frame_idx}")
    print(f"  Kaydedilen Görüntü : {saved_count} (Hedeflenen {target_fps} FPS hızında)")
    print(f"  Çıktı Klasörü      : {out_path.resolve()}")
    print("="*55)
    print("[TAMAMLANDI] Tüm kareler başarıyla ayrıldı.\n")


def main():
    parser = argparse.ArgumentParser(description="Videodan 30 FPS'ye sabitlenerek tüm kareleri çıkarır.")
    parser.add_argument("--video", required=True, help="Giriş MP4 dosyası")
    parser.add_argument("--output", default="extracted_frames", help="Çıktı klasörü (varsayılan: extracted_frames)")
    parser.add_argument("--fps", type=float, default=30.0, help="Çıkarılacak hedeflenen FPS (varsayılan: 30)")

    args = parser.parse_args()

    extract_frames_at_fps(
        video_path=args.video,
        output_dir=args.output,
        target_fps=args.fps
    )

if __name__ == "__main__":
    main()