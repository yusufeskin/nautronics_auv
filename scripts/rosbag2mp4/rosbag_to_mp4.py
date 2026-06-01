#!/usr/bin/env python3
"""
rosbag_to_mp4.py
────────────────────────────────────────────────────────────────
ROS2 bag (.db3) → MP4 converter

Kullanım:
    python rosbag_to_mp4.py --bag /path/to/bag_folder --topic /camera/image_raw
    python rosbag_to_mp4.py --bag /path/to/bag_folder

Gereksinimler:
    pip install rosbags opencv-python numpy
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    from rosbags.rosbag2 import Reader
    from rosbags.typesys import Stores, get_typestore
except ImportError:
    print("[ERROR] 'rosbags' kütüphanesi bulunamadı.")
    print("  Kur: pip install rosbags")
    sys.exit(1)


def find_image_topics(reader) -> list[str]:
    topics = []
    for conn in reader.connections:
        if conn.msgtype == "sensor_msgs/msg/Image":
            topics.append(conn.topic)
    return topics


def imgmsg_to_array(msg) -> np.ndarray:
    dtype = np.uint8
    n_ch = 3
    enc = msg.encoding.lower()

    if enc in ("mono8", "8uc1"):
        dtype, n_ch = np.uint8, 1
    elif enc in ("mono16", "16uc1"):
        dtype, n_ch = np.uint16, 1
    elif enc in ("rgb8", "bgr8"):
        dtype, n_ch = np.uint8, 3
    elif enc in ("rgba8", "bgra8"):
        dtype, n_ch = np.uint8, 4
    elif enc.startswith("bayer_"):
        dtype, n_ch = np.uint8, 1
    else:
        dtype, n_ch = np.uint8, 3

    img = np.frombuffer(msg.data, dtype=dtype)

    if n_ch == 1:
        img = img.reshape((msg.height, msg.width))
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif n_ch == 3:
        img = img.reshape((msg.height, msg.width, 3))
        if enc == "rgb8":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif n_ch == 4:
        img = img.reshape((msg.height, msg.width, 4))
        if enc == "rgba8":
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    if dtype == np.uint16:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    return img


def bag_to_mp4(bag_path: str, topic: str | None, output: str, fps: float):

    bag_path = Path(bag_path)
    if not bag_path.exists():
        print(f"[ERROR] Bag klasörü bulunamadı: {bag_path}")
        sys.exit(1)

    typestore = get_typestore(Stores.ROS2_HUMBLE)

    with Reader(bag_path) as reader:

        available = find_image_topics(reader)

        if not available:
            print("[ERROR] Bag dosyasında sensor_msgs/msg/Image topic'i yok!")
            print("  Mevcut topic'ler:")
            for c in reader.connections:
                print(f"    {c.topic}  [{c.msgtype}]")
            sys.exit(1)

        if topic is None:
            topic = available[0]
            print(f"[INFO] Topic otomatik seçildi: {topic}")
            if len(available) > 1:
                print(f"[INFO] Diğer mevcut image topic'ler: {available[1:]}")
        elif topic not in available:
            print(f"[ERROR] '{topic}' bulunamadı. Mevcut image topic'ler:")
            for t in available:
                print(f"    {t}")
            sys.exit(1)

        conns = [c for c in reader.connections if c.topic == topic]
        total_msgs = sum(c.msgcount for c in conns)
        print(f"[INFO] Topic: {topic}")
        print(f"[INFO] Toplam mesaj sayısı: {total_msgs}")

        writer = None
        count = 0
        skipped = 0

        print(f"[INFO] Dönüştürme başlıyor...")

        for conn, timestamp, rawdata in reader.messages(connections=conns):
            try:
                msg = typestore.deserialize_cdr(rawdata, conn.msgtype)
                frame = imgmsg_to_array(msg)
            except Exception as e:
                skipped += 1
                if skipped <= 5:
                    print(f"  [WARN] Kare atlandı ({skipped}): {e}")
                continue

            h, w = frame.shape[:2]

            if writer is None:
                output_path = Path(output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
                print(f"[INFO] Video boyutu: {w}x{h} @ {fps} FPS")
                print(f"[INFO] Çıktı: {output_path}")

            writer.write(frame)
            count += 1

            if count % 100 == 0:
                print(f"  {count}/{total_msgs} kare işlendi...", end="\r")

    if writer:
        writer.release()

    print(f"\n[DONE] {count} kare yazıldı → {output}")
    if skipped > 0:
        print(f"[WARN] {skipped} kare hatalı olduğu için atlandı.")

    return count


def main():
    parser = argparse.ArgumentParser(
        description="ROS2 bag dosyasını MP4'e çevirir",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnekler:
  python rosbag_to_mp4.py --bag ./my_bag
  python rosbag_to_mp4.py --bag ./my_bag --topic /camera/image_raw --fps 15
  python rosbag_to_mp4.py --bag ./my_bag --output sualty_kayit.mp4
        """
    )
    parser.add_argument("--bag", required=True, help="ROS2 bag klasör yolu")
    parser.add_argument("--topic", default=None, help="Image topic adı (varsayılan: otomatik)")
    parser.add_argument("--output", default="output.mp4", help="Çıktı MP4 dosyası (varsayılan: output.mp4)")
    parser.add_argument("--fps", type=float, default=10.0, help="Video FPS (varsayılan: 10)")

    args = parser.parse_args()
    bag_to_mp4(args.bag, args.topic, args.output, args.fps)


if __name__ == "__main__":
    main()