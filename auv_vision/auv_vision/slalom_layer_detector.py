"""
slalom_layer_detector.py
========================
Slalom gate layering algoritması.

Giriş (step 1+2'den geliyor):
  - YOLO segmentasyon sonuçları (class_name, bbox, mask_points, confidence)
  - Her duba için RealSense stereo depth (mask centroid üzerinden)

Çıkış:
  - SlalomLayer listesi, depth'e göre sıralı (Layer 0 = en yakın gate)
  - Her layer: 1 kırmızı duba + sol/sağ beyaz dubalar

Bumblebee'den farklı olarak:
  - TF tree / world frame yok  -> kamera frame'inde çalışıyoruz
  - Zamansal clustering yok   -> tek frame anlık karar
  - Odom yok                  -> RealSense depth tek kaynak
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import String
import cv2
import json


# ─────────────────────────────────────────────────────────────────────────────
# Veri yapıları
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BuoyDetection:
    """Tek bir duba tespiti — step 2'nin çıktısını temsil eder."""
    class_name: str           # 'red_buoy' | 'white_buoy'
    depth_m: float            # RealSense stereo'dan gelen mesafe (metre)
    center_x: float           # Görüntüdeki bbox/mask merkezi (piksel)
    center_y: float           # Görüntüdeki bbox/mask merkezi (piksel)
    mask_points: np.ndarray   # (N, 2) piksel koordinatları — PnP için
    confidence: float         # YOLO confidence skoru (0-1)
    bbox: Tuple[int, int, int, int] = field(default_factory=lambda: (0, 0, 0, 0))  # x, y, w, h


@dataclass
class SlalomLayer:
    """Bir slalom gate'ini temsil eder: 1 kırmızı + 0-2 beyaz duba."""
    layer_idx: int                        # 0 = en yakın
    red: BuoyDetection                    # Kırmızı duba (layer referansı)
    white_left: Optional[BuoyDetection]   # Kırmızı dubanın sol beyazı
    white_right: Optional[BuoyDetection]  # Kırmızı dubanın sağ beyazı

    @property
    def depth_m(self) -> float:
        """Layer'ın referans derinliği = kırmızı dubanın derinliği."""
        return self.red.depth_m

    @property
    def all_buoys(self) -> List[BuoyDetection]:
        """Layer'daki tüm dubalar."""
        buoys = [self.red]
        if self.white_left:
            buoys.append(self.white_left)
        if self.white_right:
            buoys.append(self.white_right)
        return buoys

    @property
    def is_complete(self) -> bool:
        """Her iki beyaz duba da tespit edildi mi?"""
        return self.white_left is not None and self.white_right is not None


# ─────────────────────────────────────────────────────────────────────────────
# Layerlama algoritması (saf Python — ROS bağımsız, test edilebilir)
# ─────────────────────────────────────────────────────────────────────────────

class SlalomLayerDetector:
    """
    Kırmızı ve beyaz duba tespitlerini depth'e göre gate layer'larına böler.

    Algoritma (Bumblebee'nin assign_to_centroids mantığından uyarlandı):
    1. Kırmızı dubaları depth'e göre sırala (en yakın = Layer 0)
    2. Yakın depth'teki kırmızıları birleştir (aynı fiziksel dubanın 2 tespiti)
    3. Her beyazı, depth farkı en az olan kırmızıya ata (1D k-means)
    4. Her layer için: beyaz dubaları kırmızının sağ/sol tarafına göre ayır
    5. Birden fazla sağ/sol varsa en yüksek confidence'lı olanı seç
    """

    def __init__(
        self,
        max_reds: int = 3,                # Slalom'da maksimum 3 gate
        min_depth_m: float = 0.3,         # Bu altı tespitler outlier
        max_depth_m: float = 8.0,         # Bu üstü tespitler outlier
        min_red_separation_m: float = 0.4,# İki kırmızı arasındaki min depth farkı
        max_white_depth_diff_m: float = 1.2, # Beyazın kırmızısından max depth sapması
        min_confidence: float = 0.35,     # Bu altı confidence reddedilir
    ):
        self.max_reds = max_reds
        self.min_depth_m = min_depth_m
        self.max_depth_m = max_depth_m
        self.min_red_separation_m = min_red_separation_m
        self.max_white_depth_diff_m = max_white_depth_diff_m
        self.min_confidence = min_confidence

    def detect_layers(
        self, detections: List[BuoyDetection]
    ) -> List[SlalomLayer]:
        """
        Duba tespitlerinden layer listesi oluşturur.

        Returns:
            depth'e göre sıralı SlalomLayer listesi.
            Boş liste = yeterli tespit yok.
        """
        if not detections:
            return []

        # 1. Filtrele: outlier depth ve düşük confidence
        valid = [
            d for d in detections
            if (self.min_depth_m <= d.depth_m <= self.max_depth_m
                and d.confidence >= self.min_confidence
                and not math.isnan(d.depth_m))
        ]

        reds = [d for d in valid if d.class_name == 'red_buoy']
        whites = [d for d in valid if d.class_name == 'white_buoy']

        if not reds:
            return []

        # 2. Kırmızıları depth'e göre sırala (en yakın önce)
        reds.sort(key=lambda d: d.depth_m)

        # 3. Confidence'a göre en iyi max_reds kırmızıyı al
        reds = sorted(reds[:self.max_reds * 2], key=lambda d: d.confidence, reverse=True)
        reds = reds[:self.max_reds]
        reds.sort(key=lambda d: d.depth_m)  # tekrar depth sırası

        # 4. Aynı fiziksel kırmızının çift tespitini birleştir
        merged_reds = self._merge_close_reds(reds)

        # 5. Her beyazı en yakın kırmızıya ata (1D k-means, depth ekseni)
        red_depths = np.array([r.depth_m for r in merged_reds])
        layer_whites: List[List[BuoyDetection]] = [[] for _ in merged_reds]

        for white in whites:
            diffs = np.abs(red_depths - white.depth_m)
            nearest_idx = int(np.argmin(diffs))

            # Çok uzak depth farkı varsa bu beyaz o layer'a ait değil
            if diffs[nearest_idx] <= self.max_white_depth_diff_m:
                layer_whites[nearest_idx].append(white)

        # 6. Her layer için sol/sağ beyazı belirle, SlalomLayer oluştur
        layers: List[SlalomLayer] = []
        for idx, (red, whites_in_layer) in enumerate(zip(merged_reds, layer_whites)):
            left, right = self._assign_left_right(red, whites_in_layer)
            layers.append(SlalomLayer(
                layer_idx=idx,
                red=red,
                white_left=left,
                white_right=right,
            ))

        return layers

    # ──────────────────────────────────────────────────────────────────────
    # Yardımcı metodlar
    # ──────────────────────────────────────────────────────────────────────

    def _merge_close_reds(
        self, reds: List[BuoyDetection]
    ) -> List[BuoyDetection]:
        """
        Depth farkı min_red_separation_m'den az olan kırmızıları birleştir.
        (Aynı fiziksel dubanın iki farklı tespiti — daha yüksek confidence'lı tut)
        """
        if not reds:
            return []

        merged: List[BuoyDetection] = [reds[0]]
        for red in reds[1:]:
            prev = merged[-1]
            if abs(red.depth_m - prev.depth_m) < self.min_red_separation_m:
                # Aynı duba — confidence'ı yüksek olanı tut
                if red.confidence > prev.confidence:
                    merged[-1] = red
            else:
                merged.append(red)
        return merged

    def _assign_left_right(
        self,
        red: BuoyDetection,
        whites: List[BuoyDetection],
    ) -> Tuple[Optional[BuoyDetection], Optional[BuoyDetection]]:
        """
        Beyaz dubaları kırmızı dubanın piksel x konumuna göre sol/sağ ata.
        Birden fazla sol/sağ varsa en yüksek confidence'lıyı seç.

        Görüntü koordinatı: x arttıkça sağa gider.
        """
        left_candidates: List[BuoyDetection] = []
        right_candidates: List[BuoyDetection] = []

        for white in whites:
            if white.center_x < red.center_x:
                left_candidates.append(white)
            else:
                right_candidates.append(white)

        best_left = (
            max(left_candidates, key=lambda d: d.confidence)
            if left_candidates else None
        )
        best_right = (
            max(right_candidates, key=lambda d: d.confidence)
            if right_candidates else None
        )
        return best_left, best_right


# ─────────────────────────────────────────────────────────────────────────────
# ROS 2 Node — step 2 çıktısını alır, layer sonuçlarını yayımlar
# ─────────────────────────────────────────────────────────────────────────────

class SlalomLayerDetectorNode(Node):
    """
    Slalom gate layering node'u.

    Subscriptions:
        /slalom/detections_with_depth  (std_msgs/String — JSON)
            Step 2 node'unun çıktısı. Format:
            [
              {
                "class_name": "red_buoy",
                "depth_m": 2.34,
                "center_x": 312.0,
                "center_y": 180.0,
                "confidence": 0.87,
                "mask_points": [[x1,y1], [x2,y2], ...]
              },
              ...
            ]

    Publishes:
        /slalom/layers  (std_msgs/String — JSON)
            Layerlar, Layer 0 = en yakın gate.
    
    Not: Step 2 tamamlandığında bu node'un subscriber topic tipi
         step 2'nin çıktı mesaj tipine göre güncellenecek.
    """

    def __init__(self):
        super().__init__('slalom_layer_detector')

        # ── Parametreler ──────────────────────────────────────────────────
        self.declare_parameter('max_reds', 3)
        self.declare_parameter('min_depth_m', 0.3)
        self.declare_parameter('max_depth_m', 8.0)
        self.declare_parameter('min_red_separation_m', 0.4)
        self.declare_parameter('max_white_depth_diff_m', 1.2)
        self.declare_parameter('min_confidence', 0.35)

        self.detector = SlalomLayerDetector(
            max_reds=self.get_parameter('max_reds').value,
            min_depth_m=self.get_parameter('min_depth_m').value,
            max_depth_m=self.get_parameter('max_depth_m').value,
            min_red_separation_m=self.get_parameter('min_red_separation_m').value,
            max_white_depth_diff_m=self.get_parameter('max_white_depth_diff_m').value,
            min_confidence=self.get_parameter('min_confidence').value,
        )

        # ── QoS ──────────────────────────────────────────────────────────
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        # ── Subscriptions ─────────────────────────────────────────────────
        self.sub_detections = self.create_subscription(
            String,
            '/slalom/detections_with_depth',
            self._detection_callback,
            qos,
        )

        # ── Publishers ────────────────────────────────────────────────────
        self.pub_layers = self.create_publisher(
            String,
            '/slalom/layers',
            10,
        )

        self.get_logger().info('SlalomLayerDetector ready.')

    def _detection_callback(self, msg: String):
        """Gelen JSON tespitlerini parse et, layer'la, yayımla."""
        try:
            raw = json.loads(msg.data)
        except json.JSONDecodeError as e:
            self.get_logger().error(f'JSON parse error: {e}')
            return

        # JSON → BuoyDetection listesi
        detections: List[BuoyDetection] = []
        for item in raw:
            try:
                det = BuoyDetection(
                    class_name=item['class_name'],
                    depth_m=float(item['depth_m']),
                    center_x=float(item['center_x']),
                    center_y=float(item['center_y']),
                    mask_points=np.array(item.get('mask_points', []), dtype=np.float32),
                    confidence=float(item['confidence']),
                )
                detections.append(det)
            except (KeyError, ValueError) as e:
                self.get_logger().warn(f'Malformed detection item: {e}')
                continue

        # Layerlama
        layers = self.detector.detect_layers(detections)

        if not layers:
            self.get_logger().warn('No layers detected.')
            return

        # Log: Layer 0 özeti
        l0 = layers[0]
        self.get_logger().info(
            f'Layer 0 | depth={l0.depth_m:.2f}m '
            f'| left={l0.white_left is not None} '
            f'| right={l0.white_right is not None} '
            f'| complete={l0.is_complete}'
        )

        # Layer sonuçlarını JSON olarak yayımla
        output = self._layers_to_json(layers)
        out_msg = String()
        out_msg.data = json.dumps(output)
        self.pub_layers.publish(out_msg)

    def _layers_to_json(self, layers: List[SlalomLayer]) -> list:
        """SlalomLayer listesini JSON-serializable dict listesine çevirir."""
        result = []
        for layer in layers:
            def buoy_to_dict(b: Optional[BuoyDetection]) -> Optional[dict]:
                if b is None:
                    return None
                return {
                    'class_name': b.class_name,
                    'depth_m': round(b.depth_m, 3),
                    'center_x': round(b.center_x, 1),
                    'center_y': round(b.center_y, 1),
                    'confidence': round(b.confidence, 3),
                    'mask_points': b.mask_points.tolist() if len(b.mask_points) > 0 else [],
                }

            result.append({
                'layer_idx': layer.layer_idx,
                'depth_m': round(layer.depth_m, 3),
                'is_complete': layer.is_complete,
                'red': buoy_to_dict(layer.red),
                'white_left': buoy_to_dict(layer.white_left),
                'white_right': buoy_to_dict(layer.white_right),
            })
        return result


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = SlalomLayerDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
