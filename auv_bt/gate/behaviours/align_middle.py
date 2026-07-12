#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import py_trees
import rclpy
from geometry_msgs.msg import Twist
from auv_interfaces.msg import DetectionArray

class CenterTarget(py_trees.behaviour.Behaviour):
    """
    Bu düğüm (node), YOLO'dan gelen hedefin (örneğin 'compass' veya 'wrench')
    kamera kadrajında ortalanmasını sağlar. 
    Hedef ortalandığında aracın sürüklenip dönmesini engellemek için "settle" (oturma)
    süresi eklenmiştir. Bu sürede araç hızını sıfırlar ve durmasını bekler.
    """
    def __init__(self, 
                 name: str, 
                 target_class: str, 
                 topic_cmd: str = "/cmd_vel",
                 image_width: int = 640,
                 image_height: int = 480,
                 error_tol_x: float = 40.0,  # X ekseninde kabul edilebilir piksel hatası (çok küçük değil)
                 error_tol_y: float = 40.0,  # Y ekseninde kabul edilebilir piksel hatası
                 settle_time: float = 2.0    # Başarılı sayılmadan önce merkezde bekleme süresi (sürüklenmeyi önlemek için)
                 ):
        super(CenterTarget, self).__init__(name)
        self.target_class = target_class
        self.topic_cmd = topic_cmd
        
        # Kadraj merkezi
        self.center_x = image_width / 2.0
        self.center_y = image_height / 2.0
        
        self.error_tol_x = error_tol_x
        self.error_tol_y = error_tol_y
        
        # P kontrolcü katsayıları (aracın tepkisine göre ayarlanmalıdır)
        self.kp_yaw = 0.002
        self.kp_heave = 0.002
        
        self.settle_time = settle_time
        self.time_centered = None
        
        self.node = None
        self.cmd_pub = None

    def setup(self, **kwargs):
        try:
            self.node = kwargs['node']
        except KeyError:
            pass
        
        if self.node is None:
            self.node = rclpy.create_node(f"center_{self.target_class}_node")
            
        self.cmd_pub = self.node.create_publisher(Twist, self.topic_cmd, 10)
        self.logger.info(f"[{self.name}] Setup tamamlandı. Hedef: {self.target_class}")

    def initialise(self):
        self.logger.info(f"[{self.name}] Başlatıldı. {self.target_class} ortalanıyor...")
        self.time_centered = None

    def update(self):
        # Blackboard üzerinden YOLO verisini çek (object2bb düğümü ile blackboard'a yazıldığı varsayılmıştır)
        detections_msg = self.blackboard.get("yolo_detections")
        
        cmd = Twist()
        
        if detections_msg is None or not detections_msg.detections:
            self.logger.warning(f"[{self.name}] Hedef bulunamadı, bekleniyor...")
            self.cmd_pub.publish(cmd) # Dur
            self.time_centered = None
            return py_trees.common.Status.RUNNING

        # Hedef sınıfını bul
        target = None
        for det in detections_msg.detections:
            if det.class_name == self.target_class:
                target = det
                break
                
        if target is None:
            self.logger.warning(f"[{self.name}] '{self.target_class}' kadrajda yok, bekleniyor...")
            self.cmd_pub.publish(cmd) # Dur
            self.time_centered = None
            return py_trees.common.Status.RUNNING

        # Hata hesaplama (Piksel cinsinden)
        err_x = target.bbox_center_x - self.center_x
        err_y = target.bbox_center_y - self.center_y

        self.logger.debug(f"[{self.name}] Hata X: {err_x:.2f}, Y: {err_y:.2f}")

        # Hata tolerans içinde mi kontrol et (Sadece X ekseni)
        if abs(err_x) < self.error_tol_x:
            # Hedef ortalandı. Hemen başarılı demek yerine aracın durulmasını (settle) bekle
            if self.time_centered is None:
                self.time_centered = self.node.get_clock().now()
                self.logger.info(f"[{self.name}] Hedef ortalandı, sürüklenmeyi önlemek için durulması bekleniyor...")
            
            # Sürüklenmeyi önlemek için aktif olarak sıfır hız gönder
            self.cmd_pub.publish(cmd) # Bütün hızlar 0
            
            elapsed_time = (self.node.get_clock().now() - self.time_centered).nanoseconds / 1e9
            if elapsed_time >= self.settle_time:
                self.logger.info(f"[{self.name}] Araç stabil, ortalama başarılı!")
                return py_trees.common.Status.SUCCESS
            else:
                return py_trees.common.Status.RUNNING

        else:
            # Hedef ortada değil, P kontrolcü ile hızları hesapla
            self.time_centered = None
            
            # Dönüş (Yaw) hesabı
            # Eğer hedef sağdaysa (err_x pozitif), aracın sağa dönmesi (negatif yaw) gerekir.
            yaw_cmd = -self.kp_yaw * err_x
            
            # Limitler (Çok hızlı dönmesini engellemek için)
            max_yaw = 0.3
            yaw_cmd = max(-max_yaw, min(max_yaw, yaw_cmd))

            cmd.angular.z = float(yaw_cmd)
            # Heave düzeltmesi iptal edildi, linear.z varsayılan olarak 0 kalıyor.

            self.cmd_pub.publish(cmd)
            return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        self.logger.info(f"[{self.name}] Sonlandırıldı. Durum: {new_status}")
        # Görev bittiğinde veya iptal edildiğinde aracı kesin olarak durdur
        cmd = Twist()
        if self.cmd_pub is not None:
            self.cmd_pub.publish(cmd)
