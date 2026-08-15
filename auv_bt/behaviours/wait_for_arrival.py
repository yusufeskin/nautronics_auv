#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
import py_trees
import rclpy.qos
from geometry_msgs.msg import Point


class WaitForArrival(py_trees.behaviour.Behaviour):

    def __init__(self, name, target_x, target_y, tolerance=1.5):
        super().__init__(name=name)
        self.target_x = target_x
        self.target_y = target_y
        self.tolerance = tolerance
        self.current_x = None
        self.current_y = None
        self._sub = None

    def setup(self, **kwargs):
        try:
            self.node = kwargs['node']
        except KeyError as e:
            raise KeyError("WaitForArrival: 'node' bulunamadı") from e

        self._sub = self.node.create_subscription(
            Point,
            'vehicle/local_position',
            self._position_callback,
            rclpy.qos.qos_profile_sensor_data
        )
        self.logger.info(f"WaitForArrival: Hedef X={self.target_x:.2f}, Y={self.target_y:.2f}, Tolerans={self.tolerance:.2f}m")

    def _position_callback(self, msg):
        self.current_x = msg.x
        self.current_y = msg.y

    def update(self):
        if self.current_x is None or self.current_y is None:
            self.feedback_message = "Konum verisi henüz gelmedi, bekleniyor..."
            return py_trees.common.Status.RUNNING

        distance = math.sqrt(
            (self.current_x - self.target_x) ** 2 +
            (self.current_y - self.target_y) ** 2
        )

        self.feedback_message = f"Hedefe mesafe: {distance:.2f}m (Tolerans: {self.tolerance:.2f}m)"

        if distance <= self.tolerance:
            self.logger.info(f"HEDEFE ULASILDI! Mesafe: {distance:.2f}m")
            return py_trees.common.Status.SUCCESS
        else:
            return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        if self._sub is not None:
            self.node.destroy_subscription(self._sub)
            self._sub = None
