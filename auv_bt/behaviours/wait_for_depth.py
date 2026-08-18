#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import py_trees
import rclpy.qos
from geometry_msgs.msg import Point


class WaitForDepth(py_trees.behaviour.Behaviour):

    def __init__(self, name, target_depth, tolerance=0.2):
        super().__init__(name=name)
        self.target_depth = target_depth
        self.tolerance = tolerance
        self.current_depth = None
        self._sub = None

    def setup(self, **kwargs):
        try:
            self.node = kwargs['node']
        except KeyError as e:
            raise KeyError("WaitForDepth: 'node' bulunamadı") from e

        self._sub = self.node.create_subscription(
            Point,
            'vehicle/local_position',
            self._position_callback,
            rclpy.qos.qos_profile_sensor_data
        )
        self.logger.info(
            f"WaitForDepth: Hedef derinlik={self.target_depth:.2f}m, Tolerans={self.tolerance:.2f}m"
        )

    def _position_callback(self, msg):
        self.current_depth = msg.z

    def update(self):
        if self.current_depth is None:
            self.feedback_message = "Derinlik verisi henuz gelmedi, bekleniyor..."
            return py_trees.common.Status.RUNNING

        error = abs(self.current_depth - self.target_depth)
        self.feedback_message = f"Derinlik farki: {error:.2f}m (Tolerans: {self.tolerance:.2f}m)"

        if error <= self.tolerance:
            self.logger.info(f"HEDEF DERINLIGE ULASILDI! Derinlik: {self.current_depth:.2f}m")
            return py_trees.common.Status.SUCCESS

        return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        if self._sub is not None:
            self.node.destroy_subscription(self._sub)
            self._sub = None
