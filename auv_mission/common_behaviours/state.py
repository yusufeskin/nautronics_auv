#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import py_trees
import rclpy
from std_msgs.msg import String
from py_trees.common import Status

class ToBlackboard(py_trees.behaviour.Behaviour):
    """
    ROS topic üzerinden gelen Mod bilgisini (String) okur
    ve Blackboard'daki 'vehicle_mode' değişkenine yazar.
    """
    def __init__(self, name, topic_name="/vehicle/state_mode"):
        super(ToBlackboard, self).__init__(name=name)
        self.topic_name = topic_name
        
        # Blackboard erişimi (Yazma izni)
        self.blackboard = py_trees.blackboard.Client(name=name, namespace=None)
        self.blackboard.register_key(key="vehicle_mode", access=py_trees.common.Access.WRITE)
        
        self.node = None
        self.sub = None
        self.current_mode = "UNKNOWN"

    def setup(self, **kwargs):
        """
        Behavior Tree kurulurken Node buraya gelir.
        """
        try:
            self.node = kwargs['node']
        except KeyError:
            self.node = rclpy.create_node('state_to_bb_temp')

        # String mesajını dinliyoruz
        self.sub = self.node.create_subscription(
            String,
            self.topic_name,
            self.callback,
            10  # Reliable QoS
        )

    def callback(self, msg):
        """
        Mesaj geldiğinde çalışır.
        """
        self.current_mode = msg.data
        # Blackboard'a yaz (Önemli kısım burası)
        self.blackboard.vehicle_mode = self.current_mode
        
        # Ağaçta görünmesi için feedback mesajı
        self.feedback_message = f"Mode: {self.current_mode}"

    def update(self):
        """
        Her Tick'te çalışır.
        """
        # Veri gelmese bile SUCCESS dönelim ki akış durmasın.
        # Sadece henüz veri yoksa feedback verelim.
        if self.current_mode == "UNKNOWN":
            self.feedback_message = "Veri bekleniyor..."
        
        return Status.SUCCESS