#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
import py_trees
from std_msgs.msg import Empty

class TorpedoIgniter(py_trees.behaviour.Behaviour):
    def __init__(self, name="Fire Torpedo", topic_name="/torpedo/fire"):
        super(TorpedoIgniter, self).__init__(name)
        self.topic_name = topic_name
        self.node = None
        self.pub = None

    def setup(self, **kwargs):
        self.node = kwargs.get('node')
        if not self.node:
            self.node = rclpy.create_node('torpedo_igniter_action')
        self.pub = self.node.create_publisher(Empty, self.topic_name, 10)

    def update(self):
        if self.pub:
            msg = Empty()
            self.pub.publish(msg)
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.FAILURE
