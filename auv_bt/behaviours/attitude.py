#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import py_trees
import rclpy
from py_trees.common import Status
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Vector3

class AttitudeCheckerCondition(py_trees.behaviour.Behaviour):
    def __init__(self, name="Check Attitude Reached", topic="/current_attitude", tolerance=2.0):
        super(AttitudeCheckerCondition, self).__init__(name)
        self.topic = topic
        self.tolerance = tolerance
        self.node = None
        self.sub = None
        self.current_yaw = None

        self.blackboard = py_trees.blackboard.Client(name=name) 
        self.blackboard.register_key(key="target_yaw_dynamic", access=py_trees.common.Access.READ)
        self.blackboard.register_key(key="current_yaw", access=py_trees.common.Access.WRITE)

    def setup(self, **kwargs):
        self.node = kwargs.get('node')
        self.sub = self.node.create_subscription(
            Vector3, self.topic, self.callback, qos_profile=qos_profile_sensor_data
        )

    def callback(self, msg):
        self.current_yaw = msg.z
        self.blackboard.current_yaw = msg.z

    def update(self):
        if self.current_yaw is None:
            return Status.RUNNING

        try:
            target_yaw = self.blackboard.target_yaw_dynamic
        except KeyError:
            return Status.RUNNING

        diff = (target_yaw - self.current_yaw + 180.0) % 360.0 - 180.0
        if abs(diff) <= self.tolerance:
            return Status.SUCCESS
        else:
            return Status.RUNNING