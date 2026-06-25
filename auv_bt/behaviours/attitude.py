#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import py_trees
from py_trees.common import Status
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Vector3


class AttitudeCheckerCondition(py_trees.behaviour.Behaviour):
    def __init__(self, name="Check Attitude Reached",
                 topic="/current_attitude", tolerance=2.0):
        super().__init__(name)
        self.topic = topic
        self.tolerance = tolerance
        self.node = None
        self.sub = None
        self._latest_yaw = None 

        self.blackboard = py_trees.blackboard.Client(name=name)
        self.blackboard.register_key(
            key="target_yaw_dynamic", access=py_trees.common.Access.READ)
        self.blackboard.register_key(
            key="current_yaw", access=py_trees.common.Access.READ)

    def setup(self, **kwargs):
        self.node = kwargs['node']
    def update(self):
        try:
            current_yaw = self.blackboard.current_yaw
        except KeyError:
            return Status.RUNNING

        try:
            target_yaw = self.blackboard.target_yaw_dynamic
        except KeyError:
            return Status.RUNNING

        diff = (target_yaw - current_yaw + 180.0) % 360.0 - 180.0
        if abs(diff) <= self.tolerance:
            return Status.SUCCESS

        return Status.RUNNING