############### This file will be used for attitude check, e.g.  vehicle takes 90 degree yaw turn command,
# file looks if turning completed or not, anyway i will handle this file as well

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import py_trees
import rclpy
from py_trees.common import Status
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Vector3

class AttitudeCheckerCondition(py_trees.behaviour.Behaviour):
    def __init__(self, name="Check Attitude Reached", topic="/current_attitude", target_yaw=0.0, tolerance=5.0):
        super(AttitudeCheckerCondition, self).__init__(name)
        self.topic = topic
        self.target_yaw = target_yaw
        self.tolerance = tolerance
        
        self.node = None
        self.sub = None
        self.current_yaw = None

    def setup(self, **kwargs):
        self.node = kwargs.get('node')
        self.sub = self.node.create_subscription(
            Vector3, self.topic, self.callback, qos_profile=qos_profile_sensor_data
        )

    def callback(self, msg):
        self.current_yaw = msg.z

    def update(self):
        if self.current_yaw is None:
            return Status.RUNNING

        diff = (self.target_yaw - self.current_yaw + 180) % 360 - 180
        
        if abs(diff) <= self.tolerance:
            return Status.SUCCESS
        else:
            return Status.RUNNING