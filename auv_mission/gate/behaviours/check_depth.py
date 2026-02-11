#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import py_trees
import rclpy
from nav_msgs.msg import Odometry
from py_trees.common import Status
from rclpy.qos import qos_profile_sensor_data 

class CheckDepth(py_trees.behaviour.Behaviour):
    def __init__(self, name="Check Depth Range", topic_name="/odom"):
        super(CheckDepth, self).__init__(name)
        self.topic_name = topic_name
        self.current_z = None
        self.node = None
        self.sub = None

    def setup(self, **kwargs):
        try:
            self.node = kwargs['node']
        except KeyError:
            self.node = rclpy.create_node('check_depth_temp')

        self.sub = self.node.create_subscription(
            Odometry,
            self.topic_name,
            self.odom_callback,
            qos_profile=qos_profile_sensor_data 
        )

    def odom_callback(self, msg):
        self.current_z = msg.pose.pose.position.z

    def update(self):
        if self.current_z is None:
            self.feedback_message = "Waiting for data..."
            return Status.RUNNING

        if -1.7 <= self.current_z <= -1.3:
            self.feedback_message = f"OK! In range: {self.current_z:.3f}"
            return Status.SUCCESS
        else:
            self.feedback_message = f"Out of range: {self.current_z:.3f}"
            return Status.FAILURE