#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import py_trees
import rclpy
from py_trees.common import Status
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Vector3

class SetAttitudeAction(py_trees.behaviour.Behaviour):
  
    def __init__(self, name="Set Attitude", topic="/target_attitude", target_yaw=0.0, target_roll=0.0, target_pitch=0.0):
        super(SetAttitudeAction, self).__init__(name)
        self.topic = topic
        self.target_yaw = target_yaw  
        self.target_roll = target_roll
        self.target_pitch = target_pitch
        self.node = None
        self.pub = None

    def setup(self, **kwargs):
        self.node = kwargs.get('node')
        # i guess no need to these, just trust to BT
        # if not self.node:
        #     self.node = rclpy.create_node('set_attitude_actions')
        self.pub = self.node.create_publisher(Vector3, self.topic, 10)

    def update(self):
       
        msg = Vector3()
        msg.x = self.target_roll
        msg.y = self.target_pitch
        msg.z = self.target_yaw
        self.pub.publish(msg)            
        return Status.RUNNING