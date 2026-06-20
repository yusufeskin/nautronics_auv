#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import py_trees
from py_trees.common import Status
from geometry_msgs.msg import Vector3

class SetAttitudeAction(py_trees.behaviour.Behaviour):
    def __init__(self, name="Set Attitude", topic="/target_attitude", 
                 yaw_increment=15.0, target_roll=0.0, target_pitch=0.0):
        super(SetAttitudeAction, self).__init__(name)
        self.topic = topic
        self.yaw_increment = yaw_increment  
        self.target_roll = target_roll
        self.target_pitch = target_pitch
        self.pub = None
        self.target_calculated = False
        self.msg = None
        self.blackboard = py_trees.blackboard.Client(name=name)
        self.blackboard.register_key(key="current_yaw", access=py_trees.common.Access.READ)
        self.blackboard.register_key(key="target_yaw_dynamic", access=py_trees.common.Access.WRITE)

    def setup(self, **kwargs):
        self.node = kwargs.get('node')
        self.pub = self.node.create_publisher(Vector3, self.topic, 10)

    def initialise(self):
        self.target_calculated = False
        self.msg = None

        try:
            current_yaw = self.blackboard.current_yaw
        except KeyError:
            return 
        raw_target = current_yaw + self.yaw_increment
        normalized_yaw = (raw_target + 180.0) % 360.0 - 180.0

        self.blackboard.target_yaw_dynamic = normalized_yaw

        self.msg = Vector3()
        self.msg.x = self.target_roll
        self.msg.y = self.target_pitch
        self.msg.z = normalized_yaw

        self.target_calculated = True 

    def update(self):
        if not self.target_calculated:
            return Status.RUNNING

        if self.pub and self.msg:
            self.pub.publish(self.msg)

        return Status.RUNNING