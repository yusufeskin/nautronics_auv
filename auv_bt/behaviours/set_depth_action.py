#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import py_trees
import rclpy
from std_msgs.msg import Float64
from py_trees.common import Status
from rclpy.qos import qos_profile_sensor_data

class SetDepthAction(py_trees.behaviour.Behaviour):
  
    def __init__(self, name="Set Target Depth", topic="/target_depth", target_depth=-1.5):
        super(SetDepthAction, self).__init__(name)
        self.topic = topic
        self.target_depth = target_depth  
        
        self.node = None
        self.pub = None

    def setup(self, **kwargs):
        self.node = kwargs.get('node')
        # i guess no need to these, just trust to BT
        # if not self.node:
        #     self.node = rclpy.create_node('set_depth_action')

        self.pub = self.node.create_publisher(Float64, self.topic, 10)


    def update(self):
       
        msg = Float64()
        msg.data = self.target_depth
        self.pub.publish(msg)            
        return Status.RUNNING