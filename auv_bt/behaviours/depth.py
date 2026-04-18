#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import py_trees
import rclpy
from std_msgs.msg import Float64
from rclpy.qos import qos_profile_sensor_data
from py_trees.common import Status

class ToBlackboard(py_trees.behaviour.Behaviour):
   
    def __init__(self, name, topic_name="/baro_data", qos_profile=qos_profile_sensor_data):
        super(ToBlackboard, self).__init__(name=name)
        self.topic_name = topic_name
        self.qos_profile = qos_profile
        
        self.blackboard = py_trees.blackboard.Client(name=name, namespace=None)
        self.blackboard.register_key(key="depth", access=py_trees.common.Access.WRITE)
        
        self.node = None
        self.sub = None

    def setup(self, **kwargs):
        try:
            self.node = kwargs['node']
        except KeyError:
            self.node = rclpy.create_node('depth_to_bb_temp')

        self.sub = self.node.create_subscription(
            Float64,
            self.topic_name,
            self.callback,
            qos_profile=self.qos_profile
        )

    def callback(self, msg):
        current_depth = msg.data
        self.blackboard.depth = current_depth
        
        self.feedback_message = f"Depth: {current_depth:.3f}m"

    def update(self):
        if not hasattr(self.blackboard, "depth") or self.blackboard.depth is None:
            self.feedback_message = "Waiting for data..."
            return Status.RUNNING
        
        return Status.SUCCESS
    

# import py_trees_ros
# from std_msgs.msg import Float64

#
# baro_to_bb = py_trees_ros.subscribers.ToBlackboard(
#     name="Baro_To_Blackboard",
#     topic_name="/baro_data",
#     topic_type=Float64,
#     qos_profile=py_trees_ros.utilities.qos_profile_sensor_data(),
#     blackboard_variables={"depth": "data"}
# )