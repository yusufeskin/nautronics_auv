#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import py_trees
import rclpy.qos
from geometry_msgs.msg import Point
from py_trees_ros import subscribers


class LocalPositionToBlackboard(subscribers.ToBlackboard):
    def __init__(self,
                 name: str,
                 topic_name: str,
                 qos_profile: rclpy.qos.QoSProfile):
        super().__init__(name=name,
                          topic_name=topic_name,
                          topic_type=Point,
                          qos_profile=qos_profile,
                          blackboard_variables={"local_position_msg": None},
                          clearing_policy=py_trees.common.ClearingPolicy.NEVER
                          )
        self.blackboard.register_key(key="current_x", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="current_y", access=py_trees.common.Access.WRITE)

    def update(self) -> py_trees.common.Status:
        status = super(LocalPositionToBlackboard, self).update()

        if status != py_trees.common.Status.RUNNING:
            if hasattr(self.blackboard, 'local_position_msg') and self.blackboard.local_position_msg is not None:
                self.blackboard.current_x = self.blackboard.local_position_msg.x
                self.blackboard.current_y = self.blackboard.local_position_msg.y

        return status
