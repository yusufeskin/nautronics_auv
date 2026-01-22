#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# License: BSD
#   https://raw.github.com/splintered-reality/py_trees_ros/license/LICENSE
#

import py_trees
import rclpy.qos
import auv_interfaces.msg

from py_trees_ros import subscribers
class ToBlackboard(subscribers.ToBlackboard):
    def __init__(self,
                 name: str,
                 topic_name: str,
                 qos_profile: rclpy.qos.QoSProfile):
        super().__init__(name=name,
                         topic_name=topic_name,
                         topic_type=auv_interfaces.msg.VehicleStatus,
                         qos_profile=qos_profile,
                         blackboard_variables={"state": None},
                         clearing_policy=py_trees.common.ClearingPolicy.NEVER
                         )
        self.blackboard.register_key(
            key="vehicle_mode",
            access=py_trees.common.Access.WRITE
        )

        self.blackboard.register_key(
            key="is_guided",
            access=py_trees.common.Access.WRITE
        )
        self.blackboard.vehicle_mode = "UNKNOWN"
        self.blackboard.is_guided = False
        self.blackboard.state = auv_interfaces.msg.VehicleStatus()
        self.blackboard.state.header.frame_id = "base_link"
        self.blackboard.state.mode = "UNKNOWN"
        self.blackboard.state.is_armed = False
        self.blackboard.state.is_connected = False
        

    def update(self) -> py_trees.common.Status:
        """
        Call the parent to write the raw data to the blackboard and then check against the
        threshold to determine if the low warning flag should also be updated.

        Returns:
            :attr:`~py_trees.common.Status.SUCCESS` if a message was written, :attr:`~py_trees.common.Status.RUNNING` otherwise.
        """
        self.logger.debug("%s.update()" % self.__class__.__name__)
        status = super(ToBlackboard, self).update()
        if status != py_trees.common.Status.RUNNING:
            current_msg = self.blackboard.state
            self.blackboard.vehicle_mode = current_msg.mode
            if current_msg.mode == "GUIDED":
                self.blackboard.is_guided = True
            else:
                self.blackboard.is_guided = False
        return status
