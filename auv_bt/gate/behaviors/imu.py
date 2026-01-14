#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# License: BSD
#   https://raw.github.com/splintered-reality/py_trees_ros/license/LICENSE
#
##############################################################################
# Documentation
##############################################################################

"""
Getting the most out of your battery.
"""

##############################################################################
# Imports
##############################################################################

import py_trees
import rclpy.qos
import sensor_msgs.msg

from py_trees_ros import subscribers

##############################################################################
# Behaviours
##############################################################################


class ToBlackboard(subscribers.ToBlackboard):
    """
    Subscribes to the battery message and writes battery data to the blackboard.
    Also adds a warning flag to the blackboard if the battery
    is low - note that it does some buffering against ping-pong problems so the warning
    doesn't trigger on/off rapidly when close to the threshold.

    When ticking, updates with :attr:`~py_trees.common.Status.RUNNING` if it got no data,
    :attr:`~py_trees.common.Status.SUCCESS` otherwise.

    Blackboard Variables:
        * battery (:class:`sensor_msgs.msg.BatteryState`)[w]: the raw battery message
        * battery_low_warning (:obj:`bool`)[w]: False if battery is ok, True if critically low

    Args:
        name: name of the behaviour
        topic_name: name of the battery state topic
        qos_profile: qos profile for the subscriber
        threshold: percentage level threshold for flagging as low (0-100)
    """
    def __init__(self,
                 name: str,
                 topic_name: str,
                 qos_profile: rclpy.qos.QoSProfile):
        super().__init__(name=name,
                         topic_name=topic_name,
                         topic_type=sensor_msgs.msg.Imu,
                         qos_profile=qos_profile,
                         blackboard_variables={"imu": None},
                         clearing_policy=py_trees.common.ClearingPolicy.NEVER
                         )
        self.blackboard.register_key(
            key="is_upside_down",
            access=py_trees.common.Access.WRITE
        )
        self.blackboard.is_upside_down = False
        self.blackboard.imu = sensor_msgs.msg.Imu()
        self.blackboard.imu.orientation.x = 0.0
        self.blackboard.imu.orientation.y = 0.0
        self.blackboard.imu.orientation.z = 0.0
        self.blackboard.imu.orientation.w = 1.0
        self.blackboard.imu.orientation_covariance[0] = -1.0
        self.blackboard.imu.angular_velocity_covariance[0] = -1.0
        self.blackboard.imu.linear_acceleration_covariance[0] = -1.0
        self.blackboard.imu.header.frame_id = "imu_link"

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
            # we got something
            current_roll = abs(self.blackboard.imu.orientation.x) 
            
            if current_roll > 0.7:
                self.blackboard.is_upside_down = True
                self.feedback_message = "ALARM!"
            else:
                self.blackboard.is_upside_down = False
                self.feedback_message = "Durum Stabil"

        return status
