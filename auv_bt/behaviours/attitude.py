#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import py_trees
from py_trees.common import Status


class AttitudeCheckerCondition(py_trees.behaviour.Behaviour):
    def __init__(self, name="Check Attitude Reached", tolerance=2.0):
        super(AttitudeCheckerCondition, self).__init__(name)
        self.tolerance = tolerance

        self.blackboard = py_trees.blackboard.Client(name=name)
        self.blackboard.register_key(key="target_yaw_dynamic", access=py_trees.common.Access.READ)
        self.blackboard.register_key(key="current_yaw", access=py_trees.common.Access.READ)

    def update(self):
        try:
            current_yaw = self.blackboard.current_yaw
            target_yaw = self.blackboard.target_yaw_dynamic
        except KeyError:
            return Status.RUNNING

        diff = (target_yaw - current_yaw + 180.0) % 360.0 - 180.0
        if abs(diff) <= self.tolerance:
            return Status.SUCCESS
        else:
            return Status.RUNNING