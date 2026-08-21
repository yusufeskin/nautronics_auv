#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import py_trees
from py_trees.common import Status
from std_msgs.msg import Float64
from auv_interfaces.msg import PositionYawTarget
import behaviours.attitude


class SetSpeedAction(py_trees.behaviour.Behaviour):
    def __init__(self, name, speed_mps, topic="/target_speed"):
        super().__init__(name)
        self.speed_mps = speed_mps
        self.topic = topic
        self.node = None
        self.pub = None

    def setup(self, **kwargs):
        self.node = kwargs.get('node')
        self.pub = self.node.create_publisher(Float64, self.topic, 10)

    def update(self):
        msg = Float64()
        msg.data = self.speed_mps
        self.pub.publish(msg)
        return Status.SUCCESS


class SetForwardTargetAction(py_trees.behaviour.Behaviour):
    def __init__(self, name, distance, depth, topic="/target_position_yaw"):
        super().__init__(name)
        self.distance = distance
        self.depth = depth
        self.topic = topic
        self.node = None
        self.pub = None
        self.msg = None
        self.target_calculated = False

        self.blackboard = py_trees.blackboard.Client(name=name)
        self.blackboard.register_key(key="current_x", access=py_trees.common.Access.READ)
        self.blackboard.register_key(key="current_y", access=py_trees.common.Access.READ)
        self.blackboard.register_key(key="current_yaw", access=py_trees.common.Access.READ)
        self.blackboard.register_key(key="target_x_dynamic", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="target_y_dynamic", access=py_trees.common.Access.WRITE)

    def setup(self, **kwargs):
        self.node = kwargs.get('node')
        self.pub = self.node.create_publisher(PositionYawTarget, self.topic, 10)

    def initialise(self):
        self.target_calculated = False
        self.msg = None

        try:
            current_x = self.blackboard.current_x
            current_y = self.blackboard.current_y
            current_yaw = self.blackboard.current_yaw
        except KeyError:
            return

        yaw_rad = math.radians(current_yaw)
        target_x = current_x + self.distance * math.cos(yaw_rad)
        target_y = current_y + self.distance * math.sin(yaw_rad)

        self.blackboard.target_x_dynamic = target_x
        self.blackboard.target_y_dynamic = target_y

        self.msg = PositionYawTarget()
        self.msg.x = target_x
        self.msg.y = target_y
        self.msg.z = self.depth
        self.msg.yaw_deg = 0.0
        self.msg.use_yaw = False

        self.target_calculated = True

    def update(self):
        if not self.target_calculated:
            return Status.RUNNING

        if self.pub and self.msg:
            self.pub.publish(self.msg)

        return Status.SUCCESS


class SetTurnInPlaceAction(py_trees.behaviour.Behaviour):
    def __init__(self, name, yaw_increment, depth, topic="/target_position_yaw"):
        super().__init__(name)
        self.yaw_increment = yaw_increment
        self.depth = depth
        self.topic = topic
        self.node = None
        self.pub = None
        self.msg = None
        self.target_calculated = False

        self.blackboard = py_trees.blackboard.Client(name=name)
        self.blackboard.register_key(key="current_x", access=py_trees.common.Access.READ)
        self.blackboard.register_key(key="current_y", access=py_trees.common.Access.READ)
        self.blackboard.register_key(key="current_yaw", access=py_trees.common.Access.READ)
        self.blackboard.register_key(key="target_yaw_dynamic", access=py_trees.common.Access.WRITE)

    def setup(self, **kwargs):
        self.node = kwargs.get('node')
        self.pub = self.node.create_publisher(PositionYawTarget, self.topic, 10)

    def initialise(self):
        self.target_calculated = False
        self.msg = None

        try:
            current_x = self.blackboard.current_x
            current_y = self.blackboard.current_y
            current_yaw = self.blackboard.current_yaw
        except KeyError:
            return

        raw_target = current_yaw + self.yaw_increment
        target_yaw = (raw_target + 180.0) % 360.0 - 180.0
        self.blackboard.target_yaw_dynamic = target_yaw

        self.msg = PositionYawTarget()
        self.msg.x = current_x
        self.msg.y = current_y
        self.msg.z = self.depth
        self.msg.yaw_deg = target_yaw
        self.msg.use_yaw = True

        self.target_calculated = True

    def update(self):
        if not self.target_calculated:
            return Status.RUNNING

        if self.pub and self.msg:
            self.pub.publish(self.msg)

        return Status.SUCCESS


class WaitForArrivalDynamic(py_trees.behaviour.Behaviour):
    def __init__(self, name, tolerance=0.4):
        super().__init__(name)
        self.tolerance = tolerance

        self.blackboard = py_trees.blackboard.Client(name=name)
        self.blackboard.register_key(key="current_x", access=py_trees.common.Access.READ)
        self.blackboard.register_key(key="current_y", access=py_trees.common.Access.READ)
        self.blackboard.register_key(key="target_x_dynamic", access=py_trees.common.Access.READ)
        self.blackboard.register_key(key="target_y_dynamic", access=py_trees.common.Access.READ)

    def update(self):
        try:
            current_x = self.blackboard.current_x
            current_y = self.blackboard.current_y
            target_x = self.blackboard.target_x_dynamic
            target_y = self.blackboard.target_y_dynamic
        except KeyError:
            self.feedback_message = "Konum/hedef verisi henuz gelmedi, bekleniyor..."
            return Status.RUNNING

        distance = math.sqrt((current_x - target_x) ** 2 + (current_y - target_y) ** 2)
        self.feedback_message = f"Hedefe mesafe: {distance:.2f}m (Tolerans: {self.tolerance:.2f}m)"

        if distance <= self.tolerance:
            return Status.SUCCESS

        return Status.RUNNING


def create_forward_step(distance: float, depth: float, step_name: str) -> py_trees.composites.Sequence:
    set_target = SetForwardTargetAction(
        name=f"Send {distance:.1f}m Forward",
        distance=distance,
        depth=depth
    )
    checker = WaitForArrivalDynamic(
        name=f"Check {distance:.1f}m Forward",
        tolerance=0.4
    )
    return py_trees.composites.Sequence(name=step_name, memory=True, children=[set_target, checker])


def create_turn_step(yaw_increment: float, depth: float, step_name: str) -> py_trees.composites.Sequence:
    set_target = SetTurnInPlaceAction(
        name=f"Turn {yaw_increment:+.0f}°",
        yaw_increment=yaw_increment,
        depth=depth
    )
    checker = behaviours.attitude.AttitudeCheckerCondition(
        name=f"Check Turn {yaw_increment:+.0f}°",
        tolerance=3.0
    )
    return py_trees.composites.Sequence(name=step_name, memory=True, children=[set_target, checker])
