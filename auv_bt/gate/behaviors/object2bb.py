#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import py_trees
import rclpy.qos
from auv_interfaces.msg import DetectionArray
from py_trees_ros import subscribers
class ToBlackboard(subscribers.ToBlackboard):
    def __init__(self,
                 name: str,
                 topic_name: str,
                 qos_profile: rclpy.qos.QoSProfile):
        super().__init__(name=name,
                         topic_name=topic_name,
                         topic_type=DetectionArray,
                         qos_profile=qos_profile,
                         blackboard_variables={"yolo_detections": None},
                         clearing_policy=py_trees.common.ClearingPolicy.NEVER
                         )
        self.blackboard.register_key(
            key="is_gate_founded",
            access=py_trees.common.Access.WRITE
        )
        self.blackboard.register_key(
            key="is_torpedo_founded",
            access=py_trees.common.Access.WRITE
        )
        self.blackboard.is_gate_founded = False
        self.blackboard.is_torpedo_founded = False
        self.blackboard.yolo_detections = DetectionArray()
        self.blackboard.yolo_detections.detections = []
        

    def update(self) -> py_trees.common.Status:
        self.logger.debug("%s.update()" % self.__class__.__name__)
        status = super(ToBlackboard, self).update()
        if status == py_trees.common.Status.SUCCESS:
            detected_names = [obj.class_name for obj in self.blackboard.yolo_detections.detections]
            self.blackboard.is_gate_founded = "gate" in detected_names
            self.blackboard.is_torpedo_founded = "torpedo" in detected_names

        return status
