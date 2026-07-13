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
            key="is_gate_found",
            access=py_trees.common.Access.WRITE
        )
        self.blackboard.register_key(
            key="is_realtorpedo_found",
            access=py_trees.common.Access.WRITE
        )
        self.blackboard.is_gate_found = False
        self.blackboard.is_realtorpedo_found = False
        self.blackboard.yolo_detections = DetectionArray()
        self.blackboard.yolo_detections.detections = []
        
        # Sliding window history for filtering false positives
        self.gate_history = []
        self.realtorpedo_history = []
        self.history_len = 5
        self.min_detects = 3

    def update(self) -> py_trees.common.Status:
        self.logger.debug("%s.update()" % self.__class__.__name__)
        status = super(ToBlackboard, self).update()
        if status != py_trees.common.Status.RUNNING: 
            detected_names = [obj.class_name for obj in self.blackboard.yolo_detections.detections]
            
            gate_detected = "gate" in detected_names
            self.gate_history.append(gate_detected)
            if len(self.gate_history) > self.history_len:
                self.gate_history.pop(0)
                
            realtorpedo_detected = "realtorpedo" in detected_names
            self.realtorpedo_history.append(realtorpedo_detected)
            if len(self.realtorpedo_history) > self.history_len:
                self.realtorpedo_history.pop(0)

            self.blackboard.is_gate_found = sum(self.gate_history) >= self.min_detects
            self.blackboard.is_realtorpedo_found = sum(self.realtorpedo_history) >= self.min_detects

        return status