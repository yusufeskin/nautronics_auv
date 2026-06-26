#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import py_trees
import rclpy.qos
from geometry_msgs.msg import Vector3
from py_trees_ros import subscribers

class AttitudeToBlackboard(subscribers.ToBlackboard):
    """
    /current_attitude topiğinden Vector3 mesajını alır ve 
    z eksenini (yaw) blackboard üzerindeki 'current_yaw' değişkenine yazar.
    """
    def __init__(self,
                 name: str,
                 topic_name: str,
                 qos_profile: rclpy.qos.QoSProfile):
        super().__init__(name=name,
                         topic_name=topic_name,
                         topic_type=Vector3,
                         qos_profile=qos_profile,
                         blackboard_variables={"attitude_msg": None},
                         clearing_policy=py_trees.common.ClearingPolicy.NEVER
                         )
        # current_yaw key'ini deklare et
        self.blackboard.register_key(
            key="current_yaw",
            access=py_trees.common.Access.WRITE
        )

    def update(self) -> py_trees.common.Status:
        status = super(AttitudeToBlackboard, self).update()
        
        # Eğer mesaj gelmişse (RUNNING değilse), yaw değerini blackboard'a yaz
        if status != py_trees.common.Status.RUNNING:
            if hasattr(self.blackboard, 'attitude_msg') and self.blackboard.attitude_msg is not None:
                self.blackboard.current_yaw = self.blackboard.attitude_msg.z
                
        return status
