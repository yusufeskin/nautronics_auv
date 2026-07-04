#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import py_trees
import rclpy
from std_msgs.msg import Float64  # DEĞİŞİKLİK 1: Odometry yerine Float32 eklendi
from geometry_msgs.msg import Twist
from py_trees.common import Status
from rclpy.qos import qos_profile_sensor_data

class ArrangeDepthAction(py_trees.behaviour.Behaviour):
  
    def __init__(self, name="Smart Depth Adjustment", topic_odom="/baro_data", topic_cmd="/cmd_vel", target_depth=-1.5, tolerance=0.2, speed=0.2):
        super(ArrangeDepthAction, self).__init__(name)
        self.topic_odom = topic_odom  # İsim topic_odom kalsa da aslında baro_data dinliyor
        self.topic_cmd = topic_cmd
        self.target_depth = target_depth  
        self.tolerance = tolerance       
        self.speed = abs(speed)          
        
        self.node = None
        self.sub = None
        self.pub = None
        self.current_z = None

    def setup(self, **kwargs):
        self.node = kwargs.get('node')
        if not self.node:
            self.node = rclpy.create_node('smart_depth_action')

        self.sub = self.node.create_subscription(
            Float64,  # DEĞİŞİKLİK 2: Odometry yerine Float32 mesaj tipini dinliyoruz
            self.topic_odom,
            self.baro_callback, # Metot adını mantığa uyması için baro_callback yaptık
            qos_profile=qos_profile_sensor_data
        )
        
        self.pub = self.node.create_publisher(Twist, self.topic_cmd, 10)

    # DEĞİŞİKLİK 3: Gelen verinin içeriğini okuma mantığı değişti
    def baro_callback(self, msg):
        self.current_z = msg.data  # Artık pose.pose.position.z değil, doğrudan 'data'

    def update(self):
        if self.current_z is None:
            self.feedback_message = "Waiting for data..."
            return Status.RUNNING
       
        lower_limit = self.target_depth - self.tolerance
        upper_limit = self.target_depth + self.tolerance

        twist = Twist()

        if lower_limit <= self.current_z <= upper_limit:
            self.stop_vehicle()
            self.feedback_message = f"Target reached! Z: {self.current_z:.3f}"
            return Status.SUCCESS
       
        elif self.current_z > upper_limit:
            twist.linear.z = -self.speed
            self.feedback_message = f"Diving (Shallow)... Z: {self.current_z:.3f} -> Target: {self.target_depth}"
            
        elif self.current_z < lower_limit:
            twist.linear.z = self.speed 
            self.feedback_message = f"Ascending (Deep)... Z: {self.current_z:.3f} -> Target: {self.target_depth}"
       
        if self.pub:
            self.pub.publish(twist)
            
        return Status.RUNNING

    def terminate(self, new_status):
        self.stop_vehicle()

    def stop_vehicle(self):
        if self.pub:
            stop_msg = Twist()
            self.pub.publish(stop_msg)