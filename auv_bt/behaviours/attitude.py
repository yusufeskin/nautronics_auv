############### This file will be used for attitude check, e.g.  vehicle takes 90 degree yaw turn command,
# file looks if turning completed or not, anyway i will handle this file as well


import py_trees
import rclpy
from geometry_msgs.msg import Vector3
from py_trees.common import Status
from rclpy.qos import qos_profile_sensor_data

class DepthCheckerCondition(py_trees.behaviour.Behaviour):
    def __init__(self, name="Check Depth Reached", topic="/baro_data", target_depth=-1.5, tolerance=0.2):
        super(DepthCheckerCondition, self).__init__(name)
        self.topic = topic
        self.target_depth = target_depth
        self.tolerance = tolerance
        
        self.node = None
        self.sub = None
        self.current_z = None

    def setup(self, **kwargs):
        self.node = kwargs.get('node')
        self.sub = self.node.create_subscription(
            Vector3, self.topic, self.callback, qos_profile=qos_profile_sensor_data
        )

    def callback(self, msg):
        self.current_z = msg.data

    def update(self):
        if self.current_z is None:
            return Status.RUNNING

        lower_limit = self.target_depth - self.tolerance
        upper_limit = self.target_depth + self.tolerance

        if lower_limit <= self.current_z <= upper_limit:
            return Status.SUCCESS
        else:
            return Status.RUNNING
        
