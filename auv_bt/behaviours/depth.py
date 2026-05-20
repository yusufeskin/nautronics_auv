import py_trees
import rclpy
from std_msgs.msg import Float64
from py_trees.common import Status
from rclpy.qos import qos_profile_sensor_data

class DepthCheckerCondition(py_trees.behaviour.Behaviour):
    def __init__(self, name="Check Depth Reached", topic="/baro_data2", target_depth=-1.5, tolerance=0.2):
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
            Float64, self.topic, self.callback, qos_profile=qos_profile_sensor_data
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