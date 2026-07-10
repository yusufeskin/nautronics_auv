import py_trees
import rclpy
from geometry_msgs.msg import Vector3
from py_trees.common import Status
from rclpy.qos import qos_profile_sensor_data
from rclpy.action import ActionClient
from auv_interfaces.action import YawAndScan

def normalize_angle_deg(angle):
    while angle > 180.0: 
        angle -= 360.0
    while angle < -180.0: 
        angle += 360.0
    return angle

class SaveInitialYaw(py_trees.behaviour.Behaviour):
    def __init__(self, name="Save Initial Yaw"):
        super(SaveInitialYaw, self).__init__(name)
        self.node = None
        self.sub = None
        self.current_yaw = None
        
        self.blackboard = py_trees.blackboard.Client(name=self.name)
        self.blackboard.register_key(key="reference_yaw", access=py_trees.common.Access.WRITE)

    def setup(self, **kwargs):
        self.node = kwargs.get('node')
        if not self.node:
            self.node = rclpy.create_node('save_initial_yaw_node')
            
        self.sub = self.node.create_subscription(
            Vector3,
            '/current_attitude',
            self.attitude_callback,
            qos_profile_sensor_data
        )

    def attitude_callback(self, msg):
        self.current_yaw = msg.z

    def update(self):
        if self.current_yaw is None:
            return Status.RUNNING
        
        self.blackboard.reference_yaw = self.current_yaw
        return Status.SUCCESS

class AbsoluteYawClient(py_trees.behaviour.Behaviour):
    def __init__(self, name="Absolute Yaw Client", angle_increment=90.0, speed=0.05):
        super(AbsoluteYawClient, self).__init__(name=name)
        self.angle_increment = angle_increment
        self.speed = speed
        self.node = None
        self.action_client = None
        self.send_goal_future = None
        self.get_result_future = None
        self.goal_handle = None

        self.blackboard = py_trees.blackboard.Client(name=self.name)
        self.blackboard.register_key(key="reference_yaw", access=py_trees.common.Access.READ)
        self.blackboard.register_key(key="reference_yaw", access=py_trees.common.Access.WRITE)

    def setup(self, **kwargs):
        try:
            self.node = kwargs['node']
        except KeyError:
            safe_name = self.name.replace(" ", "_").replace("(", "").replace(")", "").lower()
            self.node = rclpy.create_node(f"{safe_name}_node")
            
        self.action_client = ActionClient(self.node, YawAndScan, "/absolute_yaw")

    def initialise(self):
        self.send_goal_future = None
        self.get_result_future = None
        self.goal_handle = None
        
        try:
            ref_yaw = self.blackboard.reference_yaw
        except KeyError:
            ref_yaw = 0.0
            
        target_yaw = normalize_angle_deg(ref_yaw + self.angle_increment)
        self.blackboard.reference_yaw = target_yaw
        
        goal_msg = YawAndScan.Goal()
        goal_msg.target_angle_deg = float(target_yaw)
        goal_msg.angular_speed = float(self.speed)
        
        self.send_goal_future = self.action_client.send_goal_async(goal_msg)

    def update(self):
        if self.send_goal_future and not self.send_goal_future.done():
            return Status.RUNNING

        if self.send_goal_future and self.send_goal_future.done() and not self.get_result_future:
            self.goal_handle = self.send_goal_future.result()
            
            if not self.goal_handle.accepted:
                return Status.FAILURE
                
            self.get_result_future = self.goal_handle.get_result_async()
            return Status.RUNNING

        if self.get_result_future and not self.get_result_future.done():
            return Status.RUNNING

        if self.get_result_future and self.get_result_future.done():
            return Status.SUCCESS

        return Status.FAILURE