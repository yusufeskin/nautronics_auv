import py_trees
import rclpy
from geometry_msgs.msg import Vector3
from py_trees.common import Status
from rclpy.qos import qos_profile_sensor_data
import py_trees_ros.action_clients
from auv_interfaces.action import YawAndScan

def normalize_angle_deg(angle):
    while angle > 180.0: angle -= 360.0
    while angle < -180.0: angle += 360.0
    return angle

class SaveInitialYaw(py_trees.behaviour.Behaviour):
    def __init__(self, name="Save Initial Yaw"):
        super(SaveInitialYaw, self).__init__(name)
        self.node = None
        self.sub = None
        self.current_yaw = None

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
        self.blackboard = py_trees.blackboard.Client(name=self.name)
        self.blackboard.register_key(key="reference_yaw", access=py_trees.common.Access.WRITE)

    def attitude_callback(self, msg):
        self.current_yaw = msg.z

    def update(self):
        if self.current_yaw is None:
            return Status.RUNNING
        
        self.blackboard.reference_yaw = self.current_yaw
        return Status.SUCCESS

class AbsoluteYawClient(py_trees_ros.action_clients.ActionClient):
    def __init__(self, name="Absolute Yaw Client", angle_increment=90.0, speed=0.05):
        super(AbsoluteYawClient, self).__init__(
            name=name,
            action_type=YawAndScan,
            action_name="/absolute_yaw",
            action_goal=YawAndScan.Goal(), # Dummy goal, replaced in initialise
            generate_feedback_message=lambda msg: f"Turned {msg.feedback.current_angle_deg:.2f} deg"
        )
        self.angle_increment = angle_increment
        self.speed = speed
        self.blackboard = py_trees.blackboard.Client(name=self.name)
        self.blackboard.register_key(key="reference_yaw", access=py_trees.common.Access.READ)
        self.blackboard.register_key(key="reference_yaw", access=py_trees.common.Access.WRITE)

    def initialise(self):
        try:
            ref_yaw = self.blackboard.reference_yaw
        except KeyError:
            self.logger.warning("No reference_yaw found on blackboard! Using 0.0")
            ref_yaw = 0.0
            
        target_yaw = normalize_angle_deg(ref_yaw + self.angle_increment)
        self.blackboard.reference_yaw = target_yaw
        
        self.action_goal = YawAndScan.Goal()
        self.action_goal.target_angle_deg = float(target_yaw)
        self.action_goal.angular_speed = float(self.speed)
        
        self.logger.info(f"AbsoluteYawClient sending absolute target: {target_yaw}")
        
        super(AbsoluteYawClient, self).initialise()
