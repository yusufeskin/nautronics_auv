from geometry_msgs.msg import Vector3
import math

class AttitudeHandler:
    def __init__(self, node, publisher):
        self.node      = node
        self.logger    = node.get_logger()
        self.publisher = publisher

    def handle_message(self, msg):
        ros_msg = Vector3()
        ros_msg.x = math.degrees(msg.roll)
        ros_msg.y = math.degrees(msg.pitch)
        ros_msg.z = math.degrees(msg.yaw)
        self.publisher.publish(ros_msg)