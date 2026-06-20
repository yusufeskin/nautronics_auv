from std_msgs.msg import UInt16MultiArray

class MotorOutputHandler:
    def __init__(self, node, publisher):
        self.node = node
        self.publisher = publisher

    def handle_message(self, msg):
        ros_msg = UInt16MultiArray()
        ros_msg.data = [
            msg.servo1_raw,
            msg.servo2_raw,
            msg.servo3_raw,
            msg.servo4_raw,
            msg.servo5_raw,
            msg.servo6_raw,
            msg.servo7_raw,
            msg.servo8_raw
        ]
        self.publisher.publish(ros_msg)
