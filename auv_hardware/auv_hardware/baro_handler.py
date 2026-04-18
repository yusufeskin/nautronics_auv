from std_msgs.msg import Float64

class BaroHandler:
    def __init__(self, node, publisher):
        self.node      = node
        self.logger    = node.get_logger()
        self.publisher = publisher

    def handle_message(self, msg):
        depth_msg      = Float64()
        depth_msg.data = abs(msg.alt)
        self.publisher.publish(depth_msg)