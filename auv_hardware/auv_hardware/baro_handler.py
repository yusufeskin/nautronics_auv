from std_msgs.msg import Float64

class BaroHandler:
    def __init__(self, node, publisher):
        self.node      = node
        self.logger    = node.get_logger()
        self.publisher = publisher

    def handle_message(self, msg):
        if msg.get_srcComponent() != 1:
            return

        rel_alt = (msg.relative_alt / 1000.0)
        if rel_alt == 0.0:
            return 
        depth_msg = Float64()
        depth_msg.data = rel_alt
        self.publisher.publish(depth_msg)