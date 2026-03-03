import py_trees
from geometry_msgs.msg import Twist

class StopVehicle(py_trees.behaviour.Behaviour):
    def __init__(self, name, duration=3.0):
        super(StopVehicle, self).__init__(name)
        self.duration = duration
        self.start_time = None
        self.node = None
        self.publisher = None

    def setup(self, **kwargs):
        try:
            self.node = kwargs['node']
        except KeyError:
            pass
        if self.node is not None:
            self.publisher = self.node.create_publisher(Twist, '/cmd_vel', 10)

    def initialise(self):
        if self.node is not None:
            self.start_time = self.node.get_clock().now()

    def update(self):
        if self.node is None:
            return py_trees.common.Status.FAILURE
            
        cmd = Twist()
        self.publisher.publish(cmd)
        
        elapsed = self.node.get_clock().now() - self.start_time
        if elapsed.nanoseconds / 1e9 >= self.duration:
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.RUNNING
