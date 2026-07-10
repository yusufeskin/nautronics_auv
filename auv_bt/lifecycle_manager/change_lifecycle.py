import py_trees
import rclpy
from lifecycle_msgs.srv import ChangeState
from lifecycle_msgs.msg import Transition

class ChangeLifecycleState(py_trees.behaviour.Behaviour):
    def __init__(self, name: str, node_name: str, transition_id: int):
        super().__init__(name=name)
        self.target_node_name = node_name
        self.transition_id = transition_id
        
        self.client = None
        self.future = None

    def setup(self, **kwargs):
        try:
            self.node = kwargs['node']
        except KeyError as e:
            raise KeyError("ROS 2 node 'kwargs' içinde bulunamadı.") from e

        service_name = f'{self.target_node_name}/change_state'
        self.client = self.node.create_client(ChangeState, service_name)
        
        if not self.client.wait_for_service(timeout_sec=3.0):
            raise RuntimeError(f"Servis zaman aşımına uğradı: {service_name}")

    def initialise(self):
        request = ChangeState.Request()
        request.transition.id = self.transition_id
        self.future = self.client.call_async(request)

    def update(self) -> py_trees.common.Status:
        if self.future is None:
            return py_trees.common.Status.FAILURE

        if not self.future.done():
            return py_trees.common.Status.RUNNING

        response = self.future.result()
        if response is not None and response.success:
            self.feedback_message = f"Başarılı geçiş. Transition ID: {self.transition_id}"
            return py_trees.common.Status.SUCCESS
        else:
            self.feedback_message = "Lifecycle durum değişikliği reddedildi!"
            return py_trees.common.Status.FAILURE