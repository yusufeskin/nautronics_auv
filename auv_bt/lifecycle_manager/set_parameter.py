import py_trees
import rcl_interfaces.srv as rcl_srvs
import rcl_interfaces.msg as rcl_msgs
import rclpy

class SetYoloParameters(py_trees.behaviour.Behaviour):
    def __init__(self, name: str, node_name: str, parameters_dict: dict):
        super().__init__(name=name)
        self.target_node_name = node_name
        self.parameters_dict = parameters_dict
        
        self.client = None
        self.future = None

    def setup(self, **kwargs):
        try:
            self.node = kwargs['node']
        except KeyError as e:
            raise KeyError("ROS 2 node 'kwargs' içinde bulunamadı.") from e

        service_name = f'{self.target_node_name}/set_parameters'
        self.client = self.node.create_client(rcl_srvs.SetParameters, service_name)
        
        if not self.client.wait_for_service(timeout_sec=3.0):
            raise RuntimeError(f"Servis zaman aşımına uğradı: {service_name}")

    def initialise(self):
        request = rcl_srvs.SetParameters.Request()

        for p_name, p_value in self.parameters_dict.items():
            param_msg = rcl_msgs.Parameter()
            param_msg.name = p_name
            
            if isinstance(p_value, float):
                param_msg.value.type = rcl_msgs.ParameterType.PARAMETER_DOUBLE
                param_msg.value.double_value = p_value
            elif isinstance(p_value, bool):
                param_msg.value.type = rcl_msgs.ParameterType.PARAMETER_BOOL
                param_msg.value.bool_value = p_value
            elif isinstance(p_value, int):
                param_msg.value.type = rcl_msgs.ParameterType.PARAMETER_INTEGER
                param_msg.value.integer_value = p_value
            elif isinstance(p_value, str):
                param_msg.value.type = rcl_msgs.ParameterType.PARAMETER_STRING
                param_msg.value.string_value = p_value
            else:
                self.node.get_logger().error(f"Desteklenmeyen veri tipi: {type(p_value)} parametre: {p_name}")
                continue
                
            request.parameters.append(param_msg)

        self.future = self.client.call_async(request)

    def update(self) -> py_trees.common.Status:
        if self.future is None:
            return py_trees.common.Status.FAILURE

        if not self.future.done():
            return py_trees.common.Status.RUNNING

        response = self.future.result()
        if response is not None:
            all_successful = all(result.successful for result in response.results)
            if all_successful:
                self.feedback_message = f"{len(self.parameters_dict)} parametre başarıyla ayarlandı."
                return py_trees.common.Status.SUCCESS
            else:
                self.feedback_message = "Bazı parametreler reddedildi."
                return py_trees.common.Status.FAILURE
        
        return py_trees.common.Status.FAILURE