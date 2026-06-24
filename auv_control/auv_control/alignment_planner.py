import math
import rclpy
from rclpy.node import Node
from auv_interfaces.srv import ComputeAlignmentPlan

def normalize_angle_deg(angle_deg: float) -> float:
    angle = math.fmod(angle_deg + 180.0, 360.0)
    if angle <= 0.0:
        angle += 360.0
    return angle - 180.0

class AlignmentPlannerNode(Node):
    def __init__(self):
        super().__init__('alignment_planner')
        self._srv = self.create_service(
            ComputeAlignmentPlan,
            'compute_alignment_plan',
            self._handle_request,
        )
        self.get_logger().info(
            "service ready"
        )

    def _handle_request(
        self,
        request: ComputeAlignmentPlan.Request,
        response: ComputeAlignmentPlan.Response,
    ) -> ComputeAlignmentPlan.Response:

        pix_deg  = request.pixhawk_angle_deg
        pnp_deg  = request.pnp_angle_deg
        dist_m   = request.pnp_distance_m
        off_x    = request.offset_x   
        off_y    = request.offset_y   

        if dist_m < 0.0:
            response.success = False
            response.message = f"pnp_distance_m invalid {dist_m:.3f} m"
            self.get_logger().warn(response.message)
            return 

        global_bearing_to_origin_deg = normalize_angle_deg(pix_deg + pnp_deg)
        gbr = math.radians(global_bearing_to_origin_deg)

        origin_north = dist_m * math.cos(gbr)   
        origin_east  = dist_m * math.sin(gbr)   
        
        target_north = origin_north + off_y
        target_east  = origin_east  + off_x

        target_yaw_rad = math.atan2(target_east, target_north)
        target_yaw_deg = math.degrees(target_yaw_rad)

        net_surge_m = math.sqrt(target_north ** 2 + target_east ** 2)

        self.get_logger().info(
            f" Target Yaw {target_yaw_deg:+8.2f}°  \n"
            f" Net Surge  {net_surge_m:8.3f} m     \n"
        )

        response.success        = True
        response.target_yaw_deg = target_yaw_deg
        response.net_surge_m    = net_surge_m
        return response

def main(args=None):
    rclpy.init(args=args)
    node = AlignmentPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
