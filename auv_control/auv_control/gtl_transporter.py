#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import math
from geometry_msgs.msg import Point
from auv_interfaces.srv import GoToGpsTarget 

class GtlTransporter(Node):
    def __init__(self):
        super().__init__('gtl_transporter')
        self.target_pub = self.create_publisher(Point, 'target_position_local', 10)

        self.srv = self.create_service(
            GoToGpsTarget,
            'compute_and_go_gps', 
            self.handle_gps_request
        )
        
        self.METERS_PER_DEGREE = 111320.0
        self.get_logger().info("service started")

    def handle_gps_request(self, request, response):

        baslangic_lat = request.baslangic_lat
        baslangic_lon = request.baslangic_lon
        hedef_lat  = request.hedef_lat
        hedef_lon  = request.hedef_lon

        delta_x = (hedef_lat - baslangic_lat) * self.METERS_PER_DEGREE
        lat_rad = math.radians(baslangic_lat)
        delta_y = (hedef_lon - baslangic_lon) * self.METERS_PER_DEGREE * math.cos(lat_rad)
        
        msg = Point()
        msg.x = float(delta_x)
        msg.y = float(delta_y)
        msg.z = float(request.target_depth)
        self.target_pub.publish(msg)
        
        self.get_logger().info(f"gonderildi X={delta_x:.2f}m, Y={delta_y:.2f}m")

        response.success = True
        response.calculated_x = float(delta_x)
        response.calculated_y = float(delta_y)
        return response

def main(args=None):
    rclpy.init(args=args)
    node = GtlTransporter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()

if __name__ == '__main__':
    main()
