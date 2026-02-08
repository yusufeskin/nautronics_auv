#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from pymavlink import mavutil
from auv_interfaces.srv import SetVehicleMode

CONNECTION_STRING = 'udpin:0.0.0.0:14550'
BAUD_RATE = 57600

class VehicleManager(Node):
    def __init__(self):
        super().__init__('vehicle_manager')
        self.get_logger().info(f"Pixhawk aranıyor: {CONNECTION_STRING}...")
        self.link_bridge()

        self.srv = self.create_service(
            SetVehicleMode, 
            '/change_mode', 
            self.handle_mode_service
        )

        self.modes = {
            'STABILIZE': 0, 'ACRO': 1, 'ALT_HOLD': 2, 'AUTO': 3,
            'GUIDED': 4, 'LOITER': 5, 'CIRCLE': 7, 'SURFACE': 9,
            'POSHOLD': 16, 'MANUAL': 19
        }

    def link_bridge(self):
        try:
            self.connection = mavutil.mavlink_connection(CONNECTION_STRING, baud=BAUD_RATE)
            self.connection.wait_heartbeat()
            self.get_logger().info("BAĞLANDI! Pixhawk Heartbeat alındı.")
        except Exception as e:
            self.get_logger().error(f"Bağlantı Hatası: {e}")
            self.connection = None
        
    def handle_mode_service(self, request, response):
        target_mode = request.mode_name.upper()
        self.get_logger().info(f"[SERVİS] Mod Değiştirme İsteği: {target_mode}")

        if self.connection is None:
            response.success = False
            response.message = "Pixhawk bağlı değil!"
            return response

        if target_mode in self.modes:
            mode_id = self.modes[target_mode]
            try:
                self.connection.set_mode(mode_id)
                response.success = True
                response.message = f"Pixhawk'a {target_mode} komutu yollandı."
            except Exception as e:
                response.success = False
                response.message = f"MAVLink Hatası: {e}"

        elif target_mode == 'ARM':
            self.connection.arducopter_arm()
            response.success = True
            response.message = "Arm edildi, komutlara hazır"

        elif target_mode == 'DISARM':
            self.connection.arducopter_disarm()
            response.success = True
            response.message = "Disarm edildi, komutlara kapali"

        
        else:
            response.success = False
            response.message = f"Geçersiz Mod: {target_mode}"
        
        return response
def main():
    rclpy.init()
    node = VehicleManager()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()