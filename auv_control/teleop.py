#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist
from auv_interfaces.srv import SetVehicleMode

class AUVTeleop(Node):
    def __init__(self):
        super().__init__('auv_teleop_joy')

        self.AXIS_LEFT_STICK_Y = 1   # Heave için
        self.AXIS_RIGHT_STICK_Y = 4  # Surge için 
        self.AXIS_L2 = 2             # Sola Yaw
        self.AXIS_R2 = 5             # Sağa Yaw
        

        self.BUTTON_ENABLE = 4 # LB 

        self.SCALE_SURGE = 1.0  # m/s
        self.SCALE_HEAVE = 0.8  # m/s
        self.SCALE_YAW = 1.5    # rad/s

        self.is_armed = False        
        self.prev_enable_button = 0  


        
        self.arm_client = self.create_client(SetVehicleMode, '/change_mode')
        while not self.arm_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Mode servisi bekleniyor...')
        


        self.joy_sub = self.create_subscription(
            Joy, 'joy', self.joy_callback, 10)
        
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        self.get_logger().info("AUV Teleop Başlatıldı.")


    def send_mode_command(self, mode_string: str):

        request = SetVehicleMode.Request()
        request.mode_name = mode_string
        
        future = self.arm_client.call_async(request)
        
        future.add_done_callback(self.mode_response_callback)

    def mode_response_callback(self, future):
        try:
            response = future.result()
            if response.success:
                self.get_logger().info(f"BAŞARILI: {response.message}")
            else:
                self.get_logger().error(f"BAŞARISIZ: {response.message}")
                self.is_armed = not self.is_armed
        except Exception as e:
            self.get_logger().error(f"Servis çağrısı sırasında hata oluştu: {e}")
    

    def joy_callback(self, msg: Joy):

        twist = Twist()
        
        current_button_state = msg.buttons[self.BUTTON_ENABLE]

        if current_button_state == 1 and self.prev_enable_button == 0:
            self.is_armed = not self.is_armed 
            mode_str = "ARM" if self.is_armed else "DISARM"
            self.send_mode_command(mode_str)
            self.get_logger().info(f"Komut gönderiliyor: {mode_str}")
        
        self.prev_enable_button = current_button_state


        if self.is_armed:
            
            twist.linear.z = msg.axes[self.AXIS_LEFT_STICK_Y] * self.SCALE_HEAVE

            twist.linear.x = msg.axes[self.AXIS_RIGHT_STICK_Y] * self.SCALE_SURGE

            l2_val = (1.0 - msg.axes[self.AXIS_L2]) / 2.0
            r2_val = (1.0 - msg.axes[self.AXIS_R2]) / 2.0

            twist.angular.z = (l2_val - r2_val) * self.SCALE_YAW

        else:

            twist.linear.x = 0.0
            twist.linear.z = 0.0
            twist.angular.z = 0.0

        self.cmd_pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = AUVTeleop()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()