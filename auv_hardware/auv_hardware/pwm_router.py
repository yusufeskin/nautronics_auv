import rclpy
from std_msgs.msg import UInt16MultiArray
import rclpy
from rclpy.node import Node
from pymavlink import mavutil

class PwmRouter(Node):
    def __init__(self):
        super().__init__('pwm_router')
        self.get_logger().info("baŞladı")
        self.connect_pixhawk()
        self.subscription = self.create_subscription(
            UInt16MultiArray, 'pwm_router', self.callback, 10)
        
    def callback(self, msg):
        message = msg.data
        self.send_pwm(message)



    def connect_pixhawk(self):
        self.connection_string = '/dev/ttyACM0'
        self.baudrate = 57600
        self.master = mavutil.mavlink_connection(self.connection_string, baud=self.baudrate)
        self.master.wait_heartbeat()
        self.get_logger().info("connected")


    def send_pwm(self, pwm_values):
        channels = [65535] * 8
        for i in range(min(len(pwm_values), 8)):
            channels[i] = pwm_values[i]
        self.master.mav.rc_channels_override_send(
            self.master.target_system,
            self.master.target_component,
            channels[0],
            channels[1],
            channels[2],
            channels[3],
            channels[4],
            channels[5],
            channels[6],
            channels[7],
        )


def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(PwmRouter())
    rclpy.shutdown()

if __name__ == '__main__':
    main()










