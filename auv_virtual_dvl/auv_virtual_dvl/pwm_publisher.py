import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from pymavlink import mavutil
class rc_data_reader(Node):
    def __init__(self):
        super().__init__('rc_data_reader')
        self.connect_mavlink()
        self.publisher = self.create_publisher(Float64MultiArray, '/auv/pwm_data', 10)
        self.timer = self.create_timer(0.02, self.timer_callback)

    def timer_callback(self):
        #get the PWM values sent to the each thrusters and publish them to the ROS topic, so I can record a rosbag
        msg = self.connection.recv_match(type='SERVO_OUTPUT_RAW', blocking=False)
        if msg:
            pwm_values = [
                float(msg.servo1_raw),
                float(msg.servo2_raw),
                float(msg.servo3_raw),
                float(msg.servo4_raw),
                float(msg.servo5_raw),
                float(msg.servo6_raw),
                float(msg.servo7_raw),
                float(msg.servo8_raw)
            ]
            ros_msg = Float64MultiArray()
            ros_msg.data = pwm_values
            self.publisher.publish(ros_msg)


        
    def connect_mavlink(self):
        self.connection = mavutil.mavlink_connection('tcp:127.0.0.1:5762') #I used tcp instead of udp because there was buffer problems due to the data stream rate 
        #and tcp more suitable for now, if we face an issue like that in real env we should put "--streamrate=-1" when executing the mavproxy.py
        #https://github.com/ArduPilot/ardupilot/issues/19761, I found the solution from this topic
        self.connection.wait_heartbeat()
        self.get_logger().info('connected to mavlink')
        frequency_hz = 50
        self.connection.mav.command_long_send(
            self.connection.target_system,
            self.connection.target_component,
            mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
            0,
            36,
            1e6 / frequency_hz,
            0, 0, 0, 0, 0
        )

def main(args=None):
    rclpy.init(args=args)
    rc_reader = rc_data_reader()
    rclpy.spin(rc_reader)
    rc_reader.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()