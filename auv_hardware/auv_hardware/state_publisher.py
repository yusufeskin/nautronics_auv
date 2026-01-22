from pymavlink import mavutil
import rclpy
from rclpy.node import Node
from auv_interfaces.msg import VehicleStatus


MAVLINK_PORT = "udpin:0.0.0.0:14550"
BAUD_RATE = 57600
ROS_TOPIC = 'vehicle/state'
NODE_NAME = 'state_publisher'


class StatePublisher(Node):
    def __init__(self):
        super().__init__(NODE_NAME)
        self.get_logger().info(f"Starting -> {NODE_NAME}...")
        self.publisher_ = self.create_publisher(VehicleStatus, ROS_TOPIC, 10)
        self.connect_to_pixhawk()
        timer_period = 0.05
        self.timer = self.create_timer(timer_period, self.publish)

    def connect_to_pixhawk(self):
        try:
            self.connection = mavutil.mavlink_connection(MAVLINK_PORT, BAUD_RATE)
            self.connection.wait_heartbeat()
            self.get_logger().info("MAVLink Heartbeat received.")
        except Exception as e:
            self.get_logger().error(f"MAVLink connection error: {e}")
            self.connection = None
            return


    def get_mode_string(self, custom_mode):
        modes = {
            0: 'STABILIZE',
            1: 'ACRO',
            2: 'ALT_HOLD',
            3: 'AUTO',
            4: 'GUIDED',
            7: 'CIRCLE',
            9: 'SURFACE',
            16: 'POSHOLD',
            19: 'MANUAL'
        }
        return modes.get(custom_mode, f"UNKNOWN({custom_mode})")
    

    def publish(self):
        if self.connection is None:
            return
        msg_mav = self.connection.recv_match(type='HEARTBEAT', blocking=False)

        if msg_mav:
            ros_msg = VehicleStatus()
            ros_msg.header.stamp = self.get_clock().now().to_msg()
            ros_msg.header.frame_id = "base_link"
            ros_msg.is_connected = True
            ros_msg.mode = self.get_mode_string(msg_mav.custom_mode)

            is_armed = bool(msg_mav.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
            ros_msg.is_armed = is_armed

            self.publisher_.publish(ros_msg)

def main(args=None):
    rclpy.init(args=args)
    state_publisher = StatePublisher()
    rclpy.spin(state_publisher) 
    state_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()