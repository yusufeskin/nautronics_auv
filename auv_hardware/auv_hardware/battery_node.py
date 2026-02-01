import rclpy
from rclpy.node import Node
from sensor_msgs.msg import BatteryState
from std_msgs.msg import Float32
from pymavlink import mavutil
import sys

class BatteryNode(Node):
    def __init__(self):
        super().__init__('battery_node')
        
        # If using Simulator: 'udpin:0.0.0.0:14550'
        # If using USB:       '/dev/ttyACM0'
        self.connection_string = 'udpin:0.0.0.0:14550' 
        self.baud_rate = 57600
        
        self.get_logger().info(f'Connecting to Pixhawk at {self.connection_string}...')

        try:
            self.master = mavutil.mavlink_connection(self.connection_string, baud=self.baud_rate)
            self.master.wait_heartbeat()
            self.get_logger().info('Connected to Pixhawk (Heartbeat Received)')
            
            self.master.mav.request_data_stream_send(
                self.master.target_system,
                self.master.target_component,
                mavutil.mavlink.MAV_DATA_STREAM_EXTENDED_STATUS, 
                20, 
                1
            )
        except Exception as e:
            self.get_logger().error(f'Failed to connect: {e}')
            sys.exit(1)

        
        self.bat_state_pub = self.create_publisher(BatteryState, '/battery/status', 10)
        
        self.volt_pub = self.create_publisher(Float32, '/battery/voltage', 10)
        self.curr_pub = self.create_publisher(Float32, '/battery/current', 10)

        # Timer 20 Hz
        self.timer = self.create_timer(0.05, self.read_mavlink_data)

    def read_mavlink_data(self):
        while True:
            msg = self.master.recv_match(type='SYS_STATUS', blocking=False)
            
            if not msg:
                break 
            
            
            # mV to Volts
            voltage_v = msg.voltage_battery / 1000.0
            # cA to Amps
            current_a = msg.current_battery / 100.0  
            # 0-100 to 0.0-1.0
            percentage = msg.battery_remaining / 100.0 

            bat_msg = BatteryState()
            bat_msg.header.stamp = self.get_clock().now().to_msg()
            bat_msg.header.frame_id = "base_link"
            bat_msg.voltage = voltage_v
            bat_msg.current = current_a
            bat_msg.percentage = percentage
            bat_msg.present = True
            bat_msg.power_supply_technology = BatteryState.POWER_SUPPLY_TECHNOLOGY_LIPO
            
            self.bat_state_pub.publish(bat_msg)

            v_msg = Float32()
            v_msg.data = voltage_v
            self.volt_pub.publish(v_msg)

            c_msg = Float32()
            c_msg.data = current_a
            self.curr_pub.publish(c_msg)

def main(args=None):
    rclpy.init(args=args)
    node = BatteryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
