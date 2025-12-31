#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import BatteryState
from std_msgs.msg import Float32
from pymavlink import mavutil
import sys

class PixhawkBatteryNode(Node):
    def __init__(self):
        super().__init__('pixhawk_battery_node')

        
        # For Usb : '/dev/ttyACM0'
        # For raspberryPI use (a bug could occur): '/dev/ttyAMA0' or '/dev/serial0'
        # For simulation 'udpin:0.0.0.0:14550'
        self.connection_string = 'udpin:0.0.0.0:14550' 
        self.baud_rate = 57600
        
        self.get_logger().info(f'Attempting to connect to Pixhawk at {self.connection_string}...')

        try:
        
            self.master = mavutil.mavlink_connection(self.connection_string, baud=self.baud_rate)
            
        
            self.master.wait_heartbeat()
            self.get_logger().info(
                f'Connected to system (System ID: {self.master.target_system}, '
                f'Component ID: {self.master.target_component})'
            )
        except Exception as e:
            self.get_logger().error(f'Failed to connect to Pixhawk: {e}')
            sys.exit(1)

        # Publisher 1 
        self.bat_state_pub = self.create_publisher(BatteryState, '/battery/status', 10)

        #  Publisher 2 and 3 
        self.volt_pub = self.create_publisher(Float32, '/battery/voltage', 10)
        self.curr_pub = self.create_publisher(Float32, '/battery/current', 10)

        
        self.timer = self.create_timer(0.05, self.read_mavlink_data)

    def read_mavlink_data(self):
        
        msg = self.master.recv_match(type='SYS_STATUS', blocking=False)

        if msg:
            
            voltage_v = msg.voltage_battery / 1000.0
            
           
            current_a = msg.current_battery / 100.0
            
           
            percentage = msg.battery_remaining / 100.0

            # 1. Publish /battery/status (sensor_msgs/BatteryState) 
            bat_msg = BatteryState()
            bat_msg.header.stamp = self.get_clock().now().to_msg()
            bat_msg.header.frame_id = "base_link"
            
            bat_msg.voltage = voltage_v
            bat_msg.current = current_a
            bat_msg.percentage = percentage
            bat_msg.present = True
            
            # For lipo battery
            bat_msg.power_supply_technology = BatteryState.POWER_SUPPLY_TECHNOLOGY_LIPO
            
            self.bat_state_pub.publish(bat_msg)

            # 2. Publish /battery/voltage (std_msgs/Float32) 
            v_msg = Float32()
            v_msg.data = voltage_v
            self.volt_pub.publish(v_msg)

            # 3. Publish /battery/current (std_msgs/Float32) 
            c_msg = Float32()
            c_msg.data = current_a
            self.curr_pub.publish(c_msg)

def main(args=None):
    rclpy.init(args=args)
    node = PixhawkBatteryNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()