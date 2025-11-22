from pymavlink import mavutil
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, MultiArrayDimension

RHO_WATER = 1025.0
G_ACCEL = 9.81
MAVLINK_PORT = "/dev/ttyACM0"
BAUD_RATE = 57600
ROS_TOPIC = 'baro_data'
NODE_NAME = 'pixhawk_baro_reader'
P_SURFACE_HPA = 0.0

class BaroDataPublisher(Node):
    def __init__(self):
        super().__init__(NODE_NAME)
        self.get_logger().info(f"Starting -> {NODE_NAME}...")
        self.publisher_ = self.create_publisher(Float64MultiArray, ROS_TOPIC, 10)
        self.connect_to_pixhawk()
        self.calibrate_surface_pressure()
        timer_period = 0.1 
        self.timer = self.create_timer(timer_period, self.publish_baro_data)

    def connect_to_pixhawk(self):
        try:
            self.connection = mavutil.mavlink_connection(MAVLINK_PORT, BAUD_RATE)
            self.connection.wait_heartbeat()
            self.get_logger().info("MAVLink Heartbeat received.")
        except Exception as e:
            self.get_logger().error(f"MAVLink connection error: {e}")
            self.connection = None
            return
    
    def calibrate_surface_pressure(self):
        self.get_logger().info("Calibrating surface pressure (P0)...")
        pressure_sum = 0
        self.msg = self.connection.recv_match(type='SCALED_PRESSURE', blocking=True, timeout=5)
        for n in range(10):
            pressure_sum += self.msg.press_abs
            time.sleep(0.2)
        P_SURFACE_HPA = pressure_sum / 10

    def calculate_depth(self, current_pressure_hpa):
        if P_SURFACE_HPA == 0.0: return 0.0
        P_diff_pa = (current_pressure_hpa - P_SURFACE_HPA) * 100.0
        depth_m = P_diff_pa / (RHO_WATER * G_ACCEL)
        return max(0.0, depth_m)
    
    def publish_baro_data(self):
        if self.msg:
            current_pressure_hpa = self.msg.press_abs
            depth_m = self.calculate_depth(current_pressure_hpa)
            
            # --- Float64MultiArray Mesajını Hazırlama ---
            multi_array = Float64MultiArray()
            
            # Veriyi [Derinlik (m), Basınç (hPa)] olarak paketle
            multi_array.data = [depth_m, current_pressure_hpa]
            
            # Düzen (Layout) Bilgisini Ayarlama (Alıcının yorumlaması için)
            multi_array.layout.dim.append(MultiArrayDimension(label="depth_pressure", size=2, stride=2))
            
            # ROS'ta Yayınla
            self.publisher_.publish(multi_array)
            
            self.get_logger().info(f"Published: Depth: {depth_m:.2f} m | Pressure: {current_pressure_hpa:.2f} hPa")

        

"""
connection.request_data_stream_send(
     connection.target_system,
     connection.target_component,,
     mavutil.mavlink.MAV_DATA_STREAM_RAW_SENSORS,
     50, # Requests per second
     1 # Start time
)
"""
def main(args=None):
    rclpy.init(args=args)
    baro_publisher = BaroDataPublisher()
    
    # Düğümü çalıştır ve olayları bekle
    rclpy.spin(baro_publisher) 

    # Düğüm kapatıldığında temizlik yap
    baro_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()