from rclpy.node import Node
import rclpy
from pymavlink import mavutil
from geometry_msgs.msg import PoseStamped

class PointFollower(Node):
    def __init__(self):
        super().__init__('point_follower')
        self.connect_mavlink()
        self.subscription = self.create_subscription(PoseStamped, '/target_point', self.callback,10)

    def connect_mavlink(self):
        connection_string = '/dev/ttyACM0'  
        self.master = mavutil.mavlink_connection(connection_string)
        self.master.wait_heartbeat()
        self.get_logger().info("connected to mavlink")

    def callback(self, msg):
        target_x = msg.pose.position.x
        target_y = msg.pose.position.y
        target_z = msg.pose.position.z

        type_mask = 0b110111111000

        self.master.mav.set_position_target_local_ned_send(
        0,                                   # time_boot_ms (0=sistem saati)
        self.master.target_system,                # Hedef Sistem ID
        self.master.target_component,             # Hedef Bileşen ID
        mavutil.mavlink.MAV_FRAME_LOCAL_NED, # KOORDİNAT SİSTEMİ (Açıklaması aşağıda)
        type_mask,                           # Maske (Hız/İvme yoksay)
        target_x,                                   # X (Kuzey)
        target_y,                                   # Y (Doğu)
        target_z,                                   # Z (Aşağı/Derinlik)
        0, 0, 0,                             # Hız (Kullanılmıyor)
        0, 0, 0,                             # İvme (Kullanılmıyor)
        0, 0                                 # Yaw (Kullanılmıyor)
    )
        

def main(args=None):
    rclpy.init(args=args)
    node = PointFollower()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()