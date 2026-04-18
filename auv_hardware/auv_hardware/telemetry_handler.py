from pymavlink import mavutil
from auv_interfaces.msg import VehicleStatus

class TelemetryHandler:
    def __init__(self, node, mav_connection, publisher):
        self.node = node
        self.master = mav_connection
        self.logger = node.get_logger()
        self.publisher = publisher


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


    def read_and_publish(self):
        msg_mav = self.master.recv_match(type='HEARTBEAT', blocking=False)

        if msg_mav:
            ros_msg = VehicleStatus()
            ros_msg.header.stamp = self.node.get_clock().now().to_msg()
            ros_msg.header.frame_id = "base_link"
            ros_msg.is_connected = True
            ros_msg.mode = self.get_mode_string(msg_mav.custom_mode)

            is_armed = bool(msg_mav.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
            ros_msg.is_armed = is_armed

            self.publisher.publish(ros_msg)

