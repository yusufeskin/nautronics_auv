import math
from pymavlink import mavutil

class SetPositionTargetHandler:
    def __init__(self, mav_connection, logger):
        self.master = mav_connection
        self.logger = logger
        self.sys_id = self.master.target_system
        self.comp_id = self.master.target_component

    def set_target_position_local(self, x, y, z, yaw=None):
        if yaw is None:
            # Bitmask to ignore everything except position (x, y, z)
            # Type_mask: 0b0000111111111000 = 0x0FF8
            type_mask = int(0b0000111111111000)
            yaw_rad = 0.0
        else:
            # Same as above but also use yaw (bit10=0), still ignore yaw_rate
            # Type_mask: 0b0000101111111000 = 0x0BF8
            type_mask = int(0b0000101111111000)
            yaw_rad = math.radians(yaw)

        self.master.mav.set_position_target_local_ned_send(
            0,                      # time_boot_ms (not used)
            self.sys_id,
            self.comp_id,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            type_mask,
            x, y, z,                # Position (m)
            0, 0, 0,                # Velocity (m/s) (ignored by mask)
            0, 0, 0,                # Acceleration (ignored by mask)
            yaw_rad, 0               # Yaw (rad), Yaw_rate (ignored by mask)
        )
        self.logger.info(f"pixhawka gonderildi X={x:.2f}m, Y={y:.2f}m, Z={z:.2f}m" + (f", Yaw={yaw:.1f}°" if yaw is not None else ""))
