from pymavlink import mavutil

class SetPositionTargetHandler:
    def __init__(self, mav_connection, logger):
        self.master = mav_connection
        self.logger = logger
        self.sys_id = self.master.target_system
        self.comp_id = self.master.target_component

    def set_target_position_local(self, x, y, z):
        # Bitmask to ignore everything except position (x, y, z)
        # Type_mask: 0b0000111111111000 = 0x0FF8
        type_mask = int(0b0000111111111000)

        self.master.mav.set_position_target_local_ned_send(
            0,                      # time_boot_ms (not used)
            self.sys_id,
            self.comp_id,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED, 
            type_mask,
            x, y, z,                # Position (m)
            0, 0, 0,                # Velocity (m/s) (ignored by mask)
            0, 0, 0,                # Acceleration (ignored by mask)
            0, 0                    # Yaw, Yaw_rate (ignored by mask)
        )
        self.logger.info(f"pixhawka gonderildi X={x:.2f}m, Y={y:.2f}m, Z={z:.2f}m")
