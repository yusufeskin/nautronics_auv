from pymavlink import mavutil


class SetSpeedHandler:
    def __init__(self, mav_connection, logger):
        self.master = mav_connection
        self.logger = logger

    def set_speed(self, speed_mps, speed_type=1):
        # speed_type: 0=airspeed, 1=ground speed (yatay), 2=climb, 3=descent
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_DO_CHANGE_SPEED,
            0,
            speed_type, speed_mps, -1, 0, 0, 0, 0
        )
        self.logger.info(f"Hedef hiz ayarlandi: {speed_mps:.2f} m/s (tip={speed_type})")
