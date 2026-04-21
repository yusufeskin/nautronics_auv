from pymavlink import mavutil
import time
from pymavlink.quaternion import QuaternionBase
import math 
class SetAttitudeHandler:
    def __init__(self, mav_connection, logger):
        self.master = mav_connection
        self.logger = logger
        self.boot_time = time.time()

    #https://github.com/clydemcqueen/ardusub-gitbook/blob/6b26cd3bdd5140ec9b0286fb161e311cc5b2a329/developers/pymavlink/set_target_depth_attitude.py

    def set_target_attitude(self, roll, pitch, yaw):
        self.master.mav.set_attitude_target_send(
        int(1e3 * (time.time() - self.boot_time)), # ms since boot
        self.master.target_system, self.master.target_component,
        # allow throttle to be controlled by depth_hold mode
        mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_THROTTLE_IGNORE,
        # -> attitude quaternion (w, x, y, z | zero-rotation is 1, 0, 0, 0)
        QuaternionBase([math.radians(angle) for angle in (roll, pitch, yaw)]),
        0, 0, 0, 0 # roll rate, pitch rate, yaw rate, thrust
    )
