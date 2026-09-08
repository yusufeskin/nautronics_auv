"""Forwards a pilot demand to ArduSub as MANUAL_CONTROL.

Deliberately thin. All the thinking - converting a commanded wrench into
stick counts, the full-scale calibration, the enable gate, the watchdog that
falls back to neutral - happens upstream in auv_actuation's pilot_demand node.
This handler exists only because that node must not open its own MAVLink
connection: this process owns the link, and two processes cannot bind the same
endpoint.

It sends only what it is given, and only when it is given something. If the
upstream node dies, nothing arrives here and nothing is sent, which is the
safe direction - the autopilot simply stops seeing pilot input and holds.

Do not run this alongside PwmHandler's RC_CHANNELS_OVERRIDE. Both are pilot
input by another name, and the autopilot will act on whichever arrived last.
"""

from pymavlink import mavutil


class ManualControlHandler:
    def __init__(self, mav_connection, logger):
        self.master = mav_connection
        self.logger = logger

    def send_demand(self, msg):
        """Forward one auv_core_msgs/PilotDemand.

        The axis values are already in MANUAL_CONTROL's [-1000, 1000] range
        and already in the autopilot's own axis conventions; converting them
        is upstream's job, not this handler's.
        """
        try:
            self.master.mav.manual_control_send(
                self.master.target_system,
                int(msg.x),
                int(msg.y),
                int(msg.z),
                int(msg.r),
                int(msg.buttons),
            )
        except Exception as e:
            self.logger.error(
                f"MANUAL_CONTROL gonderilemedi: {e}", throttle_duration_sec=5.0
            )
