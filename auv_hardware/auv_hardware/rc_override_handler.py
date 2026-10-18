"""Forwards a pilot demand to ArduSub as RC_CHANNELS_OVERRIDE.

Deliberately thin. All the thinking - converting the controller's demand
into PWM, the calibration, the enable gate, the watchdog that centres the
sticks - happens upstream in auv_actuation's pilot_demand node. This handler
exists only because that node must not open its own MAVLink connection: this
process owns the link, and two processes cannot bind the same endpoint.

It sends only what it is given, and only when it is given something. If the
upstream node dies or is disabled, nothing arrives here and nothing is sent;
ArduSub's RC_OVERRIDE_TIME then expires and the sticks return to neutral,
which in ALT_HOLD means hold depth and heading.

Only channels 1-6 are overridden. 7 and 8 carry UINT16_MAX, which MAVLink
defines as "leave this channel alone", so camera tilt and lights stay with
whoever else drives them.
"""

IGNORE = 65535


class RcOverrideHandler:
    def __init__(self, mav_connection, logger):
        self.master = mav_connection
        self.logger = logger

    def send_demand(self, msg):
        """Forward one auv_core_msgs/PilotDemand.

        The values are already PWM in ArduSub's own channel directions;
        converting them is upstream's job, not this handler's.
        """
        try:
            self.master.mav.rc_channels_override_send(
                self.master.target_system,
                self.master.target_component,
                int(msg.pitch),      # ch1
                int(msg.roll),       # ch2
                int(msg.throttle),   # ch3
                int(msg.yaw),        # ch4
                int(msg.forward),    # ch5
                int(msg.lateral),    # ch6
                IGNORE,              # ch7
                IGNORE,              # ch8
            )
        except Exception as e:
            self.logger.error(
                f"RC_CHANNELS_OVERRIDE gonderilemedi: {e}", throttle_duration_sec=5.0
            )
