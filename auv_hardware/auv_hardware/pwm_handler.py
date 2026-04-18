from pymavlink import mavutil

class PwmHandler:
    def __init__(self, mav_connection, logger):
        self.master = mav_connection
        self.logger = logger
        
    def send_pwm(self, message):
        self.message = message
        channels = [65535] * 8
        for i in range(min(len(message), 8)):
            channels[i] = message[i]
        self.master.mav.rc_channels_override_send(
            self.master.target_system,
            self.master.target_component,
            channels[0],
            channels[1],
            channels[2],
            channels[3],
            channels[4],
            channels[5],
            channels[6],
            channels[7],
        )
