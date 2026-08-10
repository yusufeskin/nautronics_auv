from pymavlink import mavutil

class LedHandler:
    def __init__(self, master, logger):
        self.master = master
        self.logger = logger
        self.servo_channel = 9 

    def set_led_pwm(self, pwm_value):
        pwm_value = max(1000, min(2000, int(pwm_value)))

        self.master.mav.command_long_send(
            self.master.target_system,     
            self.master.target_component,  
            mavutil.mavlink.MAV_CMD_DO_SET_SERVO, 
            0,                             
            self.servo_channel,            # Kanal 9
            pwm_value,                 
            0, 0, 0, 0, 0                  
        )
        
        # self.logger.info(f"LED 9. Kanala ayarlandı: {pwm_value}")