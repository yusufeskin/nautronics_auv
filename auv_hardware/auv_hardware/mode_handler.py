from pymavlink import mavutil

class ModeHandler:
    def __init__(self, mav_connection, logger):
        self.master = mav_connection
        self.logger = logger
        self.modes = {
            'STABILIZE': 0, 'ACRO': 1, 'ALT_HOLD': 2, 'AUTO': 3,
            'MANUAL': 19, 'GUIDED': 4, 'POSHOLD': 16
        }

    def change_mode_callback(self, request, response):
        requested_mode = request.mode_name.upper()
        if requested_mode in self.modes:
            mode_id = self.modes[requested_mode]
            self.master.set_mode(mode_id)
            self.logger.info(f"Mod değiştirildi: {requested_mode}")
            response.success = True
        else:
            self.logger.error(f"Geçersiz mod: {requested_mode}")
            response.success = False
            
        return response

    def arm_callback(self, request, response):
        arm = 1 if request.data else 0
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            arm, 0, 0, 0, 0, 0, 0
        )
        action = "Arm" if request.data else "Disarm"
        self.logger.info(f"Komut gönderildi: {action}")
        response.success = True
        response.message = f"{action} komutu iletildi"
        return response
