from pymavlink import mavutil

class ModeHandler:
    def __init__(self, mav_connection, logger):
        self.master = mav_connection
        self.logger = logger
        self.modes = {
            'STABILIZE': 0, 'ACRO': 1, 'ALT_HOLD': 2, 'AUTO': 3,
            'MANUAL': 19
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
