from std_msgs.msg import Float64

class BaroHandler2:
    def __init__(self, node, publisher):
        self.node      = node
        self.logger    = node.get_logger()
        self.publisher = publisher

        self.fluid_density = 997.0
        self.gravity       = 9.80665

        self.surface_pressure = None
        self.calibration_samples = []
        self.required_samples = 20

    def handle_message(self, msg):
        if msg.get_srcComponent() != 1:
            return
        press_abs_pa = msg.press_abs * 100.0

        if self.surface_pressure is None:
            self.calibration_samples.append(press_abs_pa)
            if len(self.calibration_samples) >= self.required_samples:
                self.surface_pressure = sum(self.calibration_samples) / self.required_samples
                self.logger.info(f"Barometer calibrated surface pressure is: {self.surface_pressure:.2f} Pa")
            return

        depth_meters = -((press_abs_pa - self.surface_pressure) / (self.fluid_density * self.gravity))

        depth_msg = Float64()
        depth_msg.data = depth_meters
        self.publisher.publish(depth_msg)