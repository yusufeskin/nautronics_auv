from pymavlink import mavutil
from std_msgs.msg import Float64MultiArray, MultiArrayDimension

class BaroHandler:
    def __init__(self, node, mav_connection, publisher):
        self.node = node
        self.master = mav_connection
        self.publisher = publisher
        self.logger = node.get_logger()
        self.RHO_WATER = 1000.0 
        self.G_ACCEL = 9.81
        self.p_surface_hpa = 0.0
        self.calibration_samples = []
        self.is_calibrated = False

    def calculate_depth(self, current_pressure_hpa):
        if not self.is_calibrated or self.p_surface_hpa == 0.0: 
            return 0.0
            
        P_diff_pa = (current_pressure_hpa - self.p_surface_hpa) * 100.0
        depth_m = P_diff_pa / (self.RHO_WATER * self.G_ACCEL)
        return max(0.0, depth_m)

    def read_and_publish(self):
        msg_mav = self.master.recv_match(type='SCALED_PRESSURE', blocking=False)
        
        if msg_mav:
            current_pressure = msg_mav.press_abs
            if not self.is_calibrated:
                self.calibration_samples.append(current_pressure)
                if len(self.calibration_samples) >= 10:
                    self.p_surface_hpa = sum(self.calibration_samples) / 10.0
                    self.is_calibrated = True
                    self.logger.info(f"Barometre Kalibre Edildi! Yüzey Basıncı: {self.p_surface_hpa:.2f} hPa")
                return
            depth_m = self.calculate_depth(current_pressure)
            
            multi_array = Float64MultiArray()
            multi_array.data = [depth_m, current_pressure]
            multi_array.layout.dim.append(MultiArrayDimension(label="depth_pressure", size=2, stride=2))
            
            self.publisher.publish(multi_array)