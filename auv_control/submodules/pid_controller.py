class PID():
    def __init__(self, KP, KI, KD):
        self.KP = KP
        self.KI = KI
        self.KD = KD
        self.error_sum = 0
        self.last_error = 0
    
    def calculate_power(self, error):
        self.error_sum += error
        self.p_gain = error * self.KP
        self.i_gain = self.error_sum * self.KI
        self.d_gain = (error - self.last_error) * self.KD
        self.last_error = error
        return self.p_gain + self.i_gain + self.d_gain