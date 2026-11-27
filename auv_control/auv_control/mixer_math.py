"""Pure thruster-mixing math used by the thruster_mixer node (no ROS imports)."""

import numpy as np

# PWM range
PWM_MIN = 1000
PWM_MAX = 2000
PWM_NEUTRAL = 1500

# 16V thruster data mapping (PWM vs thrust in kgf).
# Reverse thrusts are represented as negative values.
THRUSTER_DATA_PWM = np.array([
    1000, 1050, 1100, 1150, 1200, 1250, 1300, 1350, 1500,
    1650, 1700, 1750, 1800, 1850, 1900, 1950, 2000
])
THRUSTER_DATA_THRUST = np.array([
    -3.52, -3.06, -2.75, -2.11, -1.86, -1.37, -1.05, -0.74, 0.0,
    0.73, 1.13, 1.69, 2.54, 3.15, 3.76, 4.43, 4.68
])
MAX_FORWARD_THRUST = 4.68
MAX_REVERSE_THRUST = 3.52

# Weights (can be tuned)
MIXING_MATRIX = np.array([
    # x(surge) y(sway) z(heave) r(roll) p(pitch) yw(yaw)
    [0.0, 0.0, 0.0, 0.0, -1.0, 0.0],  # T1
    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],  # T2
    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # T3
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # T4

    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # T5
    [0.0, -1.0, 0.0, 0.0, 0.0, 0.0],  # T6
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # T7
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]   # T8
])


def mix(mixing_matrix, control_vector):
    """
    Map a 6-DOF control vector to per-thruster inputs.

    If any input exceeds 1.0 in magnitude, all inputs are scaled down by the
    largest magnitude so the ratios between thrusters are kept.
    """
    motor_inputs = np.dot(mixing_matrix, control_vector)
    max_val = np.max(np.abs(motor_inputs))
    if max_val > 1.0:
        motor_inputs = motor_inputs / max_val
    return motor_inputs


def input_to_pwm(input_val,
                 thruster_data_thrust=THRUSTER_DATA_THRUST,
                 thruster_data_pwm=THRUSTER_DATA_PWM,
                 max_forward_thrust=MAX_FORWARD_THRUST,
                 max_reverse_thrust=MAX_REVERSE_THRUST,
                 pwm_min=PWM_MIN,
                 pwm_max=PWM_MAX):
    """Convert one normalized thruster input [-1, 1] to an integer PWM value."""
    # Scale normalized input [-1.0, 1.0] to target thrust in kgf
    if input_val >= 0:
        target_thrust = input_val * max_forward_thrust
    else:
        target_thrust = input_val * max_reverse_thrust

    # Interpolate to find the required PWM for the target thrust
    pwm = int(np.interp(target_thrust, thruster_data_thrust, thruster_data_pwm))

    # Clamp to safe range just in case
    return max(pwm_min, min(pwm_max, pwm))
