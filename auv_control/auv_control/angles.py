"""Pure angle / attitude-control helpers shared by the control nodes (no ROS imports)."""

import math


def normalize_angle(angle):
    """Wrap an angle in radians into [-pi, pi] (both ends are kept as-is)."""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def p_turn_rate(error, kp, max_speed, tolerance_rad, min_speed):
    """
    Proportional angular-rate command for a wrapped angle error (radians).

    The command is saturated to +/-max_speed. While the error is larger than
    3 * tolerance_rad, the magnitude is raised to at least min_speed.
    """
    calculated_speed = error * kp

    if abs(calculated_speed) > max_speed:
        calculated_speed = math.copysign(max_speed, calculated_speed)
    if abs(error) > tolerance_rad * 3 and abs(calculated_speed) < min_speed:
        calculated_speed = math.copysign(min_speed, calculated_speed)
    return calculated_speed


def accumulate_turn(total_turned, prev_angle, current_angle):
    """Add the wrapped step between two angle samples to a running total."""
    return total_turned + normalize_angle(current_angle - prev_angle)


def loop_velocities(duration, radius):
    """
    Constant (linear, angular) velocities for one full circle in `duration` s.

    A negative radius turns the other way; the linear speed is always >= 0.
    """
    w_const = (2.0 * math.pi) / duration
    if radius < 0:
        w_const = -w_const
        radius = -radius
    v_const = radius * abs(w_const)
    return v_const, w_const


def clamp(value, low, high):
    """Clamp value into [low, high]."""
    return max(low, min(high, value))


def downward_thrust_body(roll, base_thrust):
    """
    Body-frame (linear.y, linear.z) that pushes along world -Z at a given roll.

    roll is in radians.
    """
    return -base_thrust * math.sin(roll), -base_thrust * math.cos(roll)
