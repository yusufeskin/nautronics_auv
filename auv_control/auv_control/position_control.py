"""Position controller logic for ArduSub in ALT_HOLD. No ROS imports.

Frames: odom is the EKF world frame (z up), the body frame is FLU.

Per axis, matching what each RC channel means in ALT_HOLD:

  x, y (surge, sway)  the channels are thrust fractions, so the loop is a
                      cascade: position error -> desired velocity (P, capped)
                      -> velocity PI closed on the DVL-measured velocity.
                      The PI works in m/s; a measured stick -> steady speed
                      curve (SpeedCurve) turns its output into a stick value.
                      That inverse undoes the plant's sqrt-like shape, so the
                      loop behaves the same at 0.05 and at 0.5 m/s.
                      The curve only knows steady speeds, so the reference
                      acceleration is added in stick units (vel_ka): extra
                      push to speed up, less push (or reverse) to slow down.
  z    (heave)        the channel is a climb-rate demand and ArduSub closes
  yaw                 the rate loop itself, so we only send error x gain,
                      capped. A PID here would close the same loop twice.

When every error is inside its deadband the demand is exactly zero, the
sticks centre and ArduSub holds depth and heading on its own.
"""
import bisect
import math
from dataclasses import dataclass, field

# Steady speed in ALT_HOLD vs stick fraction, measured in sim with
# stick_calibration (2026-11-28, bags/stick_calibration*.csv). Symmetric
# within 0.003 m/s, so one curve per axis serves both directions.
# Re-measure on the real vehicle.
SIM_STICK = [0.0, 0.0025, 0.005, 0.0075, 0.0125, 0.02, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4]
SIM_SURGE_SPEED = [0.0, 0.048, 0.075, 0.098, 0.136, 0.181, 0.23, 0.308, 0.451, 0.562, 0.655,
                   0.813, 0.944]
SIM_SWAY_SPEED = [0.0, 0.047, 0.077, 0.101, 0.139, 0.186, 0.237, 0.316, 0.465, 0.578, 0.674,
                  0.834, 0.97]


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def clamp(value, limit):
    return max(-limit, min(limit, value))


class SpeedCurve:
    """Measured stick -> steady speed, used backwards: speed -> stick.

    Piecewise linear between the points, odd-symmetric, linear extrapolation
    past the last point (the output is clamped by max_thrust anyway).
    """

    def __init__(self, stick, speed):
        if len(stick) != len(speed) or len(stick) < 2:
            raise ValueError('stick and speed need the same length, at least 2')
        if stick[0] != 0.0 or speed[0] != 0.0:
            raise ValueError('curve must start at (0, 0)')
        if any(b <= a for a, b in zip(stick, stick[1:])) or \
                any(b <= a for a, b in zip(speed, speed[1:])):
            raise ValueError('stick and speed must be strictly increasing')
        self.stick, self.speed = list(stick), list(speed)

    def stick_for(self, v):
        a = abs(v)
        i = min(max(bisect.bisect_right(self.speed, a) - 1, 0), len(self.speed) - 2)
        s0, s1, v0, v1 = self.stick[i], self.stick[i + 1], self.speed[i], self.speed[i + 1]
        u = s0 + (s1 - s0) * (a - v0) / (v1 - v0)
        return math.copysign(u, v) if v else 0.0


@dataclass
class ControllerParams:
    pos_kp: float = 0.5           # [1/s]  position error -> desired speed
    max_speed: float = 0.3        # [m/s]  cap on the horizontal speed demand
    vel_kp: float = 0.5           # [-]    velocity error -> extra speed command
    vel_ki: float = 0.5           # [1/s]
    vel_i_limit: float = 0.3      # [m]    integrator state; ki * limit = max correction [m/s]
    vel_ka: float = 0.09          # [stick/(m/s^2)] acceleration feedforward
    max_thrust: float = 0.4       # surge/sway output limit
    yaw_kp: float = 1.0           # [1/rad]
    max_yaw: float = 0.5
    yaw_deadband: float = math.radians(1.0)
    depth_kp: float = 1.0         # [1/m]
    max_heave: float = 0.5
    depth_deadband: float = 0.03  # [m]
    pos_tolerance: float = 0.15   # [m]    "at setpoint" thresholds
    depth_tolerance: float = 0.10  # [m]
    yaw_tolerance: float = math.radians(5.0)
    surge_curve_stick: list = field(default_factory=lambda: list(SIM_STICK))
    surge_curve_speed: list = field(default_factory=lambda: list(SIM_SURGE_SPEED))
    sway_curve_stick: list = field(default_factory=lambda: list(SIM_STICK))
    sway_curve_speed: list = field(default_factory=lambda: list(SIM_SWAY_SPEED))


@dataclass
class State:
    x: float
    y: float
    z: float
    yaw: float
    vx: float  # body-frame forward velocity [m/s]
    vy: float  # body-frame left velocity [m/s]


@dataclass
class Reference:
    x: float
    y: float
    z: float
    yaw: float
    vx_ff: float = 0.0  # feedforward velocity in odom [m/s]
    vy_ff: float = 0.0
    ax_ff: float = 0.0  # feedforward acceleration in odom [m/s^2]
    ay_ff: float = 0.0


@dataclass
class Demand:
    surge: float = 0.0
    sway: float = 0.0
    heave: float = 0.0
    yaw: float = 0.0


class VelocityPid:
    """Velocity PI in m/s, then the inverse speed curve to a stick value.

    speed command = desired + kp * error + ki * integral   (desired = feedforward)
    stick         = curve.stick_for(speed command) + ka * accel, clamped to the limit
    Conditional anti-windup: integrate only while not pushing further into
    the clamp.
    """

    def __init__(self, kp, ki, i_limit, curve, ka=0.0):
        self.kp, self.ki, self.i_limit, self.curve, self.ka = kp, ki, i_limit, curve, ka
        self.integral = 0.0

    def reset(self):
        self.integral = 0.0

    def step(self, desired, measured, dt, limit, accel=0.0):
        error = desired - measured
        command = desired + self.kp * error + self.ki * self.integral
        unsaturated = self.curve.stick_for(command) + self.ka * accel
        output = clamp(unsaturated, limit)
        if output == unsaturated or error * unsaturated < 0.0:
            self.integral = clamp(self.integral + error * dt, self.i_limit)
        return output


class PositionController:
    def __init__(self, params: ControllerParams):
        self.p = params
        self.pid_x = VelocityPid(params.vel_kp, params.vel_ki, params.vel_i_limit,
                                 SpeedCurve(params.surge_curve_stick, params.surge_curve_speed),
                                 params.vel_ka)
        self.pid_y = VelocityPid(params.vel_kp, params.vel_ki, params.vel_i_limit,
                                 SpeedCurve(params.sway_curve_stick, params.sway_curve_speed),
                                 params.vel_ka)

    def reset(self):
        self.pid_x.reset()
        self.pid_y.reset()

    def compute(self, s: State, r: Reference, dt: float, velocity_ok: bool = True) -> Demand:
        p = self.p
        d = Demand()

        # Horizontal: position P -> desired velocity in odom, capped.
        ex, ey = r.x - s.x, r.y - s.y
        vdx = r.vx_ff + p.pos_kp * ex
        vdy = r.vy_ff + p.pos_kp * ey
        speed = math.hypot(vdx, vdy)
        if speed > p.max_speed:
            vdx, vdy = vdx * p.max_speed / speed, vdy * p.max_speed / speed

        if velocity_ok:
            c, sn = math.cos(s.yaw), math.sin(s.yaw)
            vbx = c * vdx + sn * vdy        # odom -> body
            vby = -sn * vdx + c * vdy
            abx = c * r.ax_ff + sn * r.ay_ff
            aby = -sn * r.ax_ff + c * r.ay_ff
            d.surge = self.pid_x.step(vbx, s.vx, dt, p.max_thrust, abx)
            d.sway = self.pid_y.step(vby, s.vy, dt, p.max_thrust, aby)
        else:
            # No trustworthy velocity: stop pushing horizontally rather than
            # drive on a stale estimate. Depth and heading keep working.
            self.reset()

        e_yaw = wrap(r.yaw - s.yaw)
        if abs(e_yaw) > p.yaw_deadband:
            d.yaw = clamp(p.yaw_kp * e_yaw, p.max_yaw)

        e_z = r.z - s.z
        if abs(e_z) > p.depth_deadband:
            d.heave = clamp(p.depth_kp * e_z, p.max_heave)

        return d

    def at_reference(self, s: State, r: Reference) -> bool:
        p = self.p
        return (math.hypot(r.x - s.x, r.y - s.y) < p.pos_tolerance
                and abs(r.z - s.z) < p.depth_tolerance
                and abs(wrap(r.yaw - s.yaw)) < p.yaw_tolerance)
