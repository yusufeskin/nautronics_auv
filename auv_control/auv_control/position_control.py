"""Position controller logic for ArduSub in ALT_HOLD. No ROS imports.

Frames: odom is the EKF world frame (z up), the body frame is FLU.

Per axis, matching what each RC channel means in ALT_HOLD:

  x, y (surge, sway)  the channels are thrust fractions, so the loop is a
                      cascade: position error -> desired velocity (P, capped)
                      -> velocity PID closed on the DVL-measured velocity.
  z    (heave)        the channel is a climb-rate demand and ArduSub closes
  yaw                 the rate loop itself, so we only send error x gain,
                      capped. A PID here would close the same loop twice.

When every error is inside its deadband the demand is exactly zero, the
sticks centre and ArduSub holds depth and heading on its own.
"""
import math
from dataclasses import dataclass


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def clamp(value, limit):
    return max(-limit, min(limit, value))


@dataclass
class ControllerParams:
    pos_kp: float = 0.5           # [1/s]  position error -> desired speed
    max_speed: float = 0.3        # [m/s]  cap on the horizontal speed demand
    vel_kff: float = 0.45         # [1/(m/s)] desired speed -> thrust fraction
    vel_kp: float = 0.2           # [1/(m/s)] keep vel_kp * (m/s per unit stick) < 1
    vel_ki: float = 0.1           # [1/m]
    vel_i_limit: float = 0.5      # [m]    integrator state limit
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


@dataclass
class Demand:
    surge: float = 0.0
    sway: float = 0.0
    heave: float = 0.0
    yaw: float = 0.0


class VelocityPid:
    """PI on velocity error plus feedforward, with conditional anti-windup."""

    def __init__(self, kp, ki, kff, i_limit):
        self.kp, self.ki, self.kff, self.i_limit = kp, ki, kff, i_limit
        self.integral = 0.0

    def reset(self):
        self.integral = 0.0

    def step(self, desired, measured, dt, limit):
        error = desired - measured
        unsaturated = self.kff * desired + self.kp * error + self.ki * self.integral
        output = clamp(unsaturated, limit)
        # Integrate only while that does not push further into saturation.
        if output == unsaturated or error * unsaturated < 0.0:
            self.integral = clamp(self.integral + error * dt, self.i_limit)
        return output


class PositionController:
    def __init__(self, params: ControllerParams):
        self.p = params
        self.pid_x = VelocityPid(params.vel_kp, params.vel_ki, params.vel_kff, params.vel_i_limit)
        self.pid_y = VelocityPid(params.vel_kp, params.vel_ki, params.vel_kff, params.vel_i_limit)

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
            d.surge = self.pid_x.step(vbx, s.vx, dt, p.max_thrust)
            d.sway = self.pid_y.step(vby, s.vy, dt, p.max_thrust)
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
