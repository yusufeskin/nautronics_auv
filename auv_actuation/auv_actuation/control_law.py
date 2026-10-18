"""Pose controller for ArduSub in ALT_HOLD, free of ROS dependencies.

ArduSub's ALT_HOLD sticks do not all mean the same thing, and this controller
is shaped by that:

    forward, lateral   a thrust fraction       -> velocity PID on the DVL
    throttle           a climb-rate demand     -> depth error x gain, capped
    yaw                a turn-rate demand      -> yaw error x gain, capped

On forward and lateral nobody else closes a loop, so this one does: position
error becomes a desired body velocity, and a PID drives the measured DVL
velocity onto it. The integral lives there because that is where the steady
errors live - buoyancy trim pushing the vehicle along, a thruster that is
slightly stronger one way.

On throttle and yaw ArduSub already closes the rate loop internally. Putting a
PID on those axes here would close the same loop twice and tune neither, so
they are proportional only: an error becomes a rate, the rate shrinks as the
error does, and at zero the stick is centred and ArduSub holds depth and
heading by itself.

Position hold is not a separate mode. It is this controller with a reference
that does not move.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

__all__ = ["Axis", "ControlGains", "ControlOutput", "PoseController", "Reference"]


def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


@dataclass
class Axis:
    """One PID axis with output limiting and clamping anti-windup."""

    kp: float = 0.0
    ki: float = 0.0
    kd: float = 0.0
    # Hard cap on the axis output, in the units the axis produces.
    output_limit: float = 0.0
    # Cap on the integral term's own contribution, so a long stall cannot
    # store enough history to overshoot on the way back.
    integral_limit: float = 0.0

    _integral: float = 0.0
    _last_error: float | None = None

    def reset(self) -> None:
        self._integral = 0.0
        self._last_error = None

    def step(self, error: float, dt: float) -> float:
        if dt <= 0.0:
            return 0.0

        derivative = 0.0
        if self._last_error is not None:
            derivative = (error - self._last_error) / dt
        self._last_error = error

        previous_integral = self._integral
        if self.ki != 0.0:
            self._integral += error * dt
            if self.integral_limit > 0.0:
                cap = self.integral_limit / abs(self.ki)
                self._integral = _clamp(self._integral, cap)

        output = self.kp * error + self.ki * self._integral + self.kd * derivative

        if self.output_limit > 0.0 and abs(output) > self.output_limit:
            # Clamping anti-windup: while the output is pinned against its
            # limit, integrating further in the same direction stores error
            # the vehicle never receives and has to be unwound later.
            if error * output > 0.0:
                self._integral = previous_integral
                output = self.kp * error + self.ki * self._integral + self.kd * derivative
            output = _clamp(output, self.output_limit)
        return output


@dataclass
class ControlGains:
    # Horizontal outer loop: position error to desired speed, and the speed
    # cap that makes an approach sane rather than a lunge.
    kp_position: float = 0.8          # (m/s) per m
    max_speed_xy: float = 0.5         # m/s

    # Depth: error straight to a climb-rate demand for ArduSub.
    kp_depth: float = 0.8             # (m/s) per m
    max_climb_rate: float = 0.3       # m/s

    # Yaw: error straight to a turn-rate demand for ArduSub.
    kp_yaw: float = 1.0               # (rad/s) per rad
    max_yaw_rate: float = 0.5         # rad/s

    # Horizontal inner loop: velocity error to thrust fraction in [-1, 1].
    surge: Axis = field(default_factory=lambda: Axis(output_limit=1.0, integral_limit=0.3))
    sway: Axis = field(default_factory=lambda: Axis(output_limit=1.0, integral_limit=0.3))

    # Errors below these count as arrived.
    position_tolerance_m: float = 0.1
    depth_tolerance_m: float = 0.1
    yaw_tolerance_rad: float = math.radians(5.0)


@dataclass
class Reference:
    """Where to be now, in odom, plus how fast the reference itself is moving.

    Any of x, y, z, yaw may be None to leave that axis free: no position
    correction on it, only the feedforward. A free depth or heading with zero
    feedforward centres the stick, and ArduSub holds what it has.

    The feedforward is in the body frame (FLU), the way guidance produces it
    for a moving reference. For a fixed target or a hold it is all zero.
    """

    x: float | None = None
    y: float | None = None
    z: float | None = None
    yaw: float | None = None
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    yaw_rate: float = 0.0


@dataclass
class ControlOutput:
    # What goes to the output stage.
    surge: float                 # thrust fraction, forward positive
    sway: float                  # thrust fraction, left positive (FLU)
    climb_rate: float            # m/s, up positive
    yaw_rate: float              # rad/s, counter-clockwise positive (FLU)

    # For logging and tuning.
    desired_velocity: np.ndarray  # body frame vx, vy the inner loop chased
    position_error: np.ndarray    # body frame x, y; odom z
    yaw_error: float
    at_target: bool


@dataclass
class PoseController:
    gains: ControlGains = field(default_factory=ControlGains)

    def reset(self) -> None:
        """Clear integrator state. Call on enable and on a new target."""
        self.gains.surge.reset()
        self.gains.sway.reset()

    def step(self, reference: Reference, position, yaw: float, measured_velocity,
             dt: float, horizontal_ok: bool = True) -> ControlOutput:
        """One control period.

        ``position`` is the vehicle in odom, ``yaw`` its heading in odom,
        ``measured_velocity`` its body-frame (vx, vy) from the estimator.

        ``horizontal_ok`` false means the velocity measurement cannot be
        trusted - the DVL has been out too long. Forward and lateral are then
        centred rather than driven on a stale velocity, and depth and heading
        carry on, because those do not depend on the DVL.
        """
        position = np.asarray(position, dtype=float)
        measured = np.asarray(measured_velocity, dtype=float)[:2]
        feedforward = np.asarray(reference.velocity, dtype=float)

        # -- horizontal ---------------------------------------------------
        error_odom = np.zeros(2)
        if reference.x is not None:
            error_odom[0] = reference.x - position[0]
        if reference.y is not None:
            error_odom[1] = reference.y - position[1]

        # Only yaw is used to rotate into the body frame: ArduSub holds roll
        # and pitch near level, and the full rotation would couple a
        # transient lean into the horizontal command.
        cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
        error_body = np.array([
            cos_yaw * error_odom[0] + sin_yaw * error_odom[1],
            -sin_yaw * error_odom[0] + cos_yaw * error_odom[1],
        ])

        desired = self.gains.kp_position * error_body + feedforward[:2]
        speed = float(np.linalg.norm(desired))
        if speed > self.gains.max_speed_xy:
            desired = desired * (self.gains.max_speed_xy / speed)

        if horizontal_ok:
            surge = self.gains.surge.step(desired[0] - measured[0], dt)
            sway = self.gains.sway.step(desired[1] - measured[1], dt)
        else:
            self.reset()
            surge = 0.0
            sway = 0.0

        # -- depth --------------------------------------------------------
        depth_error = 0.0
        if reference.z is not None:
            depth_error = reference.z - position[2]
        climb_rate = _clamp(
            self.gains.kp_depth * depth_error + float(feedforward[2]),
            self.gains.max_climb_rate,
        )

        # -- yaw ----------------------------------------------------------
        yaw_error = 0.0
        if reference.yaw is not None:
            yaw_error = _wrap(reference.yaw - yaw)
        yaw_rate = _clamp(
            self.gains.kp_yaw * yaw_error + reference.yaw_rate,
            self.gains.max_yaw_rate,
        )

        at_target = (
            float(np.linalg.norm(error_body)) < self.gains.position_tolerance_m
            and abs(depth_error) < self.gains.depth_tolerance_m
            and abs(yaw_error) < self.gains.yaw_tolerance_rad
        )

        return ControlOutput(
            surge=surge,
            sway=sway,
            climb_rate=climb_rate,
            yaw_rate=yaw_rate,
            desired_velocity=desired,
            position_error=np.array([error_body[0], error_body[1], depth_error]),
            yaw_error=yaw_error,
            at_target=at_target,
        )
