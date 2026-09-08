"""Cascaded pose controller, free of ROS dependencies.

Two nested loops, which is the arrangement having a DVL earns you:

    pose error  ──▶ desired body velocity ──▶ velocity error ──▶ wrench
      (outer, P)         (saturated)            (inner, PID)

The outer loop is deliberately proportional only. Its job is to produce a
*sensible speed to approach at*, and it saturates: far from the target, ask
for the maximum safe speed; close in, ask for proportionally less. Integral
action there would fight the saturation and wind up during every long transit.

The inner loop is where the integral belongs, because that is where the
steady-state errors live - buoyancy trim, a current, a thruster that pushes
slightly harder one way. It closes on the DVL's *measured* velocity, which is
the whole point: the vehicle is corrected against what it is actually doing,
not against a model of what it should be doing.

Yaw works exactly the same way and is not special. Heading error produces a
yaw rate, that rate is tracked, and when the error reaches zero the command
stops. The autopilot underneath then simply holds whatever attitude it is at.
Its own idea of heading is never read here - only its behaviour, which is that
it stays put when left alone. That is why the number in the autopilot does not
have to agree with the number in this stack.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

__all__ = ["Axis", "CascadeGains", "ControlOutput", "PoseController", "PoseTarget"]


def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


@dataclass
class Axis:
    """One PID axis with output limiting and conditional integration."""

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

    def step(self, error: float, dt: float, freeze_integral: bool = False) -> float:
        if dt <= 0.0:
            return 0.0

        derivative = 0.0
        if self._last_error is not None:
            derivative = (error - self._last_error) / dt
        self._last_error = error

        # Conditional integration: while the actuator is saturated the extra
        # integral cannot reach the vehicle, so accumulating it only stores
        # error that has to be unwound later.
        if not freeze_integral and self.ki != 0.0:
            self._integral += error * dt
            if self.integral_limit > 0.0:
                cap = self.integral_limit / self.ki
                self._integral = max(-cap, min(cap, self._integral))

        output = self.kp * error + self.ki * self._integral + self.kd * derivative
        if self.output_limit > 0.0:
            output = max(-self.output_limit, min(self.output_limit, output))
        return output


@dataclass
class CascadeGains:
    # Outer loop: proportional gain from position error to desired speed, and
    # the speed cap that makes the approach sane rather than a lunge.
    kp_position: float = 0.8          # (m/s) per m
    kp_yaw: float = 1.2               # (rad/s) per rad
    max_speed_xy: float = 0.5         # m/s
    max_speed_z: float = 0.3          # m/s
    max_yaw_rate: float = 0.6         # rad/s

    # Inner loop: one PID per commanded axis.
    surge: Axis = field(default_factory=Axis)
    sway: Axis = field(default_factory=Axis)
    heave: Axis = field(default_factory=Axis)
    yaw_rate: Axis = field(default_factory=Axis)

    # Errors below these count as arrived, so the controller stops nudging.
    position_tolerance_m: float = 0.1
    yaw_tolerance_rad: float = math.radians(5.0)


@dataclass
class PoseTarget:
    """Where to be, in the odom frame. Any field may be None to leave it free."""

    x: float | None = None
    y: float | None = None
    z: float | None = None
    yaw: float | None = None


@dataclass
class ControlOutput:
    force: np.ndarray            # body frame FLU, newtons
    torque: np.ndarray           # body frame FLU, newton-metres
    desired_velocity: np.ndarray  # what the outer loop asked for, body frame
    position_error: np.ndarray   # body frame, for logging and tuning
    yaw_error: float
    at_target: bool


@dataclass
class PoseController:
    gains: CascadeGains = field(default_factory=CascadeGains)

    def reset(self) -> None:
        """Clear integrator state. Call on enable and on a new target."""
        for axis in (self.gains.surge, self.gains.sway,
                     self.gains.heave, self.gains.yaw_rate):
            axis.reset()

    # -- outer loop --------------------------------------------------------

    def desired_velocity(self, target: PoseTarget, position, yaw: float):
        """Position error in odom -> desired body velocity, saturated.

        Returns the desired body-frame velocity, the body-frame position
        error, and the yaw error, so a caller can log what the outer loop
        actually saw.
        """
        position = np.asarray(position, dtype=float)

        error_odom = np.zeros(3)
        for index, value in enumerate((target.x, target.y, target.z)):
            if value is not None:
                error_odom[index] = value - position[index]

        # Rotate the horizontal error into the body frame. Only yaw matters:
        # roll and pitch are held near level by the autopilot, and using the
        # full rotation would couple a transient lean into the horizontal
        # command.
        cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
        error_body = np.array([
            cos_yaw * error_odom[0] + sin_yaw * error_odom[1],
            -sin_yaw * error_odom[0] + cos_yaw * error_odom[1],
            error_odom[2],
        ])

        horizontal = self.gains.kp_position * error_body[:2]
        speed = float(np.linalg.norm(horizontal))
        if speed > self.gains.max_speed_xy and speed > 0.0:
            horizontal = horizontal * (self.gains.max_speed_xy / speed)

        vertical = self.gains.kp_position * error_body[2]
        vertical = max(-self.gains.max_speed_z, min(self.gains.max_speed_z, vertical))

        yaw_error = 0.0
        yaw_rate = 0.0
        if target.yaw is not None:
            yaw_error = _wrap(target.yaw - yaw)
            yaw_rate = self.gains.kp_yaw * yaw_error
            yaw_rate = max(-self.gains.max_yaw_rate,
                           min(self.gains.max_yaw_rate, yaw_rate))

        desired = np.array([horizontal[0], horizontal[1], vertical, yaw_rate])
        return desired, error_body, yaw_error

    # -- inner loop --------------------------------------------------------

    def velocity_to_wrench(self, desired, measured, dt: float,
                           saturated: bool = False) -> tuple:
        """Desired minus measured body velocity -> body wrench.

        ``saturated`` comes from the hardware boundary and freezes the
        integrators: while the actuator is on its limit, integrating stores
        error the vehicle never received.
        """
        desired = np.asarray(desired, dtype=float)
        measured = np.asarray(measured, dtype=float)
        error = desired - measured

        force = np.array([
            self.gains.surge.step(error[0], dt, saturated),
            self.gains.sway.step(error[1], dt, saturated),
            self.gains.heave.step(error[2], dt, saturated),
        ])
        # Roll and pitch torque are deliberately zero: the autopilot holds
        # those axes and accepts a lean angle rather than a torque, so asking
        # for one would be asking for something that cannot be delivered.
        torque = np.array([
            0.0,
            0.0,
            self.gains.yaw_rate.step(error[3], dt, saturated),
        ])
        return force, torque

    # -- both together -----------------------------------------------------

    def step(self, target: PoseTarget, position, yaw: float, measured_velocity,
             dt: float, saturated: bool = False) -> ControlOutput:
        desired, error_body, yaw_error = self.desired_velocity(target, position, yaw)

        at_target = (
            float(np.linalg.norm(error_body)) < self.gains.position_tolerance_m
            and abs(yaw_error) < self.gains.yaw_tolerance_rad
        )

        force, torque = self.velocity_to_wrench(desired, measured_velocity, dt, saturated)

        return ControlOutput(
            force=force,
            torque=torque,
            desired_velocity=desired,
            position_error=error_body,
            yaw_error=yaw_error,
            at_target=at_target,
        )
