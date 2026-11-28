"""Straight-line reference trajectory with a trapezoidal speed profile. No ROS.

The vehicle is fully actuated and the pool is open water, so guidance is
trajectory generation, not path search: go along the straight 3D line from
start to goal, accelerate at a_max up to the cruise speed, cruise, decelerate
at a_max to a stop exactly on the goal. Yaw turns from the start heading to
the goal heading over the same time, the short way round.

The cruise speed is lowered when needed so that
  * the vertical component stays under max_vertical_speed, and
  * the trajectory lasts at least as long as the yaw turn needs at
    max_yaw_rate - translation and rotation then finish together.

Obstacle avoidance, if it is ever needed, replaces this class behind the same
sample() interface.
"""
import math
from dataclasses import dataclass

EPS = 1e-6


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


@dataclass
class Limits:
    max_speed: float = 0.25          # [m/s]   along the line
    max_accel: float = 0.1           # [m/s^2]
    max_vertical_speed: float = 0.15  # [m/s]
    max_yaw_rate: float = math.radians(15.0)  # [rad/s]


@dataclass
class Sample:
    x: float
    y: float
    z: float
    yaw: float
    vx: float
    vy: float
    vz: float
    yaw_rate: float


class StraightLineTrajectory:
    def __init__(self, start, start_yaw, goal, goal_yaw, limits: Limits):
        """start, goal: (x, y, z) in odom."""
        self.p0 = tuple(start)
        self.yaw0 = start_yaw
        self.dyaw = wrap(goal_yaw - start_yaw)
        d = [g - s for g, s in zip(goal, start)]
        self.length = math.sqrt(sum(c * c for c in d))
        self.dir = tuple(c / self.length for c in d) if self.length > EPS else (0.0, 0.0, 0.0)
        a = limits.max_accel

        # Cruise speed, capped so the vertical component respects its limit.
        v = limits.max_speed
        if abs(d[2]) > EPS:
            v = min(v, limits.max_vertical_speed * self.length / abs(d[2]))

        t_yaw = abs(self.dyaw) / limits.max_yaw_rate
        if self.length > EPS:
            v, t_acc, t_cruise = self._profile(self.length, v, a)
            t_lin = 2.0 * t_acc + t_cruise
            if t_yaw > t_lin:
                # Slow the line down so it ends together with the yaw turn:
                # T = L/v + v/a  ->  v^2 - aT v + aL = 0, smaller root.
                T = t_yaw
                v = (a * T - math.sqrt(a * a * T * T - 4.0 * a * self.length)) / 2.0
                t_acc = v / a
                t_cruise = T - 2.0 * t_acc
        else:
            v, t_acc, t_cruise = 0.0, 0.0, 0.0
        self.a, self.v, self.t_acc, self.t_cruise = a, v, t_acc, t_cruise
        self.duration = max(2.0 * t_acc + t_cruise, t_yaw)

    @staticmethod
    def _profile(length, v, a):
        """Return (cruise speed, accel time, cruise time); triangular if too short."""
        if v * v / a > length:
            v = math.sqrt(a * length)
            return v, v / a, 0.0
        return v, v / a, (length - v * v / a) / v

    def _along(self, t):
        """Distance along the line and speed at time t."""
        a, v, ta, tc = self.a, self.v, self.t_acc, self.t_cruise
        t_end = 2.0 * ta + tc
        if self.length <= EPS or t >= t_end:
            return self.length, 0.0
        if t < ta:
            return 0.5 * a * t * t, a * t
        if t < ta + tc:
            return 0.5 * a * ta * ta + v * (t - ta), v
        rem = t_end - t
        return self.length - 0.5 * a * rem * rem, a * rem

    def sample(self, t) -> Sample:
        t = min(max(t, 0.0), self.duration)
        s, speed = self._along(t)
        x, y, z = (p + c * s for p, c in zip(self.p0, self.dir))
        vx, vy, vz = (c * speed for c in self.dir)

        # Yaw progresses with the distance covered, so it turns smoothly with
        # the speed profile; a pure rotation progresses linearly in time.
        if self.length > EPS:
            frac, rate = s / self.length, self.dyaw * speed / self.length
        elif self.duration > EPS:
            frac, rate = t / self.duration, self.dyaw / self.duration
        else:
            frac, rate = 1.0, 0.0
        if t >= self.duration:
            frac, rate = 1.0, 0.0
        return Sample(x, y, z, wrap(self.yaw0 + self.dyaw * frac), vx, vy, vz, rate)
