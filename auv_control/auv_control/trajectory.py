"""Straight-line reference trajectory with a trapezoidal speed profile. No ROS.

The vehicle is fully actuated and the pool is open water, so guidance is
trajectory generation, not path search: go along the straight 3D line from
start to goal, change speed at a_max from the start speed to the cruise speed,
cruise, decelerate at a_max to a stop exactly on the goal. Yaw turns from the
start heading to the goal heading over the same time, the short way round.

The start speed is the vehicle's current velocity along the new line, so a
goal that replaces another mid-move does not ask a moving vehicle to be at
rest. Velocity across the line is not modelled; the controller removes it.

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
    ax: float = 0.0
    ay: float = 0.0
    az: float = 0.0


@dataclass
class Profile:
    """Speed along the line: v0 -> vp in t1, cruise vp for tc, vp -> 0 in t3."""
    v0: float
    vp: float
    a1: float   # signed acceleration of the first phase
    t1: float
    tc: float
    t3: float

    @property
    def duration(self):
        return self.t1 + self.tc + self.t3


def make_profile(length, v0, v_cruise, a):
    """Shortest profile from speed v0 to rest over length, capped at v_cruise.

    v0 must already allow stopping in length: v0^2 / (2a) <= length.
    """
    if v0 <= v_cruise:
        d_needed = (v_cruise ** 2 - v0 ** 2) / (2 * a) + v_cruise ** 2 / (2 * a)
        vp = v_cruise if d_needed <= length else math.sqrt(a * length + v0 ** 2 / 2)
    else:
        vp = v_cruise    # already faster than allowed: slow down first
    d1 = abs(vp ** 2 - v0 ** 2) / (2 * a)
    d3 = vp ** 2 / (2 * a)
    tc = max(length - d1 - d3, 0.0) / vp if vp > EPS else 0.0
    return Profile(v0, vp, a if vp >= v0 else -a, abs(vp - v0) / a, tc, vp / a)


class StraightLineTrajectory:
    def __init__(self, start, start_yaw, goal, goal_yaw, limits: Limits,
                 start_velocity=(0.0, 0.0, 0.0)):
        """start, goal: (x, y, z) in odom. start_velocity: (vx, vy, vz) in odom."""
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
            # Speed already along the new line; never more than lets us stop on the goal.
            v0 = sum(c * w for c, w in zip(self.dir, start_velocity))
            v0 = min(max(v0, 0.0), math.sqrt(2 * a * self.length))
            prof = make_profile(self.length, v0, v, a)
            if t_yaw > prof.duration:
                # Slow the cruise so the line ends together with the yaw turn.
                # Duration falls monotonically with the cruise speed: bisect.
                lo, hi = 1e-4, v
                for _ in range(60):
                    mid = 0.5 * (lo + hi)
                    if make_profile(self.length, v0, mid, a).duration > t_yaw:
                        lo = mid
                    else:
                        hi = mid
                prof = make_profile(self.length, v0, hi, a)
        else:
            prof = Profile(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self.a, self.profile = a, prof
        self.v, self.t_acc, self.t_cruise = prof.vp, prof.t1, prof.tc
        self.duration = max(prof.duration, t_yaw)

    def _along(self, t):
        """Distance along the line, speed and acceleration at time t."""
        p, a = self.profile, self.a
        if self.length <= EPS or t >= p.duration:
            return self.length, 0.0, 0.0
        if t < p.t1:
            return p.v0 * t + 0.5 * p.a1 * t * t, p.v0 + p.a1 * t, p.a1
        s1 = p.v0 * p.t1 + 0.5 * p.a1 * p.t1 ** 2
        if t < p.t1 + p.tc:
            return s1 + p.vp * (t - p.t1), p.vp, 0.0
        rem = p.duration - t
        return self.length - 0.5 * a * rem * rem, a * rem, -a

    def sample(self, t) -> Sample:
        t = min(max(t, 0.0), self.duration)
        s, speed, acc = self._along(t)
        x, y, z = (p + c * s for p, c in zip(self.p0, self.dir))
        vx, vy, vz = (c * speed for c in self.dir)
        ax, ay, az = (c * acc for c in self.dir)

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
        return Sample(x, y, z, wrap(self.yaw0 + self.dyaw * frac), vx, vy, vz, rate, ax, ay, az)
