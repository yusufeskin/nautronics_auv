"""Navigation status bookkeeping, free of ROS dependencies.

``robot_localization`` publishes a state estimate but says nothing about how
much that estimate is worth. It will happily keep integrating after the DVL
has stopped reporting, producing an odometry message that looks exactly like
one backed by real velocity measurements. Whoever is deciding whether to
attempt a manoeuvre needs to be able to tell those apart.

This tracker watches the Layer 1 health topics and the filter's own
covariance and turns them into one answer: is the estimate currently aided,
coasting, or not to be trusted - and if coasting, for how long.

It reports; it does not decide. "Dead reckoning for 8 seconds" is a fact.
Whether that is too long to keep going is a mission-layer policy, and putting
it here would bury a tactical decision inside a status node.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

__all__ = ["NavMode", "NavStatusConfig", "NavStatusReport", "NavStatusTracker"]


class NavMode:
    """Mirrors the constants in auv_core_msgs/NavStatus."""

    INIT = 0
    AIDED = 1
    DEAD_RECKONING = 2
    FAULT = 3


@dataclass
class NavStatusConfig:
    # No odometry from the filter within this long means the filter itself has
    # stopped, which is a fault rather than a degradation.
    odom_timeout_s: float = 1.0

    # A health message older than this is treated as absent. Gates publish at
    # 2 Hz regardless of whether samples are arriving, so silence here means
    # the gate node is gone, not that the sensor is quiet.
    health_timeout_s: float = 2.0

    # Sensors whose loss makes the whole estimate untrustworthy rather than
    # merely degraded. The IMU drives the prediction step: without it there is
    # no attitude and nothing to propagate.
    critical_sensors: tuple = ("imu",)


@dataclass
class NavStatusReport:
    mode: int
    dvl_fused: bool
    depth_fused: bool
    heading_fused: bool
    dead_reckoning_time_s: float
    position_sigma_xy_m: float
    position_sigma_z_m: float
    yaw_sigma_rad: float
    gyro_bias: tuple
    # Human-readable explanation of a non-AIDED mode, for logs and the GUI.
    reason: str = ""


@dataclass
class NavStatusTracker:
    config: NavStatusConfig = field(default_factory=NavStatusConfig)

    _health: dict = field(default_factory=dict)       # sensor -> (time, fusable)
    _last_odom_time: float | None = None
    _pose_variances: tuple = (0.0, 0.0, 0.0)          # xx, yy, yaw
    _z_variance: float = 0.0
    _gyro_bias: tuple = (0.0, 0.0, 0.0)
    _aided_since_lost: float | None = None            # when DVL aiding stopped

    def update_health(self, sensor: str, fusable: bool, now: float) -> None:
        self._health[sensor] = (now, bool(fusable))

    def update_gyro_bias(self, bias) -> None:
        self._gyro_bias = (float(bias[0]), float(bias[1]), float(bias[2]))

    def update_odometry(self, pose_covariance, now: float) -> None:
        """Feed the 6x6 row-major pose covariance from /odom/filtered."""
        cov = list(pose_covariance)
        if len(cov) != 36:
            return
        self._last_odom_time = now
        self._pose_variances = (cov[0], cov[7], cov[35])   # xx, yy, yaw
        self._z_variance = cov[14]

    def evaluate(self, now: float) -> NavStatusReport:
        dvl = self._fusable("dvl", now)
        depth = self._fusable("depth", now)
        heading = self._fusable("heading", now)

        # Track how long velocity aiding has been unavailable. Reset on the
        # transition back rather than on every aided sample, so the number is
        # the length of the current gap and not of the last one.
        if dvl:
            self._aided_since_lost = None
        elif self._aided_since_lost is None:
            self._aided_since_lost = now
        dead_reckoning = 0.0 if dvl else (now - self._aided_since_lost)

        mode, reason = self._mode(now, dvl)

        return NavStatusReport(
            mode=mode,
            dvl_fused=dvl,
            depth_fused=depth,
            heading_fused=heading,
            dead_reckoning_time_s=dead_reckoning,
            # The larger of the two horizontal sigmas, not their average: this
            # number is read to decide whether a manoeuvre is safe, and the
            # worst axis is what decides that.
            position_sigma_xy_m=math.sqrt(max(self._pose_variances[0],
                                              self._pose_variances[1])),
            position_sigma_z_m=math.sqrt(max(0.0, self._z_variance)),
            yaw_sigma_rad=math.sqrt(max(0.0, self._pose_variances[2])),
            gyro_bias=self._gyro_bias,
            reason=reason,
        )

    # -- internals ---------------------------------------------------------

    def _fusable(self, sensor: str, now: float) -> bool:
        entry = self._health.get(sensor)
        if entry is None:
            return False
        stamp, fusable = entry
        if now - stamp > self.config.health_timeout_s:
            return False
        return fusable

    def _mode(self, now: float, dvl: bool) -> tuple:
        if self._last_odom_time is None:
            return NavMode.INIT, "no odometry from the filter yet"

        age = now - self._last_odom_time
        if age > self.config.odom_timeout_s:
            return NavMode.FAULT, f"filter silent for {age:.1f} s"

        missing = [s for s in self.config.critical_sensors if not self._fusable(s, now)]
        if missing:
            return NavMode.FAULT, f"critical sensor unavailable: {', '.join(missing)}"

        if not dvl:
            return NavMode.DEAD_RECKONING, "no velocity aiding; coasting on IMU"

        return NavMode.AIDED, ""
