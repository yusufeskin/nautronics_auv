"""IMU sample validation and gyro bias estimation, free of ROS dependencies.

Two jobs live here. The first is ordinary range checking. The second matters
more: whenever no absolute heading source is aiding the estimate - because
none is fitted, or because heading_gate has judged the fitted one to be
magnetically disturbed - yaw is a pure integral of this gyro. The turn-on
bias is then the single largest error source in the whole navigation stack.
A bias of 0.5 deg/s left uncorrected is 7.5 degrees of heading error over a
15 minute run.

The bias also matters when a heading source *is* working, because
heading_gate cross-checks that source against this gyro. A biased gyro makes
a healthy compass look like it is disagreeing.

The fix is cheap and operational: hold the vehicle still on the surface
before the run and average the gyro. This module implements that, and reports
whether it has actually converged so the mission layer can refuse to start on
an uncalibrated gyro.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["GyroBiasEstimator", "ImuLimits", "ImuSample", "ImuValidator", "ImuVerdict"]

STANDARD_GRAVITY = 9.80665


@dataclass
class ImuLimits:
    # Full-scale ranges. A sample at the rail is not a measurement, it is a
    # clipped one, and fusing it drags the estimate toward the rail.
    max_gyro_rad_s: float = 8.0        # ~460 deg/s
    max_accel_mps2: float = 6.0 * STANDARD_GRAVITY

    max_clock_skew_s: float = 0.2

    # Variances substituted when the driver reports none. Replace with the
    # values from your IMU's datasheet once it is chosen; these are
    # deliberately pessimistic placeholders.
    default_gyro_variance: float = 4.0e-6       # (0.002 rad/s)^2
    default_accel_variance: float = 4.0e-4      # (0.02 m/s^2)^2


@dataclass
class ImuSample:
    stamp: float
    angular_velocity: np.ndarray
    linear_acceleration: np.ndarray


@dataclass
class ImuVerdict:
    accepted: bool
    reason: str = ""
    degraded_reason: str = ""
    angular_velocity: np.ndarray | None = None   # bias-corrected when available
    linear_acceleration: np.ndarray | None = None


@dataclass
class GyroBiasEstimator:
    """Averages the gyro while the vehicle is demonstrably still.

    Stillness is judged from the IMU alone - low rotation rate and a specific
    force close to gravity - so this works on the surface before the DVL has
    bottom lock and before anything else in the stack is running.
    """

    still_gyro_threshold_rad_s: float = 0.02      # ~1.1 deg/s
    still_accel_tolerance_mps2: float = 0.5
    settle_time_s: float = 20.0
    min_samples: int = 200

    _still_since: float | None = None
    _sum: np.ndarray = field(default_factory=lambda: np.zeros(3))
    _count: int = 0
    _bias: np.ndarray = field(default_factory=lambda: np.zeros(3))
    _ready: bool = False

    @property
    def bias(self) -> np.ndarray:
        """Best estimate so far. Zero until the first convergence."""
        return self._bias.copy()

    @property
    def ready(self) -> bool:
        """True once a full settle window has been averaged at least once."""
        return self._ready

    def still_duration(self, now: float) -> float:
        if self._still_since is None:
            return 0.0
        return now - self._still_since

    def update(self, sample: ImuSample) -> None:
        gyro = np.asarray(sample.angular_velocity, dtype=float)
        accel = np.asarray(sample.linear_acceleration, dtype=float)

        rotating = float(np.linalg.norm(gyro)) > self.still_gyro_threshold_rad_s
        accelerating = (
            abs(float(np.linalg.norm(accel)) - STANDARD_GRAVITY)
            > self.still_accel_tolerance_mps2
        )

        if rotating or accelerating:
            # Motion invalidates the current accumulation but never a bias
            # that already converged - throwing away a good calibration
            # because someone bumped the vehicle would be worse than keeping
            # a slightly stale one.
            self._still_since = None
            self._sum[:] = 0.0
            self._count = 0
            return

        if self._still_since is None:
            self._still_since = sample.stamp

        self._sum += gyro
        self._count += 1

        settled = (sample.stamp - self._still_since) >= self.settle_time_s
        if settled and self._count >= self.min_samples:
            self._bias = self._sum / self._count
            self._ready = True


@dataclass
class ImuValidator:
    limits: ImuLimits = field(default_factory=ImuLimits)
    bias_estimator: GyroBiasEstimator = field(default_factory=GyroBiasEstimator)
    subtract_bias: bool = True

    def check(self, sample: ImuSample, now: float) -> ImuVerdict:
        gyro = np.asarray(sample.angular_velocity, dtype=float)
        accel = np.asarray(sample.linear_acceleration, dtype=float)

        skew = sample.stamp - now
        if abs(skew) > self.limits.max_clock_skew_s:
            direction = "ahead of" if skew > 0 else "behind"
            return ImuVerdict(False, f"timestamp {abs(skew):.3f} s {direction} local clock")

        if not np.all(np.isfinite(gyro)) or not np.all(np.isfinite(accel)):
            return ImuVerdict(False, "non-finite sample")

        gyro_norm = float(np.linalg.norm(gyro))
        if gyro_norm > self.limits.max_gyro_rad_s:
            return ImuVerdict(
                False,
                f"gyro {gyro_norm:.1f} rad/s at or beyond full scale "
                f"({self.limits.max_gyro_rad_s:.1f})",
            )

        accel_norm = float(np.linalg.norm(accel))
        if accel_norm > self.limits.max_accel_mps2:
            return ImuVerdict(
                False,
                f"accelerometer {accel_norm:.1f} m/s^2 at or beyond full scale "
                f"({self.limits.max_accel_mps2:.1f})",
            )

        self.bias_estimator.update(sample)

        corrected = gyro - self.bias_estimator.bias if self.subtract_bias else gyro

        degraded = ""
        if self.subtract_bias and not self.bias_estimator.ready:
            degraded = (
                "gyro bias not yet estimated - hold the vehicle still on the "
                "surface before the run"
            )

        return ImuVerdict(
            accepted=True,
            degraded_reason=degraded,
            angular_velocity=corrected,
            linear_acceleration=accel,
        )
