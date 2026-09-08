"""Absolute heading validation, free of ROS dependencies.

Yaw is the only unbounded error in this vehicle's state estimate: depth is
measured absolutely, roll and pitch are pinned by gravity, velocity comes
straight from the DVL, and yaw comes from integrating a gyro with nothing to
correct it. An absolute heading source - a compass, an AHRS, a gyrocompass -
is what closes that hole, so this gate exists to make one usable.

The problem is that the cheapest such sources are magnetic, and a competition
pool is a steel box. A disturbed compass does not get noisier, it gets
*biased*, and a bias fed to a filter with an honest-looking covariance is
worse than no measurement at all. So this gate does not simply forward
headings. It cross-checks every one against the gyro, which cannot be
magnetically disturbed, and refuses the heading while the two disagree.

That check is the whole point of the file. It needs no extra hardware, it
catches exactly the failure mode that ruins magnetic headings indoors, and it
degrades to "no absolute heading, dead reckon on the gyro" - which is where
the vehicle would have been anyway.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from auv_core import frames

__all__ = [
    "HeadingLimits",
    "HeadingSample",
    "HeadingValidator",
    "HeadingVerdict",
]


@dataclass
class HeadingLimits:
    max_clock_skew_s: float = 0.2

    # Beyond this tilt an absolute heading is not trustworthy and, past 90
    # degrees of pitch, not even well defined: yaw and roll degenerate onto
    # the same axis. Magnetic sources also lose their tilt compensation long
    # before that. This vehicle deliberately rolls and pitches 90 degrees for
    # RoboSub gate style points, so it *will* spend time outside this band -
    # the correct behaviour there is to stop using the heading, not to
    # forward a meaningless one.
    max_tilt_rad: float = math.radians(60.0)

    # --- gyro cross-check --------------------------------------------------
    # How far the heading source's own rate of change may differ from the
    # gyro's before the sample is called disturbed. TUNE: swing the vehicle
    # through 360 degrees in the pool, log both, and set this above the
    # honest disagreement (source latency, gyro noise) and below the
    # disagreement seen next to a steel wall.
    max_yaw_rate_mismatch_rad_s: float = math.radians(8.0)

    # Two heading samples further apart than this are not compared, because
    # over a long interval the gyro's own bias makes the comparison
    # meaningless rather than the heading wrong.
    max_compare_dt_s: float = 1.0
    max_gyro_dt_s: float = 0.2

    # Hysteresis. Disturbance is declared quickly and cleared slowly: a
    # compass that has just walked past a steel ladder should not be trusted
    # again the instant it happens to agree.
    disturbed_streak: int = 3
    recover_streak: int = 30

    # --- optional magnetic field check -------------------------------------
    # Only used when the source also publishes the raw field vector. The total
    # field strength is roughly constant at a given place; a large deviation
    # means something ferrous or electrical is nearby.
    # MEASURE: log the field magnitude in open air at the venue and use that,
    # not a textbook value. Set to 0 to disable the check.
    expected_field_magnitude_t: float = 0.0
    field_magnitude_tolerance: float = 0.25   # fraction of the expected value

    # --- uncertainty -------------------------------------------------------
    # TUNE: from the source's datasheet heading accuracy, squared. The default
    # is (5 degrees)^2, which is a deliberately pessimistic placeholder.
    default_yaw_variance: float = math.radians(5.0) ** 2

    # Applied when the sample is usable but suspect - field magnitude slightly
    # off, or the gyro cross-check only just passing.
    degraded_variance_scale: float = 9.0


@dataclass
class HeadingSample:
    stamp: float
    # Vehicle attitude in ENU/FLU as (x, y, z, w), already corrected for the
    # device's mount rotation by the caller.
    orientation: np.ndarray
    # Raw magnetic field vector in tesla, when the source provides one.
    magnetic_field: np.ndarray | None = None


@dataclass
class HeadingVerdict:
    accepted: bool
    reason: str = ""
    degraded_reason: str = ""
    yaw_rad: float = 0.0
    yaw_variance: float = 0.0
    disturbed: bool = False


@dataclass
class HeadingValidator:
    limits: HeadingLimits = field(default_factory=HeadingLimits)

    _disturbed: bool = False
    _bad_streak: int = 0
    _good_streak: int = 0

    _gyro_integral: float = 0.0
    _last_gyro_stamp: float | None = None
    _tilt: tuple = (0.0, 0.0)          # roll, pitch from the last heading sample

    _last_yaw: float | None = None
    _last_yaw_stamp: float | None = None
    _integral_at_last_yaw: float = 0.0

    @property
    def disturbed(self) -> bool:
        return self._disturbed

    def update_gyro(self, stamp: float, angular_velocity) -> None:
        """Feed a body-frame angular velocity sample, for the cross-check.

        Called from the gyro stream (``imu_gate/data``), which runs far faster
        than the heading source. The body yaw rate is converted to the rate of
        change of world yaw using the tilt from the most recent heading
        sample; over the sub-second intervals this is compared across, holding
        tilt constant is well inside the noise.
        """
        omega = np.asarray(angular_velocity, dtype=float)
        if not np.all(np.isfinite(omega)):
            return

        if self._last_gyro_stamp is not None:
            dt = stamp - self._last_gyro_stamp
            if 0.0 < dt <= self.limits.max_gyro_dt_s:
                roll, pitch = self._tilt
                cos_pitch = math.cos(pitch)
                # Near +/-90 degrees of pitch this blows up, which is the same
                # place the heading itself stops meaning anything. The tilt
                # check rejects those samples; guard here so an out-of-band
                # gyro sample cannot poison the integral in the meantime.
                if abs(cos_pitch) > 1e-3:
                    yaw_rate = (
                        omega[1] * math.sin(roll) + omega[2] * math.cos(roll)
                    ) / cos_pitch
                    self._gyro_integral += yaw_rate * dt

        self._last_gyro_stamp = stamp

    def check(self, sample: HeadingSample, now: float) -> HeadingVerdict:
        skew = sample.stamp - now
        if abs(skew) > self.limits.max_clock_skew_s:
            direction = "ahead of" if skew > 0 else "behind"
            return self._reject(f"timestamp {abs(skew):.3f} s {direction} local clock")

        quat = np.asarray(sample.orientation, dtype=float)
        if not np.all(np.isfinite(quat)) or float(np.linalg.norm(quat)) < 1e-6:
            return self._reject("non-finite or zero orientation")

        roll, pitch, yaw = frames.euler_rpy_from_quat(quat)
        self._tilt = (roll, pitch)

        tilt = max(abs(roll), abs(pitch))
        if tilt > self.limits.max_tilt_rad:
            # Not a fault - the vehicle is simply at an attitude where an
            # absolute heading means nothing. Drop the comparison history so
            # the sample after the manoeuvre is not compared across it.
            self._forget_comparison()
            return self._reject(
                f"tilt {math.degrees(tilt):.0f} deg beyond the "
                f"{math.degrees(self.limits.max_tilt_rad):.0f} deg limit; "
                "heading is not meaningful at this attitude"
            )

        degraded = self._check_field(sample)
        if degraded is None:
            return self._reject("magnetic field magnitude implausible")

        mismatch = self._gyro_mismatch(sample.stamp, yaw)
        self._remember(sample.stamp, yaw)

        if mismatch is not None and mismatch > self.limits.max_yaw_rate_mismatch_rad_s:
            self._good_streak = 0
            self._bad_streak += 1
            if self._bad_streak >= self.limits.disturbed_streak:
                self._disturbed = True
            return self._reject(
                f"disagrees with gyro by {math.degrees(mismatch):.1f} deg/s "
                "(magnetic disturbance?)"
            )

        self._bad_streak = 0
        self._good_streak += 1
        if self._disturbed:
            if self._good_streak < self.limits.recover_streak:
                return self._reject(
                    f"recovering from disturbance "
                    f"({self._good_streak}/{self.limits.recover_streak} good samples)"
                )
            self._disturbed = False

        variance = self.limits.default_yaw_variance
        if degraded:
            variance *= self.limits.degraded_variance_scale

        return HeadingVerdict(
            accepted=True,
            degraded_reason=degraded,
            yaw_rad=yaw,
            yaw_variance=variance,
            disturbed=False,
        )

    # -- internals ---------------------------------------------------------

    def _check_field(self, sample: HeadingSample) -> str | None:
        """Returns a degraded reason, "" when clean, or None to reject."""
        expected = self.limits.expected_field_magnitude_t
        if expected <= 0.0 or sample.magnetic_field is None:
            return ""

        field = np.asarray(sample.magnetic_field, dtype=float)
        if not np.all(np.isfinite(field)):
            return None

        magnitude = float(np.linalg.norm(field))
        deviation = abs(magnitude - expected) / expected
        if deviation > 2.0 * self.limits.field_magnitude_tolerance:
            return None
        if deviation > self.limits.field_magnitude_tolerance:
            return f"field magnitude {100.0 * deviation:.0f}% off expected"
        return ""

    def _gyro_mismatch(self, stamp: float, yaw: float) -> float | None:
        """Rate of disagreement with the gyro, or None if not comparable."""
        if self._last_yaw is None or self._last_yaw_stamp is None:
            return None
        if self._last_gyro_stamp is None:
            return None

        dt = stamp - self._last_yaw_stamp
        if not (0.0 < dt <= self.limits.max_compare_dt_s):
            return None

        measured = frames.wrap_angle(yaw - self._last_yaw)
        expected = self._gyro_integral - self._integral_at_last_yaw
        return abs(frames.wrap_angle(measured - expected)) / dt

    def _remember(self, stamp: float, yaw: float) -> None:
        self._last_yaw = yaw
        self._last_yaw_stamp = stamp
        self._integral_at_last_yaw = self._gyro_integral

    def _forget_comparison(self) -> None:
        self._last_yaw = None
        self._last_yaw_stamp = None

    def _reject(self, reason: str) -> HeadingVerdict:
        return HeadingVerdict(accepted=False, reason=reason, disturbed=self._disturbed)
