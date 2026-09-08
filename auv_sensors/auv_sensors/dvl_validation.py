"""DVL sample validation, with no ROS dependency so it can be unit tested.

Everything here is a decision about a single sample: is this velocity
physically possible, was the bottom actually tracked, and how much should the
estimator trust it. Nothing here integrates, remembers a pose, or talks to an
autopilot.

The checks are ordered cheapest-first and each one names itself in the
rejection reason, because "DVL rejected" in a log at 03:00 the night before a
run is worth nothing without the reason attached.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from auv_core import frames

__all__ = ["DvlLimits", "DvlSample", "DvlValidator", "Verdict"]

# marine_acoustic_msgs/Dvl uses -1 for "this covariance is unknown".
UNKNOWN_COVAR = -1.0

# marine_acoustic_msgs/Dvl velocity_mode constants, repeated so this module
# stays importable without the message package.
MODE_BOTTOM = 1
MODE_WATER = 2


@dataclass
class DvlLimits:
    # Bottom lock needs at least this many beams. Three is enough for a 3D
    # solution but geometrically weaker, so it is accepted as degraded.
    min_good_beams: int = 3
    beams_for_full_confidence: int = 4

    # Faster than the vehicle can physically swim. Anything above this is an
    # acoustic artefact, not motion.
    max_speed_mps: float = 2.0

    # A velocity step larger than this between samples implies an
    # acceleration the thrusters cannot produce.
    max_accel_mps2: float = 5.0

    # Gap beyond which two samples are too far apart to judge acceleration
    # between them; the jump check is skipped rather than guessed at.
    max_dt_s: float = 1.0

    # Consecutive invalid samples before bottom lock is declared lost, and
    # consecutive valid ones before it is declared regained. Hysteresis stops
    # a single bad ping from tearing the estimator's velocity aiding away.
    lock_loss_streak: int = 4
    lock_regain_streak: int = 2

    # Substituted when the driver reports an unknown covariance.
    default_velocity_variance: float = 0.0025  # (0.05 m/s)^2

    # Covariance multiplier applied when fewer than
    # `beams_for_full_confidence` beams are locked.
    degraded_variance_scale: float = 4.0

    # Water-track velocity is relative to the water column, not the ground.
    # Fusing it as ground velocity is only harmless where there is no
    # current, so it is off by default and degraded when enabled.
    allow_water_track: bool = False

    # A sample stamped further than this from the local clock means the DVL's
    # clock and the companion's have drifted apart. The A50 syncs over NTP
    # and this is the first thing to break on a fresh Jetson image.
    max_clock_skew_s: float = 0.5


@dataclass
class DvlSample:
    stamp: float
    velocity: np.ndarray          # in the DVL's own frame, convention per validator
    num_good_beams: int
    beams_valid: bool
    velocity_mode: int = MODE_BOTTOM
    covariance: np.ndarray | None = None   # 3x3 or flat 9, None when unreported
    altitude: float = -1.0


@dataclass
class Verdict:
    accepted: bool
    locked: bool
    reason: str = ""
    degraded_reason: str = ""
    velocity_flu: np.ndarray | None = None
    covariance_flu: np.ndarray | None = None


@dataclass
class DvlValidator:
    limits: DvlLimits = field(default_factory=DvlLimits)

    # True when the incoming velocity is FRD (the A50's native convention and
    # what the sim bridge emits). Set False only if a driver already
    # publishes FLU.
    input_is_frd: bool = True

    _locked: bool = False
    _invalid_streak: int = 0
    _valid_streak: int = 0
    _last_good_time: float | None = None
    _last_good_velocity: np.ndarray | None = None

    @property
    def locked(self) -> bool:
        return self._locked

    def reset(self) -> None:
        """Forget motion history without touching the lock state.

        Called when the sample stream is interrupted, so the next sample is
        not compared against a velocity from before the gap.
        """
        self._last_good_time = None
        self._last_good_velocity = None

    def check(self, sample: DvlSample, now: float) -> Verdict:
        """Validate one sample. ``now`` is the local clock, for skew checks."""
        velocity = np.asarray(sample.velocity, dtype=float)

        # 1. Timestamps. A stamp in the future is impossible and means the
        #    two clocks disagree; a stamp far in the past means the sample
        #    queued up somewhere.
        skew = sample.stamp - now
        if abs(skew) > self.limits.max_clock_skew_s:
            direction = "ahead of" if skew > 0 else "behind"
            return self._invalid(
                f"timestamp {abs(skew):.2f} s {direction} local clock "
                f"(check NTP sync between DVL and companion)"
            )

        # 2. Bottom lock. Water-track reports the speed of the water, not of
        #    the ground, so it cannot substitute for bottom track silently.
        water_track_reason = ""
        if sample.velocity_mode == MODE_WATER:
            if not self.limits.allow_water_track:
                return self._invalid("water-track velocity, not bottom lock")
            water_track_reason = "water-track velocity (relative to water, not ground)"

        if not sample.beams_valid or sample.num_good_beams < self.limits.min_good_beams:
            return self._invalid(
                f"{sample.num_good_beams} good beams, need {self.limits.min_good_beams}"
            )

        # 3. Finiteness, before any arithmetic touches the values.
        if not np.all(np.isfinite(velocity)):
            return self._invalid("non-finite velocity")

        # 4. Physical speed limit.
        speed = float(np.linalg.norm(velocity))
        if speed > self.limits.max_speed_mps:
            return self._invalid(
                f"speed {speed:.2f} m/s above the {self.limits.max_speed_mps:.2f} m/s limit"
            )

        # 5. Acceleration jump, measured against the last *accepted* sample so
        #    that one outlier cannot poison the comparison for the next one.
        #    A gap longer than max_dt_s is not evidence of anything, so the
        #    check is skipped rather than guessed at.
        if self._last_good_time is not None and self._last_good_velocity is not None:
            dt = sample.stamp - self._last_good_time
            if 0.0 < dt <= self.limits.max_dt_s:
                accel = float(np.linalg.norm(velocity - self._last_good_velocity)) / dt
                if accel > self.limits.max_accel_mps2:
                    return self._invalid(
                        f"implausible acceleration {accel:.1f} m/s^2 over {dt * 1e3:.0f} ms"
                    )

        # Accepted.
        covariance = self._resolve_covariance(sample)
        if self.input_is_frd:
            velocity_out = frames.flip_body_convention(velocity)
            covariance = frames.rotate_covariance(frames.FLU_FRD, covariance)
        else:
            velocity_out = velocity

        self._last_good_time = sample.stamp
        self._last_good_velocity = velocity.copy()

        self._invalid_streak = 0
        self._valid_streak += 1
        if not self._locked and self._valid_streak >= self.limits.lock_regain_streak:
            self._locked = True

        degraded = water_track_reason
        if sample.num_good_beams < self.limits.beams_for_full_confidence:
            beam_note = (
                f"{sample.num_good_beams} of {self.limits.beams_for_full_confidence} beams locked"
            )
            degraded = f"{degraded}; {beam_note}" if degraded else beam_note

        return Verdict(
            accepted=True,
            locked=self._locked,
            degraded_reason=degraded,
            velocity_flu=velocity_out,
            covariance_flu=covariance,
        )

    # -- internals ---------------------------------------------------------

    def _invalid(self, reason: str) -> Verdict:
        self._valid_streak = 0
        self._invalid_streak += 1
        if self._locked and self._invalid_streak >= self.limits.lock_loss_streak:
            self._locked = False
            # Motion history is meaningless across a lock loss.
            self.reset()
        return Verdict(accepted=False, locked=self._locked, reason=reason)

    def _resolve_covariance(self, sample: DvlSample) -> np.ndarray:
        cov = None
        if sample.covariance is not None:
            candidate = np.asarray(sample.covariance, dtype=float).reshape(3, 3)
            diagonal = np.diag(candidate)
            usable = (
                np.all(np.isfinite(candidate))
                and np.all(diagonal >= 0.0)
                and np.any(diagonal > 0.0)
            )
            if usable:
                cov = candidate

        if cov is None:
            cov = np.eye(3) * self.limits.default_velocity_variance

        if sample.num_good_beams < self.limits.beams_for_full_confidence:
            cov = cov * self.limits.degraded_variance_scale

        return cov


def altitude_is_valid(altitude: float) -> bool:
    """A negative or non-finite altitude is the driver's 'no bottom' sentinel."""
    return math.isfinite(altitude) and altitude > 0.0
