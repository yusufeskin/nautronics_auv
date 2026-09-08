"""Sensor health bookkeeping, with no ROS dependency so it can be unit tested.

A gate's job is to decide, sample by sample, whether a measurement may reach
the estimator. This class holds the part of that decision that is identical
for every sensor - rate, staleness, and how often samples are being thrown
away - leaving each gate to implement only its own physics checks.

Both RoboSub teams whose reports informed this design lost pool time to
sensors that were degrading silently. The point of publishing this at a fixed
rate, even when nothing is arriving, is that a dead sensor and a dead node
look different in the logs.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

__all__ = ["HealthConfig", "HealthTracker", "Status"]


class Status:
    """Mirrors the constants in auv_core_msgs/SensorHealth."""

    OK = 0
    DEGRADED = 1
    REJECTED = 2
    STALE = 3
    NO_DATA = 4


@dataclass
class HealthConfig:
    sensor: str
    expected_rate_hz: float

    # No sample within this long means the sensor is stale and unfusable.
    # Set it from the sensor's own period, not from a round number: a DVL at
    # 8 Hz and an IMU at 200 Hz have very different ideas of "late".
    stale_timeout_s: float

    # Trailing window used for the rate estimate and the reject ratio. A
    # run-long ratio hides a sensor that only just started failing.
    window_s: float = 5.0

    # Below this fraction of the expected rate the sensor is degraded: it is
    # still fusable, but the estimator should widen its covariance.
    min_rate_fraction: float = 0.6

    # Above this share of rejected samples in the window the sensor is
    # degraded even if the most recent sample passed.
    degraded_reject_ratio: float = 0.25


@dataclass
class HealthReport:
    status: int
    fusable: bool
    reason: str
    measured_rate_hz: float
    accepted_count: int
    rejected_count: int
    reject_ratio: float


@dataclass
class HealthTracker:
    config: HealthConfig

    accepted_count: int = 0
    rejected_count: int = 0
    _events: deque = field(default_factory=deque)  # (time, accepted: bool)
    _last_sample_time: float | None = None
    _last_accepted: bool = False
    _last_reason: str = ""

    def record_accept(self, now: float, degraded_reason: str = "") -> None:
        """A sample passed every check.

        ``degraded_reason`` marks a sample that is usable but of reduced
        quality - a DVL holding bottom lock on three beams instead of four,
        say. It keeps the sensor fusable while telling the estimator to trust
        it less.
        """
        self.accepted_count += 1
        self._events.append((now, True))
        self._last_sample_time = now
        self._last_accepted = True
        self._last_reason = degraded_reason
        self._trim(now)

    def record_reject(self, now: float, reason: str) -> None:
        """A sample failed a check and must not reach the estimator."""
        self.rejected_count += 1
        self._events.append((now, False))
        self._last_sample_time = now
        self._last_accepted = False
        self._last_reason = reason
        self._trim(now)

    def evaluate(self, now: float) -> HealthReport:
        self._trim(now)

        window_total = len(self._events)
        window_rejected = sum(1 for _, ok in self._events if not ok)
        reject_ratio = (window_rejected / window_total) if window_total else 0.0
        measured_rate = window_total / self.config.window_s if self.config.window_s > 0 else 0.0

        if self._last_sample_time is None:
            return self._report(Status.NO_DATA, False, "no sample received yet",
                                measured_rate, reject_ratio)

        age = now - self._last_sample_time
        if age > self.config.stale_timeout_s:
            return self._report(Status.STALE, False, f"no sample for {age:.2f} s",
                                measured_rate, reject_ratio)

        if not self._last_accepted:
            return self._report(Status.REJECTED, False, self._last_reason,
                                measured_rate, reject_ratio)

        rate_floor = self.config.min_rate_fraction * self.config.expected_rate_hz
        if measured_rate < rate_floor:
            return self._report(
                Status.DEGRADED, True,
                f"rate {measured_rate:.1f} Hz below {rate_floor:.1f} Hz floor",
                measured_rate, reject_ratio,
            )

        if reject_ratio > self.config.degraded_reject_ratio:
            return self._report(
                Status.DEGRADED, True,
                f"{100.0 * reject_ratio:.0f}% of samples rejected in window",
                measured_rate, reject_ratio,
            )

        if self._last_reason:
            return self._report(Status.DEGRADED, True, self._last_reason,
                                measured_rate, reject_ratio)

        return self._report(Status.OK, True, "", measured_rate, reject_ratio)

    def _report(self, status, fusable, reason, rate, ratio) -> HealthReport:
        return HealthReport(
            status=status,
            fusable=fusable,
            reason=reason,
            measured_rate_hz=rate,
            accepted_count=self.accepted_count,
            rejected_count=self.rejected_count,
            reject_ratio=ratio,
        )

    def _trim(self, now: float) -> None:
        cutoff = now - self.config.window_s
        while self._events and self._events[0][0] < cutoff:
            self._events.popleft()
