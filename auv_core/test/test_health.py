"""Health tracker state machine tests.

The tracker decides whether the estimator may use a sensor, so every
transition it can make is pinned down here.
"""

import pytest

from auv_core.health import HealthConfig, HealthTracker, Status


def make_tracker(**overrides):
    config = HealthConfig(
        sensor="dvl",
        expected_rate_hz=8.0,
        stale_timeout_s=0.5,
        window_s=4.0,
        min_rate_fraction=0.6,
        degraded_reject_ratio=0.25,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return HealthTracker(config)


def feed(tracker, start, count, rate_hz, accepted=True, reason=""):
    """Push `count` samples at `rate_hz` starting at `start`; return next time."""
    period = 1.0 / rate_hz
    t = start
    for _ in range(count):
        if accepted:
            tracker.record_accept(t, reason)
        else:
            tracker.record_reject(t, reason or "test rejection")
        t += period
    return t


def test_starts_with_no_data_and_is_not_fusable():
    tracker = make_tracker()
    report = tracker.evaluate(0.0)
    assert report.status == Status.NO_DATA
    assert not report.fusable


def test_healthy_stream_reports_ok_and_measured_rate():
    tracker = make_tracker()
    end = feed(tracker, 0.0, 32, 8.0)
    report = tracker.evaluate(end)
    assert report.status == Status.OK
    assert report.fusable
    assert report.reason == ""
    assert report.measured_rate_hz == pytest.approx(8.0, rel=0.1)


def test_goes_stale_once_samples_stop():
    tracker = make_tracker()
    end = feed(tracker, 0.0, 32, 8.0)
    report = tracker.evaluate(end + 1.0)
    assert report.status == Status.STALE
    assert not report.fusable
    assert "no sample" in report.reason


def test_stale_boundary_is_not_crossed_early():
    tracker = make_tracker()
    tracker.record_accept(10.0)
    # Just inside the timeout the sensor is still usable...
    assert tracker.evaluate(10.0 + 0.49).status != Status.STALE
    # ...and just outside it is not.
    assert tracker.evaluate(10.0 + 0.51).status == Status.STALE


def test_latest_rejection_makes_the_sensor_unfusable():
    tracker = make_tracker()
    end = feed(tracker, 0.0, 32, 8.0)
    tracker.record_reject(end, "speed above physical limit")
    report = tracker.evaluate(end)
    assert report.status == Status.REJECTED
    assert not report.fusable
    assert report.reason == "speed above physical limit"


def test_recovers_to_ok_after_a_single_rejection():
    tracker = make_tracker()
    end = feed(tracker, 0.0, 32, 8.0)
    tracker.record_reject(end, "one bad sample")
    end += 0.125
    end = feed(tracker, end, 4, 8.0)
    report = tracker.evaluate(end)
    assert report.status == Status.OK
    assert report.fusable


def test_low_rate_is_degraded_but_still_fusable():
    tracker = make_tracker()
    # 3 Hz against an 8 Hz expectation is below the 0.6 floor (4.8 Hz).
    end = feed(tracker, 0.0, 12, 3.0)
    report = tracker.evaluate(end)
    assert report.status == Status.DEGRADED
    assert report.fusable
    assert "below" in report.reason


def test_high_rejection_share_is_degraded_but_still_fusable():
    tracker = make_tracker()
    t = 0.0
    # Alternate accept/reject so the window ends on an accepted sample but
    # carries a 50% reject ratio.
    for _ in range(16):
        tracker.record_reject(t, "beam dropout")
        t += 0.125
        tracker.record_accept(t)
        t += 0.125
    report = tracker.evaluate(t)
    assert report.status == Status.DEGRADED
    assert report.fusable
    assert "rejected in window" in report.reason


def test_explicit_degraded_reason_survives_a_healthy_stream():
    tracker = make_tracker()
    end = feed(tracker, 0.0, 32, 8.0, reason="3 of 4 beams locked")
    report = tracker.evaluate(end)
    assert report.status == Status.DEGRADED
    assert report.fusable
    assert report.reason == "3 of 4 beams locked"


def test_reject_ratio_uses_the_window_not_the_whole_run():
    """A sensor that failed early and recovered must read healthy now."""
    tracker = make_tracker()
    t = feed(tracker, 0.0, 40, 8.0, accepted=False)
    t = feed(tracker, t, 40, 8.0)
    report = tracker.evaluate(t)
    assert report.reject_ratio == pytest.approx(0.0)
    assert report.status == Status.OK
    # Lifetime counters still remember the bad patch.
    assert report.rejected_count == 40
    assert report.accepted_count == 40


def test_counters_are_cumulative():
    tracker = make_tracker()
    t = feed(tracker, 0.0, 10, 8.0)
    feed(tracker, t, 5, 8.0, accepted=False)
    report = tracker.evaluate(t + 5 / 8.0)
    assert report.accepted_count == 10
    assert report.rejected_count == 5
