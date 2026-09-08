"""Heading gate logic tests.

The gyro cross-check is the reason this gate exists, so most of these tests
are about it: a heading that disagrees with the gyro is being magnetically
disturbed, and a disturbed heading is biased rather than noisy - which makes
it worse than no heading at all.
"""

import math

import numpy as np
import pytest

from auv_core import frames
from auv_sensors.heading_validation import (
    HeadingLimits,
    HeadingSample,
    HeadingValidator,
)


def attitude(yaw_deg=0.0, roll_deg=0.0, pitch_deg=0.0):
    return frames.quat_from_euler_rpy(
        math.radians(roll_deg), math.radians(pitch_deg), math.radians(yaw_deg)
    )


def sample(t=0.0, yaw_deg=0.0, roll_deg=0.0, pitch_deg=0.0, field=None):
    return HeadingSample(
        stamp=t,
        orientation=attitude(yaw_deg, roll_deg, pitch_deg),
        magnetic_field=field,
    )


def make_validator(**overrides):
    return HeadingValidator(limits=HeadingLimits(**overrides))


def turn(validator, start_t, duration_s, source_rate_deg_s, gyro_rate_deg_s,
         gyro_hz=100.0, heading_hz=20.0, start_yaw_deg=0.0):
    """Rotate the vehicle, feeding the gyro and the heading source separately.

    Passing different rates simulates a heading source that has stopped
    tracking reality, which is what a magnetic disturbance looks like.
    Returns the list of verdicts from the heading samples.
    """
    verdicts = []
    gyro_dt = 1.0 / gyro_hz
    heading_period = 1.0 / heading_hz
    next_heading = start_t
    t = start_t
    end = start_t + duration_s

    while t <= end:
        validator.update_gyro(t, (0.0, 0.0, math.radians(gyro_rate_deg_s)))
        if t >= next_heading:
            yaw = start_yaw_deg + source_rate_deg_s * (t - start_t)
            verdicts.append(validator.check(sample(t=t, yaw_deg=yaw), now=t))
            next_heading += heading_period
        t += gyro_dt
    return verdicts


# --------------------------------------------------------------------------
# Happy path
# --------------------------------------------------------------------------


def test_steady_heading_is_accepted():
    validator = make_validator()
    validator.update_gyro(0.0, (0.0, 0.0, 0.0))
    validator.update_gyro(0.05, (0.0, 0.0, 0.0))
    validator.check(sample(t=0.0, yaw_deg=30.0), now=0.0)
    verdict = validator.check(sample(t=0.05, yaw_deg=30.0), now=0.05)
    assert verdict.accepted
    assert verdict.yaw_rad == pytest.approx(math.radians(30.0), abs=1e-9)


def test_heading_agreeing_with_the_gyro_is_accepted_through_a_turn():
    validator = make_validator()
    verdicts = turn(validator, 0.0, 2.0, source_rate_deg_s=20.0, gyro_rate_deg_s=20.0)
    assert len(verdicts) > 10
    assert all(v.accepted for v in verdicts[1:])
    assert not validator.disturbed


def test_first_sample_cannot_be_cross_checked_but_is_still_accepted():
    validator = make_validator()
    verdict = validator.check(sample(t=0.0, yaw_deg=10.0), now=0.0)
    assert verdict.accepted


# --------------------------------------------------------------------------
# The gyro cross-check
# --------------------------------------------------------------------------


def test_heading_that_disagrees_with_the_gyro_is_rejected():
    """The vehicle is turning at 20 deg/s; the compass insists it is still."""
    validator = make_validator(max_yaw_rate_mismatch_rad_s=math.radians(8.0))
    verdicts = turn(validator, 0.0, 2.0, source_rate_deg_s=0.0, gyro_rate_deg_s=20.0)
    assert not verdicts[-1].accepted
    assert "gyro" in verdicts[-1].reason
    assert validator.disturbed


def test_disturbance_needs_a_streak_not_a_single_sample():
    validator = make_validator(disturbed_streak=3)
    verdicts = turn(validator, 0.0, 0.15, source_rate_deg_s=0.0, gyro_rate_deg_s=40.0)
    # A couple of bad samples must not yet condemn the source.
    assert not validator.disturbed
    assert any(not v.accepted for v in verdicts)


def test_small_disagreement_stays_within_tolerance():
    """Source latency and gyro noise produce honest disagreement."""
    validator = make_validator(max_yaw_rate_mismatch_rad_s=math.radians(8.0))
    verdicts = turn(validator, 0.0, 2.0, source_rate_deg_s=24.0, gyro_rate_deg_s=20.0)
    assert all(v.accepted for v in verdicts[1:])


def test_recovery_from_disturbance_is_slow():
    """A compass that just walked past a steel ladder is not trusted at once."""
    validator = make_validator(disturbed_streak=3, recover_streak=30)
    turn(validator, 0.0, 2.0, source_rate_deg_s=0.0, gyro_rate_deg_s=40.0)
    assert validator.disturbed

    # Now the source agrees again, but only briefly.
    brief = turn(validator, 3.0, 0.5, source_rate_deg_s=0.0, gyro_rate_deg_s=0.0)
    assert not brief[-1].accepted
    assert "recovering" in brief[-1].reason
    assert validator.disturbed

    # Sustained agreement clears it.
    sustained = turn(validator, 4.0, 3.0, source_rate_deg_s=0.0, gyro_rate_deg_s=0.0)
    assert sustained[-1].accepted
    assert not validator.disturbed


def test_long_gaps_are_not_cross_checked():
    """Over a long interval the gyro's own bias makes the comparison useless."""
    validator = make_validator(max_compare_dt_s=1.0)
    validator.update_gyro(0.0, (0.0, 0.0, 0.0))
    validator.check(sample(t=0.0, yaw_deg=0.0), now=0.0)
    validator.update_gyro(10.0, (0.0, 0.0, 0.0))
    verdict = validator.check(sample(t=10.0, yaw_deg=90.0), now=10.0)
    assert verdict.accepted


def test_cross_check_accounts_for_tilt():
    """Body yaw rate is not world yaw rate unless the vehicle is level.

    Rolled 45 degrees, a pure body-z rotation turns the vehicle in world yaw
    more slowly than the raw gyro z axis suggests. Ignoring that would flag a
    healthy source as disturbed every time the vehicle banked.
    """
    validator = make_validator(max_yaw_rate_mismatch_rad_s=math.radians(3.0))
    roll = 45.0
    body_rate = math.radians(20.0)
    world_rate_deg_s = math.degrees(body_rate * math.cos(math.radians(roll)))

    t = 0.0
    verdicts = []
    while t <= 2.0:
        validator.update_gyro(t, (0.0, 0.0, body_rate))
        if abs((t * 20.0) % 1.0) < 1e-9:
            verdicts.append(
                validator.check(
                    sample(t=t, yaw_deg=world_rate_deg_s * t, roll_deg=roll), now=t
                )
            )
        t += 0.01
    assert all(v.accepted for v in verdicts[1:])


# --------------------------------------------------------------------------
# Attitude limits
# --------------------------------------------------------------------------


def test_heading_is_refused_at_high_tilt():
    """Rolling through the gate for style points is not a heading fix."""
    validator = make_validator(max_tilt_rad=math.radians(60.0))
    verdict = validator.check(sample(t=0.0, yaw_deg=0.0, roll_deg=90.0), now=0.0)
    assert not verdict.accepted
    assert "not meaningful at this attitude" in verdict.reason


def test_high_pitch_is_refused_too():
    validator = make_validator(max_tilt_rad=math.radians(60.0))
    verdict = validator.check(sample(t=0.0, pitch_deg=75.0), now=0.0)
    assert not verdict.accepted


def test_comparison_history_is_dropped_across_a_manoeuvre():
    """Yaw before and after a barrel roll must not be differenced."""
    validator = make_validator(max_tilt_rad=math.radians(60.0))
    validator.update_gyro(0.0, (0.0, 0.0, 0.0))
    validator.check(sample(t=0.0, yaw_deg=0.0), now=0.0)

    validator.check(sample(t=0.5, yaw_deg=0.0, roll_deg=90.0), now=0.5)

    validator.update_gyro(1.0, (0.0, 0.0, 0.0))
    verdict = validator.check(sample(t=1.0, yaw_deg=80.0), now=1.0)
    assert verdict.accepted


# --------------------------------------------------------------------------
# Basic checks
# --------------------------------------------------------------------------


def test_clock_skew_is_rejected():
    validator = make_validator(max_clock_skew_s=0.2)
    verdict = validator.check(sample(t=5.0), now=0.0)
    assert not verdict.accepted
    assert "local clock" in verdict.reason


def test_non_finite_orientation_is_rejected():
    validator = make_validator()
    bad = HeadingSample(stamp=0.0, orientation=np.array([float("nan"), 0.0, 0.0, 1.0]))
    verdict = validator.check(bad, now=0.0)
    assert not verdict.accepted
    assert "non-finite" in verdict.reason


def test_zero_orientation_is_rejected():
    validator = make_validator()
    bad = HeadingSample(stamp=0.0, orientation=np.zeros(4))
    verdict = validator.check(bad, now=0.0)
    assert not verdict.accepted


# --------------------------------------------------------------------------
# Magnetic field magnitude check
# --------------------------------------------------------------------------


def test_field_check_is_off_by_default():
    validator = make_validator()
    verdict = validator.check(sample(t=0.0, field=np.array([1.0, 1.0, 1.0])), now=0.0)
    assert verdict.accepted
    assert verdict.degraded_reason == ""


def test_field_slightly_off_is_degraded():
    validator = make_validator(
        expected_field_magnitude_t=48e-6, field_magnitude_tolerance=0.25
    )
    field = np.array([0.0, 0.0, 48e-6 * 1.4])
    verdict = validator.check(sample(t=0.0, field=field), now=0.0)
    assert verdict.accepted
    assert "field magnitude" in verdict.degraded_reason


def test_degraded_sample_carries_a_larger_variance():
    validator = make_validator(
        expected_field_magnitude_t=48e-6,
        field_magnitude_tolerance=0.25,
        degraded_variance_scale=9.0,
    )
    clean = validator.check(sample(t=0.0, field=np.array([0.0, 0.0, 48e-6])), now=0.0)
    dirty = validator.check(
        sample(t=0.05, field=np.array([0.0, 0.0, 48e-6 * 1.4])), now=0.05
    )
    assert dirty.yaw_variance > clean.yaw_variance


def test_field_wildly_off_is_rejected():
    validator = make_validator(
        expected_field_magnitude_t=48e-6, field_magnitude_tolerance=0.25
    )
    field = np.array([0.0, 0.0, 48e-6 * 3.0])
    verdict = validator.check(sample(t=0.0, field=field), now=0.0)
    assert not verdict.accepted
    assert "field magnitude" in verdict.reason
