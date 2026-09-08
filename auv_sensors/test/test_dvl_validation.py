"""DVL gate logic tests.

Each test pins one rejection rule or one piece of lock hysteresis. Together
they are the specification of what the gate lets through, which matters
because everything downstream trusts anything that gets past it.
"""

import numpy as np
import pytest

from auv_core import frames
from auv_sensors.dvl_validation import (
    MODE_BOTTOM,
    MODE_WATER,
    DvlLimits,
    DvlSample,
    DvlValidator,
    altitude_is_valid,
)


def make_validator(**limit_overrides):
    limits = DvlLimits(**limit_overrides)
    validator = DvlValidator(limits=limits)
    # Start locked so tests exercise the interesting transitions rather than
    # the acquisition ramp; acquisition has its own tests below.
    validator._locked = True
    return validator


def sample(t=0.0, velocity=(0.1, 0.0, 0.0), beams=4, valid=True,
           mode=MODE_BOTTOM, covariance=None, altitude=2.0):
    return DvlSample(
        stamp=t,
        velocity=np.array(velocity, dtype=float),
        num_good_beams=beams,
        beams_valid=valid,
        velocity_mode=mode,
        covariance=covariance,
        altitude=altitude,
    )


# --------------------------------------------------------------------------
# Happy path and frame conversion
# --------------------------------------------------------------------------


def test_good_sample_is_accepted():
    validator = make_validator()
    verdict = validator.check(sample(), now=0.0)
    assert verdict.accepted
    assert verdict.reason == ""
    assert verdict.degraded_reason == ""


def test_frd_input_is_converted_to_flu():
    """The A50 reports FRD; every ROS topic in this stack is FLU."""
    validator = make_validator()
    verdict = validator.check(sample(velocity=(0.3, 0.2, 0.1)), now=0.0)
    np.testing.assert_allclose(verdict.velocity_flu, [0.3, -0.2, -0.1], atol=1e-12)


def test_flu_input_is_passed_through_unchanged():
    validator = make_validator()
    validator.input_is_frd = False
    verdict = validator.check(sample(velocity=(0.3, 0.2, 0.1)), now=0.0)
    np.testing.assert_allclose(verdict.velocity_flu, [0.3, 0.2, 0.1], atol=1e-12)


def test_covariance_is_rotated_with_the_velocity():
    """Rotating a mean without its covariance feeds the filter a wrong uncertainty."""
    validator = make_validator()
    cov = np.diag([0.01, 0.02, 0.03])
    verdict = validator.check(sample(covariance=cov), now=0.0)
    expected = frames.rotate_covariance(frames.FLU_FRD, cov)
    np.testing.assert_allclose(verdict.covariance_flu, expected, atol=1e-12)


# --------------------------------------------------------------------------
# Rejection rules
# --------------------------------------------------------------------------


def test_too_few_beams_is_rejected():
    validator = make_validator(min_good_beams=3)
    verdict = validator.check(sample(beams=2), now=0.0)
    assert not verdict.accepted
    assert "good beams" in verdict.reason


def test_invalid_beam_flag_is_rejected_even_with_enough_beams():
    validator = make_validator()
    verdict = validator.check(sample(beams=4, valid=False), now=0.0)
    assert not verdict.accepted


def test_three_beams_is_accepted_but_degraded_with_inflated_covariance():
    validator = make_validator()
    four = validator.check(sample(t=0.0, beams=4), now=0.0)
    validator.reset()
    three = validator.check(sample(t=1.0, beams=3), now=1.0)

    assert three.accepted
    assert "3 of 4 beams" in three.degraded_reason
    # A weaker geometry must be trusted less, not equally.
    assert three.covariance_flu[0, 0] > four.covariance_flu[0, 0]


def test_water_track_is_rejected_by_default():
    """Water-track velocity is relative to the water, not the ground."""
    validator = make_validator()
    verdict = validator.check(sample(mode=MODE_WATER), now=0.0)
    assert not verdict.accepted
    assert "water-track" in verdict.reason


def test_water_track_is_degraded_when_explicitly_allowed():
    validator = make_validator(allow_water_track=True)
    verdict = validator.check(sample(mode=MODE_WATER), now=0.0)
    assert verdict.accepted
    assert "relative to water" in verdict.degraded_reason


def test_non_finite_velocity_is_rejected():
    validator = make_validator()
    verdict = validator.check(sample(velocity=(float("nan"), 0.0, 0.0)), now=0.0)
    assert not verdict.accepted
    assert "non-finite" in verdict.reason


def test_speed_above_the_physical_limit_is_rejected():
    validator = make_validator(max_speed_mps=2.0)
    verdict = validator.check(sample(velocity=(5.0, 0.0, 0.0)), now=0.0)
    assert not verdict.accepted
    assert "above" in verdict.reason


def test_impossible_acceleration_is_rejected():
    validator = make_validator(max_accel_mps2=5.0)
    validator.check(sample(t=0.0, velocity=(0.0, 0.0, 0.0)), now=0.0)
    # 1.5 m/s in 125 ms is 12 m/s^2, which no thruster on this vehicle does.
    verdict = validator.check(sample(t=0.125, velocity=(1.5, 0.0, 0.0)), now=0.125)
    assert not verdict.accepted
    assert "acceleration" in verdict.reason


def test_acceleration_check_compares_against_the_last_accepted_sample():
    """One outlier must not poison the comparison for the next good sample.

    Comparing against the rejected sample would reject the recovery too,
    which is how a single bad ping turns into a lasting dropout.
    """
    validator = make_validator(max_accel_mps2=5.0)
    validator.check(sample(t=0.0, velocity=(0.0, 0.0, 0.0)), now=0.0)
    outlier = validator.check(sample(t=0.125, velocity=(1.5, 0.0, 0.0)), now=0.125)
    assert not outlier.accepted

    # Back near the last good velocity: this is plausible motion and must pass.
    recovery = validator.check(sample(t=0.25, velocity=(0.1, 0.0, 0.0)), now=0.25)
    assert recovery.accepted


def test_acceleration_check_is_skipped_across_a_long_gap():
    """A gap longer than max_dt_s is not evidence of anything."""
    validator = make_validator(max_accel_mps2=5.0, max_dt_s=1.0)
    validator.check(sample(t=0.0, velocity=(0.0, 0.0, 0.0)), now=0.0)
    verdict = validator.check(sample(t=10.0, velocity=(1.5, 0.0, 0.0)), now=10.0)
    assert verdict.accepted


def test_clock_skew_is_rejected_and_names_the_cause():
    validator = make_validator(max_clock_skew_s=0.5)
    verdict = validator.check(sample(t=100.0), now=0.0)
    assert not verdict.accepted
    assert "NTP" in verdict.reason


def test_timestamp_from_the_future_is_rejected():
    validator = make_validator(max_clock_skew_s=0.5)
    verdict = validator.check(sample(t=5.0), now=0.0)
    assert not verdict.accepted
    assert "ahead of" in verdict.reason


# --------------------------------------------------------------------------
# Lock hysteresis
# --------------------------------------------------------------------------


def test_single_bad_sample_does_not_drop_the_lock():
    """One reflected ping must not tear velocity aiding away from the filter."""
    validator = make_validator(lock_loss_streak=4)
    validator.check(sample(t=0.0), now=0.0)
    verdict = validator.check(sample(t=0.125, beams=0, valid=False), now=0.125)
    assert not verdict.accepted
    assert verdict.locked


def test_sustained_dropout_drops_the_lock():
    validator = make_validator(lock_loss_streak=4)
    validator.check(sample(t=0.0), now=0.0)
    t = 0.125
    for _ in range(4):
        verdict = validator.check(sample(t=t, beams=0, valid=False), now=t)
        t += 0.125
    assert not verdict.locked


def test_lock_is_regained_only_after_consecutive_good_samples():
    validator = DvlValidator(limits=DvlLimits(lock_regain_streak=2))
    assert not validator.locked

    first = validator.check(sample(t=0.0), now=0.0)
    assert first.accepted
    assert not first.locked  # accepted, but aiding does not restart yet

    second = validator.check(sample(t=0.125), now=0.125)
    assert second.locked


def test_lock_loss_clears_motion_history():
    """Velocity from before a dropout says nothing about velocity after it."""
    validator = make_validator(lock_loss_streak=2, max_accel_mps2=5.0, lock_regain_streak=1)
    validator.check(sample(t=0.0, velocity=(0.0, 0.0, 0.0)), now=0.0)

    t = 0.125
    for _ in range(2):
        validator.check(sample(t=t, beams=0, valid=False), now=t)
        t += 0.125
    assert not validator.locked

    # A large velocity step across the dropout must not be judged as
    # acceleration, because the elapsed motion is unknown.
    verdict = validator.check(sample(t=t, velocity=(1.5, 0.0, 0.0)), now=t)
    assert verdict.accepted


# --------------------------------------------------------------------------
# Covariance resolution
# --------------------------------------------------------------------------


def test_unknown_covariance_falls_back_to_the_configured_default():
    validator = make_validator(default_velocity_variance=0.01)
    minus_one = np.full(9, -1.0)
    verdict = validator.check(sample(covariance=minus_one), now=0.0)
    np.testing.assert_allclose(np.diag(verdict.covariance_flu), [0.01] * 3, atol=1e-12)


def test_all_zero_covariance_is_replaced_not_trusted():
    """Zero covariance means 'perfectly known', which no sensor ever is."""
    validator = make_validator(default_velocity_variance=0.01)
    verdict = validator.check(sample(covariance=np.zeros(9)), now=0.0)
    assert np.all(np.diag(verdict.covariance_flu) > 0.0)


def test_missing_covariance_falls_back_to_the_default():
    validator = make_validator(default_velocity_variance=0.01)
    verdict = validator.check(sample(covariance=None), now=0.0)
    np.testing.assert_allclose(np.diag(verdict.covariance_flu), [0.01] * 3, atol=1e-12)


# --------------------------------------------------------------------------
# Altitude
# --------------------------------------------------------------------------


@pytest.mark.parametrize("value,expected", [
    (2.5, True),
    (0.0, False),
    (-1.0, False),
    (float("nan"), False),
    (float("inf"), False),
])
def test_altitude_validity(value, expected):
    assert altitude_is_valid(value) is expected
