"""IMU gate logic and gyro bias estimator tests.

The bias tests matter most: with no magnetometer in the loop, yaw is a gyro
integral, and an unremoved turn-on bias is the largest single error in the
navigation stack.
"""

import numpy as np
import pytest

from auv_sensors.imu_validation import (
    STANDARD_GRAVITY,
    GyroBiasEstimator,
    ImuLimits,
    ImuSample,
    ImuValidator,
)

GRAVITY_UP = np.array([0.0, 0.0, STANDARD_GRAVITY])


def sample(t=0.0, gyro=(0.0, 0.0, 0.0), accel=None):
    return ImuSample(
        stamp=t,
        angular_velocity=np.array(gyro, dtype=float),
        linear_acceleration=GRAVITY_UP if accel is None else np.array(accel, dtype=float),
    )


def feed_still(estimator_or_validator, bias, duration_s, rate_hz=100.0, start=0.0):
    """Push perfectly still samples carrying a constant gyro bias."""
    period = 1.0 / rate_hz
    t = start
    steps = int(duration_s * rate_hz) + 1
    for _ in range(steps):
        s = sample(t=t, gyro=bias)
        if isinstance(estimator_or_validator, GyroBiasEstimator):
            estimator_or_validator.update(s)
        else:
            estimator_or_validator.check(s, now=t)
        t += period
    return t


# --------------------------------------------------------------------------
# Range checks
# --------------------------------------------------------------------------


def test_good_sample_is_accepted():
    validator = ImuValidator()
    verdict = validator.check(sample(gyro=(0.01, 0.0, 0.0)), now=0.0)
    assert verdict.accepted


def test_non_finite_sample_is_rejected():
    validator = ImuValidator()
    verdict = validator.check(sample(gyro=(float("nan"), 0.0, 0.0)), now=0.0)
    assert not verdict.accepted
    assert "non-finite" in verdict.reason


def test_saturated_gyro_is_rejected():
    """A clipped sample is not a measurement; fusing it drags the estimate."""
    validator = ImuValidator(limits=ImuLimits(max_gyro_rad_s=8.0))
    verdict = validator.check(sample(gyro=(20.0, 0.0, 0.0)), now=0.0)
    assert not verdict.accepted
    assert "full scale" in verdict.reason


def test_saturated_accelerometer_is_rejected():
    validator = ImuValidator(limits=ImuLimits(max_accel_mps2=58.8))
    verdict = validator.check(sample(accel=(0.0, 0.0, 200.0)), now=0.0)
    assert not verdict.accepted
    assert "full scale" in verdict.reason


def test_clock_skew_is_rejected():
    validator = ImuValidator(limits=ImuLimits(max_clock_skew_s=0.2))
    verdict = validator.check(sample(t=10.0), now=0.0)
    assert not verdict.accepted
    assert "local clock" in verdict.reason


# --------------------------------------------------------------------------
# Gyro bias estimation
# --------------------------------------------------------------------------


def test_bias_is_not_ready_before_the_settle_window():
    estimator = GyroBiasEstimator(settle_time_s=20.0, min_samples=200)
    feed_still(estimator, bias=(0.01, 0.0, 0.0), duration_s=5.0)
    assert not estimator.ready
    np.testing.assert_allclose(estimator.bias, np.zeros(3))


def test_bias_converges_to_the_true_offset_when_still():
    estimator = GyroBiasEstimator(settle_time_s=20.0, min_samples=200)
    truth = (0.004, -0.002, 0.006)
    feed_still(estimator, bias=truth, duration_s=25.0)
    assert estimator.ready
    np.testing.assert_allclose(estimator.bias, truth, atol=1e-9)


def test_motion_resets_the_accumulation():
    estimator = GyroBiasEstimator(settle_time_s=20.0, min_samples=200)
    t = feed_still(estimator, bias=(0.004, 0.0, 0.0), duration_s=15.0)
    # A clear rotation: well above the stillness threshold.
    estimator.update(sample(t=t, gyro=(0.5, 0.0, 0.0)))
    t += 0.01
    feed_still(estimator, bias=(0.004, 0.0, 0.0), duration_s=10.0, start=t)
    # 15 s + 10 s of stillness, but never 20 s uninterrupted.
    assert not estimator.ready


def test_a_converged_bias_survives_later_motion():
    """Bumping the vehicle must not discard a good calibration."""
    estimator = GyroBiasEstimator(settle_time_s=20.0, min_samples=200)
    truth = (0.004, -0.002, 0.006)
    t = feed_still(estimator, bias=truth, duration_s=25.0)
    assert estimator.ready

    estimator.update(sample(t=t, gyro=(1.0, 0.0, 0.0)))
    assert estimator.ready
    np.testing.assert_allclose(estimator.bias, truth, atol=1e-9)


def test_linear_acceleration_prevents_stillness():
    """Being lifted counts as motion even with the gyro quiet."""
    estimator = GyroBiasEstimator(
        settle_time_s=1.0, min_samples=10, still_accel_tolerance_mps2=0.5
    )
    t = 0.0
    for _ in range(200):
        estimator.update(sample(t=t, gyro=(0.001, 0.0, 0.0), accel=(0.0, 0.0, 15.0)))
        t += 0.01
    assert not estimator.ready


def test_still_duration_tracks_the_current_quiet_stretch():
    estimator = GyroBiasEstimator(settle_time_s=20.0, min_samples=200)
    t = feed_still(estimator, bias=(0.001, 0.0, 0.0), duration_s=5.0)
    assert estimator.still_duration(t) == pytest.approx(5.0, abs=0.05)

    estimator.update(sample(t=t, gyro=(1.0, 0.0, 0.0)))
    assert estimator.still_duration(t) == 0.0


# --------------------------------------------------------------------------
# Bias removal through the validator
# --------------------------------------------------------------------------


def test_validator_subtracts_the_converged_bias():
    validator = ImuValidator(
        bias_estimator=GyroBiasEstimator(settle_time_s=5.0, min_samples=50)
    )
    truth = np.array([0.004, -0.002, 0.006])
    t = feed_still(validator, bias=tuple(truth), duration_s=8.0)
    assert validator.bias_estimator.ready

    # A genuine 0.2 rad/s yaw rate on top of the bias.
    rate = truth + np.array([0.0, 0.0, 0.2])
    verdict = validator.check(sample(t=t, gyro=tuple(rate)), now=t)
    np.testing.assert_allclose(verdict.angular_velocity, [0.0, 0.0, 0.2], atol=1e-9)


def test_uncalibrated_gyro_is_flagged_degraded():
    validator = ImuValidator()
    verdict = validator.check(sample(gyro=(0.5, 0.0, 0.0)), now=0.0)
    assert verdict.accepted
    assert "still" in verdict.degraded_reason


def test_bias_subtraction_can_be_disabled():
    validator = ImuValidator(subtract_bias=False)
    verdict = validator.check(sample(gyro=(0.01, 0.02, 0.03)), now=0.0)
    np.testing.assert_allclose(verdict.angular_velocity, [0.01, 0.02, 0.03], atol=1e-12)
    assert verdict.degraded_reason == ""
