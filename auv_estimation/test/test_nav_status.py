"""Navigation status tests.

This tracker is what the mission layer reads before deciding whether a
manoeuvre is safe to attempt, so every state transition it can make is pinned
down here.
"""

import math

import pytest

from auv_estimation.nav_status import NavMode, NavStatusConfig, NavStatusTracker

# A 6x6 pose covariance with the diagonal entries this tracker reads.
def pose_cov(xx=0.01, yy=0.01, zz=0.0004, yaw=0.0025):
    cov = [0.0] * 36
    cov[0] = xx
    cov[7] = yy
    cov[14] = zz
    cov[35] = yaw
    return cov


def make_tracker(**overrides):
    return NavStatusTracker(NavStatusConfig(**overrides))


def healthy(tracker, now, dvl=True, imu=True, depth=True, heading=False):
    tracker.update_health("dvl", dvl, now)
    tracker.update_health("imu", imu, now)
    tracker.update_health("depth", depth, now)
    tracker.update_health("heading", heading, now)


# --------------------------------------------------------------------------
# Modes
# --------------------------------------------------------------------------


def test_starts_in_init_before_any_odometry():
    tracker = make_tracker()
    healthy(tracker, 0.0)
    report = tracker.evaluate(0.0)
    assert report.mode == NavMode.INIT
    assert "no odometry" in report.reason


def test_aided_when_everything_is_healthy():
    tracker = make_tracker()
    healthy(tracker, 0.0)
    tracker.update_odometry(pose_cov(), 0.0)
    report = tracker.evaluate(0.0)
    assert report.mode == NavMode.AIDED
    assert report.reason == ""
    assert report.dvl_fused


def test_dead_reckoning_when_the_dvl_loses_lock():
    tracker = make_tracker()
    healthy(tracker, 0.0)
    tracker.update_odometry(pose_cov(), 0.0)

    healthy(tracker, 1.0, dvl=False)
    tracker.update_odometry(pose_cov(), 1.0)
    report = tracker.evaluate(1.0)
    assert report.mode == NavMode.DEAD_RECKONING
    assert not report.dvl_fused
    assert "coasting" in report.reason


def test_fault_when_the_filter_goes_silent():
    """Odometry stopping is a different failure from a sensor stopping."""
    tracker = make_tracker(odom_timeout_s=1.0)
    healthy(tracker, 0.0)
    tracker.update_odometry(pose_cov(), 0.0)
    report = tracker.evaluate(5.0)
    assert report.mode == NavMode.FAULT
    assert "silent" in report.reason


def test_fault_when_a_critical_sensor_is_unavailable():
    """The IMU drives the prediction step; without it there is nothing to propagate."""
    tracker = make_tracker(critical_sensors=("imu",))
    healthy(tracker, 0.0, imu=False)
    tracker.update_odometry(pose_cov(), 0.0)
    report = tracker.evaluate(0.0)
    assert report.mode == NavMode.FAULT
    assert "imu" in report.reason


def test_depth_loss_degrades_but_does_not_fault():
    """Losing depth is serious but the horizontal estimate still stands."""
    tracker = make_tracker()
    healthy(tracker, 0.0, depth=False)
    tracker.update_odometry(pose_cov(), 0.0)
    report = tracker.evaluate(0.0)
    assert report.mode == NavMode.AIDED
    assert not report.depth_fused


def test_fault_takes_precedence_over_dead_reckoning():
    tracker = make_tracker()
    healthy(tracker, 0.0, dvl=False, imu=False)
    tracker.update_odometry(pose_cov(), 0.0)
    assert tracker.evaluate(0.0).mode == NavMode.FAULT


# --------------------------------------------------------------------------
# Dead reckoning clock
# --------------------------------------------------------------------------


def test_dead_reckoning_time_is_zero_while_aided():
    tracker = make_tracker()
    healthy(tracker, 0.0)
    tracker.update_odometry(pose_cov(), 0.0)
    assert tracker.evaluate(0.0).dead_reckoning_time_s == 0.0


def test_dead_reckoning_time_grows_during_a_dropout():
    tracker = make_tracker()
    healthy(tracker, 0.0)
    tracker.update_odometry(pose_cov(), 0.0)
    tracker.evaluate(0.0)

    healthy(tracker, 1.0, dvl=False)
    tracker.update_odometry(pose_cov(), 1.0)
    tracker.evaluate(1.0)
    report = tracker.evaluate(4.0)
    assert report.dead_reckoning_time_s == pytest.approx(3.0)


def test_dead_reckoning_clock_measures_the_current_gap_not_the_last_one():
    tracker = make_tracker()
    healthy(tracker, 0.0)
    tracker.update_odometry(pose_cov(), 0.0)
    tracker.evaluate(0.0)

    # A long first dropout...
    healthy(tracker, 1.0, dvl=False)
    tracker.update_odometry(pose_cov(), 1.0)
    tracker.evaluate(20.0)

    # ...then lock returns...
    healthy(tracker, 21.0)
    tracker.update_odometry(pose_cov(), 21.0)
    assert tracker.evaluate(21.0).dead_reckoning_time_s == 0.0

    # ...and a fresh short one must not inherit the first one's length.
    healthy(tracker, 22.0, dvl=False)
    tracker.update_odometry(pose_cov(), 22.0)
    tracker.evaluate(22.0)
    assert tracker.evaluate(23.0).dead_reckoning_time_s == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Health staleness
# --------------------------------------------------------------------------


def test_stale_health_counts_as_unusable():
    """Gates publish at 2 Hz regardless, so silence means the gate node died."""
    tracker = make_tracker(health_timeout_s=2.0)
    healthy(tracker, 0.0)
    tracker.update_odometry(pose_cov(), 10.0)
    report = tracker.evaluate(10.0)
    assert not report.dvl_fused
    assert report.mode == NavMode.FAULT   # the IMU health went stale too


def test_a_sensor_never_heard_from_is_not_fusable():
    tracker = make_tracker()
    tracker.update_health("dvl", True, 0.0)
    tracker.update_health("imu", True, 0.0)
    tracker.update_odometry(pose_cov(), 0.0)
    report = tracker.evaluate(0.0)
    assert not report.heading_fused
    assert not report.depth_fused


# --------------------------------------------------------------------------
# Reported uncertainty
# --------------------------------------------------------------------------


def test_sigmas_are_square_roots_of_the_covariance_diagonal():
    tracker = make_tracker()
    healthy(tracker, 0.0)
    tracker.update_odometry(pose_cov(xx=0.04, yy=0.09, zz=0.0016, yaw=0.01), 0.0)
    report = tracker.evaluate(0.0)
    assert report.position_sigma_z_m == pytest.approx(0.04)
    assert report.yaw_sigma_rad == pytest.approx(0.1)


def test_horizontal_sigma_reports_the_worse_axis():
    """This number is read to decide whether a manoeuvre is safe."""
    tracker = make_tracker()
    healthy(tracker, 0.0)
    tracker.update_odometry(pose_cov(xx=0.04, yy=0.25), 0.0)
    assert tracker.evaluate(0.0).position_sigma_xy_m == pytest.approx(0.5)


def test_malformed_covariance_is_ignored():
    tracker = make_tracker()
    healthy(tracker, 0.0)
    tracker.update_odometry([0.0] * 9, 0.0)
    assert tracker.evaluate(0.0).mode == NavMode.INIT


def test_gyro_bias_is_passed_through():
    tracker = make_tracker()
    healthy(tracker, 0.0)
    tracker.update_odometry(pose_cov(), 0.0)
    tracker.update_gyro_bias((0.001, -0.002, 0.003))
    assert tracker.evaluate(0.0).gyro_bias == pytest.approx((0.001, -0.002, 0.003))


def test_heading_fusion_is_reported_when_a_source_is_running():
    tracker = make_tracker()
    healthy(tracker, 0.0, heading=True)
    tracker.update_odometry(pose_cov(), 0.0)
    assert tracker.evaluate(0.0).heading_fused
