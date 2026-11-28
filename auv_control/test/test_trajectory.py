import math

import pytest

from auv_control.trajectory import Limits, StraightLineTrajectory

LIM = Limits(max_speed=0.25, max_accel=0.1, max_vertical_speed=0.15, max_yaw_rate=math.radians(15))


def samples(traj, dt=0.01):
    n = int(traj.duration / dt) + 1
    return [traj.sample(i * dt) for i in range(n + 1)]


def test_starts_at_rest_on_start_and_ends_at_rest_on_goal():
    tr = StraightLineTrajectory((1, 2, -0.5), 0.0, (5, 4, -1.0), 0.5, LIM)
    s0, s1 = tr.sample(0.0), tr.sample(tr.duration)
    assert (s0.x, s0.y, s0.z, s0.yaw) == pytest.approx((1, 2, -0.5, 0.0))
    assert (s1.x, s1.y, s1.z, s1.yaw) == pytest.approx((5, 4, -1.0, 0.5))
    assert math.hypot(s0.vx, s0.vy) == pytest.approx(0.0, abs=1e-9)
    assert (s1.vx, s1.vy, s1.vz, s1.yaw_rate) == (0.0, 0.0, 0.0, 0.0)


def test_respects_speed_accel_and_vertical_limits():
    tr = StraightLineTrajectory((0, 0, 0), 0.0, (10, 0, -3), 0.0, LIM)
    ss = samples(tr)
    assert max(math.sqrt(s.vx ** 2 + s.vy ** 2 + s.vz ** 2) for s in ss) <= LIM.max_speed + 1e-9
    assert max(abs(s.vz) for s in ss) <= LIM.max_vertical_speed + 1e-9
    for a, b in zip(ss, ss[1:]):
        dv = math.sqrt((b.vx - a.vx) ** 2 + (b.vy - a.vy) ** 2 + (b.vz - a.vz) ** 2)
        assert dv / 0.01 <= LIM.max_accel + 1e-6


def test_long_move_reaches_cruise_speed_and_has_expected_duration():
    tr = StraightLineTrajectory((0, 0, 0), 0.0, (10, 0, 0), 0.0, LIM)
    # 2.5 s accel (0.3125 m) + cruise 9.375 m at 0.25 + 2.5 s decel
    assert tr.duration == pytest.approx(2.5 + 9.375 / 0.25 + 2.5)
    assert tr.sample(tr.duration / 2).vx == pytest.approx(0.25)


def test_short_move_is_triangular():
    tr = StraightLineTrajectory((0, 0, 0), 0.0, (0.2, 0, 0), 0.0, LIM)
    peak = max(s.vx for s in samples(tr))
    assert peak == pytest.approx(math.sqrt(LIM.max_accel * 0.2), rel=1e-2)
    assert tr.sample(tr.duration).x == pytest.approx(0.2)


def test_position_is_continuous_and_monotonic_along_line():
    tr = StraightLineTrajectory((0, 0, 0), 0.0, (3, 4, 0), 0.0, LIM)
    ss = samples(tr)
    d = [math.hypot(s.x, s.y) for s in ss]
    assert all(b >= a - 1e-9 for a, b in zip(d, d[1:]))
    assert max(abs(b - a) for a, b in zip(d, d[1:])) < 0.01


def test_pure_rotation_takes_short_way_at_yaw_rate_limit():
    tr = StraightLineTrajectory((0, 0, -1), math.radians(170), (0, 0, -1), math.radians(-170), LIM)
    assert tr.duration == pytest.approx(math.radians(20) / LIM.max_yaw_rate)
    assert tr.sample(tr.duration / 2).yaw_rate > 0      # CCW through 180
    assert tr.sample(tr.duration).yaw == pytest.approx(math.radians(-170))


def test_big_turn_on_short_move_slows_the_line_to_finish_together():
    tr = StraightLineTrajectory((0, 0, 0), 0.0, (0.5, 0, 0), math.pi / 2, LIM)
    t_yaw = (math.pi / 2) / LIM.max_yaw_rate
    assert tr.duration == pytest.approx(t_yaw)
    assert 2 * tr.t_acc + tr.t_cruise == pytest.approx(t_yaw)
    assert max(abs(s.yaw_rate) for s in samples(tr)) <= LIM.max_yaw_rate * 2.01  # peaks mid-move
    assert tr.sample(tr.duration).x == pytest.approx(0.5)


def test_zero_length_zero_turn_is_immediately_done():
    tr = StraightLineTrajectory((1, 1, -1), 0.3, (1, 1, -1), 0.3, LIM)
    assert tr.duration == 0.0
    s = tr.sample(0.0)
    assert (s.x, s.y, s.z, s.yaw) == pytest.approx((1, 1, -1, 0.3))


def test_acceleration_matches_velocity_derivative():
    tr = StraightLineTrajectory((0, 0, 0), 0.0, (3, 4, -1), 0.0, LIM)
    dt = 1e-4
    for t in (0.5, tr.t_acc + 1.0, tr.duration - 0.5):
        a, b = tr.sample(t), tr.sample(t + dt)
        assert (b.vx - a.vx) / dt == pytest.approx(a.ax, abs=1e-3)
        assert (b.vy - a.vy) / dt == pytest.approx(a.ay, abs=1e-3)
        assert (b.vz - a.vz) / dt == pytest.approx(a.az, abs=1e-3)
    assert tr.sample(0.0).ax > 0 and tr.sample(tr.duration - 0.1).ax < 0
    assert tr.sample(tr.duration).ax == 0.0


def test_replan_starts_at_current_speed_along_line():
    # Moving +x at 0.2 m/s, new goal straight ahead: no jump to rest.
    tr = StraightLineTrajectory((0, 0, 0), 0.0, (5, 0, 0), 0.0, LIM, start_velocity=(0.2, 0.05, 0))
    s0 = tr.sample(0.0)
    assert s0.vx == pytest.approx(0.2)
    assert s0.ax > 0                      # still speeding up to 0.25
    assert tr.sample(tr.duration).x == pytest.approx(5.0)
    rest = StraightLineTrajectory((0, 0, 0), 0.0, (5, 0, 0), 0.0, LIM)
    assert tr.duration < rest.duration


def test_replan_ignores_velocity_away_from_goal():
    tr = StraightLineTrajectory((0, 0, 0), 0.0, (5, 0, 0), 0.0, LIM, start_velocity=(-0.2, 0, 0))
    assert tr.sample(0.0).vx == 0.0


def test_start_faster_than_cruise_slows_down_first():
    tr = StraightLineTrajectory((0, 0, 0), 0.0, (5, 0, 0), 0.0, LIM, start_velocity=(0.4, 0, 0))
    assert tr.sample(0.0).vx == pytest.approx(0.4)
    assert tr.sample(0.0).ax < 0
    ss = samples(tr)
    assert max(s.vx for s in ss[len(ss) // 2:]) <= LIM.max_speed + 1e-9
    assert tr.sample(tr.duration).x == pytest.approx(5.0)


def test_start_speed_too_high_to_stop_is_clamped():
    tr = StraightLineTrajectory((0, 0, 0), 0.0, (0.1, 0, 0), 0.0, LIM, start_velocity=(0.25, 0, 0))
    v0 = tr.sample(0.0).vx
    assert v0 == pytest.approx(math.sqrt(2 * LIM.max_accel * 0.1))
    ss = samples(tr, dt=0.001)
    assert all(s.x <= 0.1 + 1e-9 for s in ss)
    assert tr.sample(tr.duration).x == pytest.approx(0.1)


def test_moving_start_with_big_turn_still_finishes_with_yaw():
    tr = StraightLineTrajectory((0, 0, 0), 0.0, (1.0, 0, 0), math.pi, LIM, start_velocity=(0.2, 0, 0))
    assert tr.duration == pytest.approx(math.pi / LIM.max_yaw_rate)
    assert tr.profile.duration == pytest.approx(tr.duration, rel=1e-6)
    assert tr.sample(tr.duration).x == pytest.approx(1.0)
