"""Unit tests for angle wrapping and the yaw/roll/loop control helpers."""

import math

import pytest

from auv_control.angles import (
    accumulate_turn, clamp, downward_thrust_body, loop_velocities,
    normalize_angle, p_turn_rate,
)

DEG = math.radians


@pytest.mark.parametrize('angle_deg, expected_deg', [
    (0.0, 0.0),
    (10.0, 10.0),
    (370.0, 10.0),
    (-370.0, -10.0),
    (190.0, -170.0),
    (-190.0, 170.0),
    (720.0 + 45.0, 45.0),
    (-1080.0 - 30.0, -30.0),
])
def test_normalize_angle_wraps_into_minus_pi_pi(angle_deg, expected_deg):
    assert normalize_angle(DEG(angle_deg)) == pytest.approx(DEG(expected_deg), abs=1e-9)


def test_normalize_angle_keeps_both_boundaries():
    assert normalize_angle(math.pi) == math.pi
    assert normalize_angle(-math.pi) == -math.pi


def test_yaw_error_takes_shortest_path_across_180():
    # target 170 deg, current -170 deg -> turn -20 deg, not +340 deg
    error = normalize_angle(DEG(170.0) - DEG(-170.0))
    assert error == pytest.approx(DEG(-20.0))
    error = normalize_angle(DEG(-170.0) - DEG(170.0))
    assert error == pytest.approx(DEG(20.0))


def test_p_turn_rate_is_proportional_inside_limits():
    assert p_turn_rate(0.5, 0.4, 1.0, 0.05, 0.0) == pytest.approx(0.2)
    assert p_turn_rate(-0.5, 0.4, 1.0, 0.05, 0.0) == pytest.approx(-0.2)
    assert p_turn_rate(0.0, 0.4, 1.0, 0.05, 0.0) == 0.0


def test_p_turn_rate_saturates_at_max_speed_with_sign():
    assert p_turn_rate(5.0, 0.4, 0.3, 0.05, 0.0) == 0.3
    assert p_turn_rate(-5.0, 0.4, 0.3, 0.05, 0.0) == -0.3


def test_p_turn_rate_min_speed_only_far_from_target():
    # |error| = 0.2 > 3 * 0.05 -> boosted from 0.08 to min speed 0.1
    assert p_turn_rate(0.2, 0.4, 1.0, 0.05, 0.1) == 0.1
    assert p_turn_rate(-0.2, 0.4, 1.0, 0.05, 0.1) == -0.1
    # |error| = 0.1 <= 0.15 -> plain P output, no boost
    assert p_turn_rate(0.1, 0.4, 1.0, 0.05, 0.1) == pytest.approx(0.04)


def test_accumulate_turn_handles_wraparound():
    # 179 deg -> -179 deg is a +2 deg step, not -358 deg
    assert accumulate_turn(0.0, DEG(179.0), DEG(-179.0)) == pytest.approx(DEG(2.0))
    assert accumulate_turn(0.0, DEG(-179.0), DEG(179.0)) == pytest.approx(DEG(-2.0))


def test_accumulate_turn_full_revolution_through_wrapped_samples():
    total = 0.0
    prev = normalize_angle(DEG(100.0))
    for step in range(1, 37):  # 36 steps of 10 deg starting at 100 deg
        current = normalize_angle(DEG(100.0 + 10.0 * step))
        total = accumulate_turn(total, prev, current)
        prev = current
    assert total == pytest.approx(2.0 * math.pi)


def test_loop_velocities_one_circle_in_duration():
    v, w = loop_velocities(10.0, 2.0)
    assert w == pytest.approx(2.0 * math.pi / 10.0)
    assert v == pytest.approx(2.0 * 2.0 * math.pi / 10.0)
    assert w * 10.0 == pytest.approx(2.0 * math.pi)


def test_loop_velocities_negative_radius_reverses_turn_only():
    v_pos, w_pos = loop_velocities(10.0, 2.0)
    v_neg, w_neg = loop_velocities(10.0, -2.0)
    assert v_neg == pytest.approx(v_pos)
    assert v_neg > 0
    assert w_neg == pytest.approx(-w_pos)


def test_clamp():
    assert clamp(0.9, -0.4, 0.4) == 0.4
    assert clamp(-0.9, -0.4, 0.4) == -0.4
    assert clamp(0.1, -0.4, 0.4) == 0.1


@pytest.mark.parametrize('roll_deg, expected_y, expected_z', [
    (0.0, 0.0, -0.25),     # upright: push straight down in body z
    (90.0, -0.25, 0.0),    # rolled 90 deg: world down is body -y
    (180.0, 0.0, 0.25),    # upside down: body +z points to world down
    (-90.0, 0.25, 0.0),
])
def test_downward_thrust_body(roll_deg, expected_y, expected_z):
    y, z = downward_thrust_body(DEG(roll_deg), 0.25)
    assert y == pytest.approx(expected_y, abs=1e-12)
    assert z == pytest.approx(expected_z, abs=1e-12)
    assert math.hypot(y, z) == pytest.approx(0.25)
