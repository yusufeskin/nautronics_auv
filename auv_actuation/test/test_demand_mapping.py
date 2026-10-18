"""Axis demand to ArduSub RC channel tests.

Pinned here: every convention flip between this stack's FLU and ArduSub's
sticks, the deadzone skip, saturation, and that neutral is the safe command.
"""

import math

import pytest

from auv_actuation.demand_mapping import AxisDemandMapper, ChannelLimits


def make_mapper(**overrides):
    limits = ChannelLimits(
        neutral_pwm=1500,
        span_pwm=400,
        max_climb_rate_up=0.5,
        max_climb_rate_down=0.5,
        max_yaw_rate=1.0,
        throttle_deadzone_pwm=0,
        yaw_deadzone_pwm=0,
        forward_deadzone_pwm=0,
        lateral_deadzone_pwm=0,
    )
    for key, value in overrides.items():
        setattr(limits, key, value)
    return AxisDemandMapper(limits=limits)


# --------------------------------------------------------------------------
# Neutral
# --------------------------------------------------------------------------


def test_zero_demand_is_every_stick_centred():
    d = make_mapper().map(0.0, 0.0, 0.0, 0.0)
    assert (d.pitch, d.roll, d.throttle, d.yaw, d.forward, d.lateral) == (1500,) * 6
    assert not d.saturated


def test_neutral_is_every_stick_centred():
    d = make_mapper().neutral()
    assert (d.pitch, d.roll, d.throttle, d.yaw, d.forward, d.lateral) == (1500,) * 6


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_demand_falls_back_to_neutral(bad):
    d = make_mapper().map(bad, 0.0, 0.0, 0.0)
    assert (d.throttle, d.yaw, d.forward, d.lateral) == (1500,) * 4


def test_roll_and_pitch_are_always_level():
    d = make_mapper().map(1.0, 1.0, 0.5, 1.0)
    assert d.pitch == 1500
    assert d.roll == 1500


# --------------------------------------------------------------------------
# Directions: FLU in, ArduSub sticks out
# --------------------------------------------------------------------------


def test_forward_is_not_flipped():
    assert make_mapper().map(0.5, 0.0, 0.0, 0.0).forward == 1700


def test_left_in_flu_is_low_lateral_stick():
    """ArduSub's lateral stick is right-positive; FLU sway is left-positive."""
    assert make_mapper().map(0.0, 0.5, 0.0, 0.0).lateral == 1300


def test_climb_is_high_throttle():
    assert make_mapper().map(0.0, 0.0, 0.25, 0.0).throttle == 1700


def test_descend_is_low_throttle():
    assert make_mapper().map(0.0, 0.0, -0.25, 0.0).throttle == 1300


def test_counter_clockwise_in_flu_is_low_yaw_stick():
    """ArduSub's yaw stick is clockwise-positive; FLU yaw rate is CCW-positive."""
    assert make_mapper().map(0.0, 0.0, 0.0, 0.5).yaw == 1300


# --------------------------------------------------------------------------
# Scaling
# --------------------------------------------------------------------------


def test_climb_rate_scales_against_the_vehicles_own_maximum():
    mapper = make_mapper(max_climb_rate_up=0.5, max_climb_rate_down=0.25)
    assert mapper.map(0.0, 0.0, 0.5, 0.0).throttle == 1900
    assert mapper.map(0.0, 0.0, -0.25, 0.0).throttle == 1100


def test_full_thrust_is_full_stick():
    d = make_mapper().map(1.0, -1.0, 0.0, 0.0)
    assert d.forward == 1900
    assert d.lateral == 1900
    assert not d.saturated


def test_beyond_full_stick_is_clipped_and_reported():
    d = make_mapper().map(1.5, 0.0, 0.0, 0.0)
    assert d.forward == 1900
    assert d.saturated


def test_a_climb_faster_than_the_vehicle_can_is_reported():
    d = make_mapper(max_climb_rate_up=0.5).map(0.0, 0.0, 0.8, 0.0)
    assert d.throttle == 1900
    assert d.saturated


# --------------------------------------------------------------------------
# Deadzones
# --------------------------------------------------------------------------


def test_small_demand_starts_at_the_edge_of_the_deadzone():
    """Inside ArduSub's deadzone a demand would be ignored, so it skips over it."""
    mapper = make_mapper(throttle_deadzone_pwm=100)
    d = mapper.map(0.0, 0.0, 0.005, 0.0)       # 1 % of full climb
    assert d.throttle > 1600


def test_full_stick_is_still_full_with_a_deadzone():
    mapper = make_mapper(throttle_deadzone_pwm=100)
    assert mapper.map(0.0, 0.0, 0.5, 0.0).throttle == 1900
    assert mapper.map(0.0, 0.0, -0.5, 0.0).throttle == 1100


def test_deadzone_scaling_is_linear_beyond_the_edge():
    mapper = make_mapper(throttle_deadzone_pwm=100)
    # Half of the climb range is halfway across the 300 PWM beyond the deadzone.
    assert mapper.map(0.0, 0.0, 0.25, 0.0).throttle == 1750


def test_zero_demand_stays_centred_with_a_deadzone():
    mapper = make_mapper(throttle_deadzone_pwm=100, yaw_deadzone_pwm=50)
    d = mapper.map(0.0, 0.0, 0.0, 0.0)
    assert d.throttle == 1500
    assert d.yaw == 1500
