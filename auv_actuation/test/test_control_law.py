"""Pose controller tests, for ArduSub in ALT_HOLD.

The behaviour being pinned here: horizontal axes are a velocity PID on the
DVL with a capped approach speed; depth and yaw are proportional rate demands
that go to zero, and stay there, at the target; a reference that does not move
is a position hold.
"""

import math

import numpy as np
import pytest

from auv_actuation.control_law import Axis, ControlGains, PoseController, Reference


def make_controller(**overrides):
    gains = ControlGains(
        kp_position=0.8,
        max_speed_xy=0.5,
        kp_depth=0.8,
        max_climb_rate=0.3,
        kp_yaw=1.0,
        max_yaw_rate=0.5,
        surge=Axis(kp=1.2, ki=0.3, output_limit=1.0, integral_limit=0.3),
        sway=Axis(kp=1.2, ki=0.3, output_limit=1.0, integral_limit=0.3),
    )
    for key, value in overrides.items():
        setattr(gains, key, value)
    return PoseController(gains=gains)


def step(controller, reference, position=(0, 0, 0), yaw=0.0, velocity=(0, 0),
         dt=0.05, horizontal_ok=True):
    return controller.step(reference, position, yaw, velocity, dt, horizontal_ok)


# --------------------------------------------------------------------------
# Yaw: a turn-rate demand, proportional only
# --------------------------------------------------------------------------


def test_heading_error_produces_a_turn_in_the_right_direction():
    out = step(make_controller(), Reference(yaw=math.radians(20.0)))
    assert out.yaw_error == pytest.approx(math.radians(20.0))
    assert out.yaw_rate > 0.0     # positive yaw rate turns left in FLU


def test_turning_stops_when_the_heading_error_reaches_zero():
    """Zero rate centres the stick, and ArduSub then holds that heading."""
    out = step(make_controller(), Reference(yaw=math.radians(45.0)), yaw=math.radians(45.0))
    assert out.yaw_rate == pytest.approx(0.0)


def test_heading_error_takes_the_short_way_round():
    """Turning from 170 to -170 degrees is 20 degrees, not 340."""
    out = step(make_controller(), Reference(yaw=math.radians(-170.0)), yaw=math.radians(170.0))
    assert out.yaw_error == pytest.approx(math.radians(20.0), abs=1e-9)
    assert out.yaw_rate > 0.0


def test_yaw_rate_is_capped():
    out = step(make_controller(max_yaw_rate=0.3), Reference(yaw=math.radians(179.0)))
    assert out.yaw_rate == pytest.approx(0.3)


def test_yaw_has_no_integral_action():
    """ArduSub closes the rate loop; a standing error here must not grow."""
    controller = make_controller()
    reference = Reference(yaw=0.1)
    first = step(controller, reference).yaw_rate
    for _ in range(200):
        last = step(controller, reference).yaw_rate
    assert last == pytest.approx(first)


def test_no_yaw_target_means_keep_whatever_heading_you_have():
    out = step(make_controller(), Reference(x=1.0), yaw=math.radians(30.0))
    assert out.yaw_rate == 0.0
    assert out.yaw_error == 0.0


def test_yaw_rate_feedforward_passes_through():
    out = step(make_controller(), Reference(yaw=0.0, yaw_rate=0.2))
    assert out.yaw_rate == pytest.approx(0.2)


# --------------------------------------------------------------------------
# Depth: a climb-rate demand, proportional only
# --------------------------------------------------------------------------


def test_target_above_means_climb():
    """ENU: z is up, so a target at -1.0 from -2.0 is above the vehicle."""
    out = step(make_controller(), Reference(z=-1.0), position=(0, 0, -2.0))
    assert out.climb_rate > 0.0


def test_target_below_means_descend():
    out = step(make_controller(), Reference(z=-2.0), position=(0, 0, -1.0))
    assert out.climb_rate < 0.0


def test_climb_rate_tapers_to_zero_at_depth():
    controller = make_controller(kp_depth=0.8)
    near = step(controller, Reference(z=-1.0), position=(0, 0, -1.1))
    assert near.climb_rate == pytest.approx(0.08)
    there = step(controller, Reference(z=-1.0), position=(0, 0, -1.0))
    assert there.climb_rate == pytest.approx(0.0)


def test_climb_rate_is_capped():
    out = step(make_controller(max_climb_rate=0.3), Reference(z=-10.0))
    assert out.climb_rate == pytest.approx(-0.3)


def test_no_depth_target_centres_the_throttle():
    out = step(make_controller(), Reference(x=1.0), position=(0, 0, -1.5))
    assert out.climb_rate == 0.0


# --------------------------------------------------------------------------
# Horizontal: outer loop
# --------------------------------------------------------------------------


def test_position_error_is_rotated_into_the_body_frame():
    """Facing east in ENU, a target to the north is off the vehicle's left."""
    out = step(make_controller(), Reference(x=0.0, y=0.1))
    assert out.position_error[0] == pytest.approx(0.0, abs=1e-9)
    assert out.position_error[1] == pytest.approx(0.1)
    assert out.sway > 0.0


def test_position_error_rotation_follows_the_vehicle_heading():
    out = step(make_controller(), Reference(x=0.0, y=0.1), yaw=math.radians(90.0))
    # Now facing north, the same target is straight ahead.
    assert out.position_error[0] == pytest.approx(0.1)
    assert out.position_error[1] == pytest.approx(0.0, abs=1e-9)
    assert out.surge > 0.0


def test_approach_speed_is_capped_far_from_the_target():
    out = step(make_controller(max_speed_xy=0.5), Reference(x=30.0, y=40.0))
    assert float(np.linalg.norm(out.desired_velocity)) == pytest.approx(0.5)


def test_approach_speed_tapers_off_close_in():
    out = step(make_controller(kp_position=0.8), Reference(x=0.2, y=0.0))
    assert out.desired_velocity[0] == pytest.approx(0.16)


def test_velocity_feedforward_is_added_to_the_approach():
    """A moving reference is followed, not just chased."""
    out = step(make_controller(), Reference(x=0.0, y=0.0, velocity=np.array([0.3, 0.0, 0.0])))
    assert out.desired_velocity[0] == pytest.approx(0.3)


# --------------------------------------------------------------------------
# Horizontal: inner loop
# --------------------------------------------------------------------------


def test_already_at_commanded_speed_needs_no_correction():
    controller = make_controller(kp_position=1.0)
    out = step(controller, Reference(x=0.2, y=0.0), velocity=(0.2, 0.0))
    assert out.surge == pytest.approx(0.0)


def test_horizontal_thrust_is_a_fraction_of_full_stick():
    out = step(make_controller(), Reference(x=50.0, y=0.0), velocity=(-2.0, 0.0))
    assert out.surge == pytest.approx(1.0)


def test_integral_removes_a_standing_offset():
    """A steady push (trim, current) is cancelled by the integral over time."""
    controller = make_controller()
    reference = Reference(x=0.0, y=0.0)
    # The vehicle drifts backward at a constant 0.05 m/s while held at x=0.
    first = step(controller, reference, velocity=(-0.05, 0.0)).surge
    for _ in range(100):
        last = step(controller, reference, velocity=(-0.05, 0.0)).surge
    assert last > first > 0.0


def test_integral_does_not_wind_up_against_the_limit():
    axis = Axis(kp=1.0, ki=1.0, output_limit=1.0, integral_limit=10.0)
    for _ in range(1000):
        axis.step(5.0, 0.05)      # far beyond what the output can deliver
    # On reversal the output must come off the limit at once, not after the
    # stored integral has been unwound.
    assert axis.step(-0.5, 0.05) < 1.0


def test_lost_velocity_aiding_centres_horizontal_but_keeps_depth_and_yaw():
    controller = make_controller()
    out = step(controller, Reference(x=5.0, y=0.0, z=-1.0, yaw=0.5),
               position=(0, 0, -2.0), horizontal_ok=False)
    assert out.surge == 0.0
    assert out.sway == 0.0
    assert out.climb_rate > 0.0
    assert out.yaw_rate > 0.0


# --------------------------------------------------------------------------
# Hold and arrival
# --------------------------------------------------------------------------


def test_holding_in_place_commands_nothing():
    """A reference equal to the current pose is a hold: every stick centred."""
    out = step(make_controller(), Reference(x=1.0, y=2.0, z=-1.5, yaw=0.3),
               position=(1.0, 2.0, -1.5), yaw=0.3)
    assert out.surge == pytest.approx(0.0)
    assert out.sway == pytest.approx(0.0)
    assert out.climb_rate == pytest.approx(0.0)
    assert out.yaw_rate == pytest.approx(0.0)
    assert out.at_target


def test_not_at_target_while_depth_is_off():
    out = step(make_controller(), Reference(x=0.0, y=0.0, z=-1.0, yaw=0.0),
               position=(0, 0, -1.5))
    assert not out.at_target


def test_dt_of_zero_is_harmless():
    out = step(make_controller(), Reference(x=1.0), dt=0.0)
    assert out.surge == 0.0
