"""Wrench-to-ArduSub-demand mapping tests.

This mapping is the boundary between our control stack and the autopilot, so
the sign conventions and the safe defaults are pinned here. A sign error in
this file makes the vehicle strafe the wrong way, and it will look like a
control tuning problem for a week.
"""

import numpy as np
import pytest

from auv_actuation.demand_mapping import DemandLimits, WrenchDemandMapper

FORCE_SCALE = 40.0
TORQUE_SCALE = 8.0


def make_mapper(**overrides):
    limits = DemandLimits(
        surge_full_scale_n=FORCE_SCALE,
        sway_full_scale_n=FORCE_SCALE,
        heave_full_scale_n=FORCE_SCALE,
        yaw_full_scale_nm=TORQUE_SCALE,
    )
    for key, value in overrides.items():
        setattr(limits, key, value)
    return WrenchDemandMapper(limits=limits)


# --------------------------------------------------------------------------
# Sign conventions: this stack is FLU, ArduSub's pilot axes are FRD
# --------------------------------------------------------------------------


def test_forward_force_commands_positive_surge():
    demand = make_mapper().map([FORCE_SCALE, 0, 0], [0, 0, 0])
    assert demand.x == 1000


def test_leftward_force_commands_negative_sway():
    """FLU +y is left; ArduSub's +y is right."""
    demand = make_mapper().map([0, FORCE_SCALE, 0], [0, 0, 0])
    assert demand.y == -1000


def test_upward_force_commands_z_above_neutral():
    """FLU +z is up; ArduSub's throttle axis rises with the stick."""
    demand = make_mapper(neutral_z=500).map([0, 0, FORCE_SCALE], [0, 0, 0])
    assert demand.z > 500


def test_downward_force_commands_z_below_neutral():
    demand = make_mapper(neutral_z=500).map([0, 0, -FORCE_SCALE], [0, 0, 0])
    assert demand.z < 500


def test_counter_clockwise_torque_commands_negative_yaw():
    """FLU +Mz turns left; ArduSub's yaw axis is clockwise-positive."""
    demand = make_mapper().map([0, 0, 0], [0, 0, TORQUE_SCALE])
    assert demand.r == -1000


def test_half_scale_force_gives_half_deflection():
    demand = make_mapper().map([FORCE_SCALE / 2, 0, 0], [0, 0, 0])
    assert demand.x == pytest.approx(500, abs=1)


# --------------------------------------------------------------------------
# Neutral is "hold", not "off"
# --------------------------------------------------------------------------


def test_neutral_holds_depth_rather_than_cutting_thrust():
    """On a positively buoyant vehicle, cutting thrust means surfacing."""
    demand = make_mapper(neutral_z=500).neutral()
    assert (demand.x, demand.y, demand.r) == (0, 0, 0)
    assert demand.z == 500
    assert not demand.saturated


def test_zero_wrench_maps_to_neutral():
    demand = make_mapper(neutral_z=500).map([0, 0, 0], [0, 0, 0])
    assert (demand.x, demand.y, demand.z, demand.r) == (0, 0, 500, 0)


def test_non_finite_wrench_falls_back_to_neutral():
    demand = make_mapper().map([float("nan"), 0, 0], [0, 0, 0])
    assert demand.z == 500
    assert demand.x == 0


# --------------------------------------------------------------------------
# Saturation and anti-windup reporting
# --------------------------------------------------------------------------


def test_excessive_force_is_clipped_and_flagged():
    demand = make_mapper().map([FORCE_SCALE * 3, 0, 0], [0, 0, 0])
    assert demand.x == 1000
    assert demand.saturated


def test_realised_wrench_reports_only_what_will_be_delivered():
    """The controller stops integrating on the gap between asked and got."""
    mapper = make_mapper()
    demand = mapper.map([FORCE_SCALE * 3, 0, 0], [0, 0, 0])
    assert demand.realised_force[0] == pytest.approx(FORCE_SCALE)


def test_realised_wrench_round_trips_within_quantisation():
    mapper = make_mapper()
    asked = np.array([12.0, -7.0, 5.0])
    demand = mapper.map(asked, [0, 0, 3.0])
    np.testing.assert_allclose(demand.realised_force, asked, atol=0.1)
    assert demand.realised_torque[2] == pytest.approx(3.0, abs=0.05)
    assert not demand.saturated


def test_z_demand_is_clamped_to_the_valid_range():
    demand = make_mapper(neutral_z=500).map([0, 0, FORCE_SCALE * 5], [0, 0, 0])
    assert 0 <= demand.z <= 1000


# --------------------------------------------------------------------------
# The axes ArduSub will not accept
# --------------------------------------------------------------------------


def test_roll_and_pitch_torque_are_reported_as_undelivered():
    """STABILIZE and ALT_HOLD take a lean angle, not a torque."""
    demand = make_mapper().map([0, 0, 0], [5.0, 0, 0])
    assert demand.dropped_roll_pitch
    assert demand.realised_torque[0] == 0.0
    assert demand.realised_torque[1] == 0.0


def test_tiny_roll_pitch_torque_is_not_worth_reporting():
    """Numerical dust from a controller must not raise a warning."""
    demand = make_mapper().map([0, 0, 0], [1e-6, 1e-6, 0])
    assert not demand.dropped_roll_pitch


def test_yaw_torque_survives_alongside_dropped_roll_pitch():
    demand = make_mapper().map([0, 0, 0], [5.0, 5.0, TORQUE_SCALE / 2])
    assert demand.dropped_roll_pitch
    assert demand.r == pytest.approx(-500, abs=1)


# --------------------------------------------------------------------------
# Deadband
# --------------------------------------------------------------------------


def test_deadband_suppresses_demands_below_arduSubs_own_stick_deadband():
    mapper = make_mapper(deadband_counts=50)
    demand = mapper.map([FORCE_SCALE * 0.01, 0, 0], [0, 0, 0])
    assert demand.x == 0


def test_deadband_leaves_larger_demands_alone():
    mapper = make_mapper(deadband_counts=50)
    demand = mapper.map([FORCE_SCALE * 0.5, 0, 0], [0, 0, 0])
    assert demand.x == pytest.approx(500, abs=1)


def test_zero_full_scale_is_treated_as_an_unusable_axis():
    """An uncalibrated axis must produce nothing, not a divide by zero."""
    mapper = make_mapper(sway_full_scale_n=0.0)
    demand = mapper.map([0, 10.0, 0], [0, 0, 0])
    assert demand.y == 0
