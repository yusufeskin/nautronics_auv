"""Cascaded pose controller tests.

The behaviour being pinned here is the one described on the whiteboard: turn
until the heading error reaches zero, then stop; move toward a position at a
sensible speed rather than lunging at it.
"""

import math

import numpy as np
import pytest

from auv_actuation.control_law import Axis, CascadeGains, PoseController, PoseTarget


def make_controller(**overrides):
    gains = CascadeGains(
        kp_position=0.8,
        kp_yaw=1.2,
        max_speed_xy=0.5,
        max_speed_z=0.3,
        max_yaw_rate=0.6,
        surge=Axis(kp=20.0, ki=2.0, kd=0.0, output_limit=40.0, integral_limit=20.0),
        sway=Axis(kp=20.0, ki=2.0, kd=0.0, output_limit=30.0, integral_limit=15.0),
        heave=Axis(kp=20.0, ki=2.0, kd=0.0, output_limit=40.0, integral_limit=20.0),
        yaw_rate=Axis(kp=5.0, ki=0.5, kd=0.0, output_limit=8.0, integral_limit=4.0),
    )
    for key, value in overrides.items():
        setattr(gains, key, value)
    return PoseController(gains=gains)


# --------------------------------------------------------------------------
# Outer loop: heading
# --------------------------------------------------------------------------


def test_heading_error_produces_a_turn_in_the_right_direction():
    controller = make_controller()
    target = PoseTarget(yaw=math.radians(45.0))
    desired, _, yaw_error = controller.desired_velocity(target, [0, 0, 0], 0.0)
    assert yaw_error == pytest.approx(math.radians(45.0))
    assert desired[3] > 0.0     # positive yaw rate turns left in FLU


def test_turning_stops_when_the_heading_error_reaches_zero():
    """The whole manoeuvre: command until error is zero, then stop asking."""
    controller = make_controller()
    target = PoseTarget(yaw=math.radians(45.0))
    desired, _, _ = controller.desired_velocity(target, [0, 0, 0], math.radians(45.0))
    assert desired[3] == pytest.approx(0.0)


def test_heading_error_takes_the_short_way_round():
    """Turning from 170 to -170 degrees is 20 degrees, not 340."""
    controller = make_controller()
    target = PoseTarget(yaw=math.radians(-170.0))
    desired, _, yaw_error = controller.desired_velocity(
        target, [0, 0, 0], math.radians(170.0)
    )
    assert yaw_error == pytest.approx(math.radians(20.0), abs=1e-9)
    assert desired[3] > 0.0


def test_yaw_rate_is_capped():
    controller = make_controller(max_yaw_rate=0.3)
    target = PoseTarget(yaw=math.radians(179.0))
    desired, _, _ = controller.desired_velocity(target, [0, 0, 0], 0.0)
    assert desired[3] == pytest.approx(0.3)


def test_no_yaw_target_means_hold_whatever_heading_you_have():
    controller = make_controller()
    desired, _, yaw_error = controller.desired_velocity(
        PoseTarget(x=1.0), [0, 0, 0], math.radians(30.0)
    )
    assert desired[3] == 0.0
    assert yaw_error == 0.0


# --------------------------------------------------------------------------
# Outer loop: position
# --------------------------------------------------------------------------


def test_position_error_is_rotated_into_the_body_frame():
    """Facing east in ENU, a target to the north is off the vehicle's left."""
    controller = make_controller()
    target = PoseTarget(x=0.0, y=2.0, z=0.0)
    _, error_body, _ = controller.desired_velocity(target, [0, 0, 0], 0.0)
    assert error_body[0] == pytest.approx(0.0, abs=1e-9)
    assert error_body[1] == pytest.approx(2.0)


def test_position_error_rotation_follows_the_vehicle_heading():
    controller = make_controller()
    target = PoseTarget(x=0.0, y=2.0, z=0.0)
    _, error_body, _ = controller.desired_velocity(
        target, [0, 0, 0], math.radians(90.0)
    )
    # Now facing north, the same target is straight ahead.
    assert error_body[0] == pytest.approx(2.0)
    assert error_body[1] == pytest.approx(0.0, abs=1e-9)


def test_approach_speed_is_capped_far_from_the_target():
    """Far away, ask for the maximum safe speed - not a lunge."""
    controller = make_controller(max_speed_xy=0.5)
    desired, _, _ = controller.desired_velocity(
        PoseTarget(x=50.0, y=0.0, z=0.0), [0, 0, 0], 0.0
    )
    assert float(np.linalg.norm(desired[:2])) == pytest.approx(0.5)


def test_approach_speed_tapers_off_close_in():
    controller = make_controller(kp_position=0.8, max_speed_xy=0.5)
    desired, _, _ = controller.desired_velocity(
        PoseTarget(x=0.2, y=0.0, z=0.0), [0, 0, 0], 0.0
    )
    assert desired[0] == pytest.approx(0.16)


def test_vertical_speed_has_its_own_cap():
    controller = make_controller(max_speed_z=0.3)
    desired, _, _ = controller.desired_velocity(
        PoseTarget(z=-20.0), [0, 0, 0], 0.0
    )
    assert desired[2] == pytest.approx(-0.3)


def test_unset_axes_are_left_free():
    controller = make_controller()
    desired, error, _ = controller.desired_velocity(
        PoseTarget(z=2.0), [5.0, 5.0, 0.0], 0.0
    )
    assert desired[0] == 0.0
    assert desired[1] == 0.0
    assert error[2] == pytest.approx(2.0)


# --------------------------------------------------------------------------
# Inner loop
# --------------------------------------------------------------------------


def test_velocity_error_produces_force_in_the_right_direction():
    controller = make_controller()
    force, _ = controller.velocity_to_wrench([0.3, 0, 0, 0], [0.0, 0, 0, 0], dt=0.05)
    assert force[0] > 0.0


def test_matching_velocity_produces_no_proportional_force():
    controller = make_controller()
    controller.gains.surge.ki = 0.0
    force, _ = controller.velocity_to_wrench([0.3, 0, 0, 0], [0.3, 0, 0, 0], dt=0.05)
    assert force[0] == pytest.approx(0.0)


def test_roll_and_pitch_torque_are_always_zero():
    """The autopilot holds those axes and takes a lean angle, not a torque."""
    controller = make_controller()
    _, torque = controller.velocity_to_wrench([1, 1, 1, 1], [0, 0, 0, 0], dt=0.05)
    assert torque[0] == 0.0
    assert torque[1] == 0.0


def test_force_is_limited():
    controller = make_controller()
    force, _ = controller.velocity_to_wrench([100.0, 0, 0, 0], [0, 0, 0, 0], dt=0.05)
    assert force[0] == pytest.approx(40.0)


def test_integral_accumulates_a_steady_error():
    """This is what removes buoyancy trim and a standing current."""
    controller = make_controller()
    controller.gains.surge.kp = 0.0
    first, _ = controller.velocity_to_wrench([0.2, 0, 0, 0], [0, 0, 0, 0], dt=0.05)
    for _ in range(10):
        latest, _ = controller.velocity_to_wrench([0.2, 0, 0, 0], [0, 0, 0, 0], dt=0.05)
    assert latest[0] > first[0]


def test_integral_freezes_while_the_actuator_is_saturated():
    """Integrating against a limit stores error the vehicle never received."""
    controller = make_controller()
    controller.gains.surge.kp = 0.0
    for _ in range(20):
        controller.velocity_to_wrench([0.2, 0, 0, 0], [0, 0, 0, 0], dt=0.05,
                                      saturated=True)
    frozen, _ = controller.velocity_to_wrench([0.2, 0, 0, 0], [0, 0, 0, 0], dt=0.05,
                                              saturated=True)
    assert frozen[0] == pytest.approx(0.0)


def test_reset_clears_the_integrators():
    controller = make_controller()
    controller.gains.surge.kp = 0.0
    for _ in range(20):
        controller.velocity_to_wrench([0.2, 0, 0, 0], [0, 0, 0, 0], dt=0.05)
    controller.reset()
    after, _ = controller.velocity_to_wrench([0.2, 0, 0, 0], [0, 0, 0, 0], dt=0.05)
    assert after[0] == pytest.approx(0.2 * 0.05 * controller.gains.surge.ki)


def test_zero_dt_produces_no_output():
    controller = make_controller()
    force, _ = controller.velocity_to_wrench([1, 1, 1, 1], [0, 0, 0, 0], dt=0.0)
    np.testing.assert_allclose(force, np.zeros(3))


# --------------------------------------------------------------------------
# Both loops, and arrival
# --------------------------------------------------------------------------


def test_at_target_is_false_while_still_approaching():
    controller = make_controller()
    output = controller.step(
        PoseTarget(x=5.0, y=0.0, z=0.0, yaw=0.0),
        [0, 0, 0], 0.0, [0, 0, 0, 0], dt=0.05,
    )
    assert not output.at_target


def test_at_target_is_true_inside_both_tolerances():
    controller = make_controller()
    output = controller.step(
        PoseTarget(x=0.02, y=0.0, z=0.0, yaw=math.radians(1.0)),
        [0, 0, 0], 0.0, [0, 0, 0, 0], dt=0.05,
    )
    assert output.at_target


def test_position_within_tolerance_but_heading_wrong_is_not_arrived():
    controller = make_controller()
    output = controller.step(
        PoseTarget(x=0.02, y=0.0, z=0.0, yaw=math.radians(40.0)),
        [0, 0, 0], 0.0, [0, 0, 0, 0], dt=0.05,
    )
    assert not output.at_target


def test_a_pure_turn_commands_torque_but_no_force():
    """'Turn to this heading and stay put' is the common mission primitive."""
    controller = make_controller()
    output = controller.step(
        PoseTarget(x=0.0, y=0.0, z=0.0, yaw=math.radians(45.0)),
        [0, 0, 0], 0.0, [0, 0, 0, 0], dt=0.05,
    )
    np.testing.assert_allclose(output.force, np.zeros(3), atol=1e-9)
    assert output.torque[2] > 0.0
