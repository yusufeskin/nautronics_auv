import math

import pytest

from auv_control.position_control import (
    ControllerParams, PositionController, Reference, SpeedCurve, State,
)
from auv_control.rc_output_node import demand_to_channels

DT = 0.05
SIGNS = dict(surge=1, sway=-1, heave=1, yaw=-1)


def make():
    return PositionController(ControllerParams())


def test_at_reference_gives_zero_demand():
    c = make()
    d = c.compute(State(1, 2, -1, 0.3, 0, 0), Reference(1, 2, -1, 0.3), DT)
    assert (d.surge, d.sway, d.heave, d.yaw) == (0.0, 0.0, 0.0, 0.0)


def test_target_ahead_pushes_forward_only():
    d = make().compute(State(0, 0, 0, 0, 0, 0), Reference(5, 0, 0, 0), DT)
    assert d.surge > 0
    assert d.sway == pytest.approx(0.0)


def test_target_is_rotated_into_body_frame():
    # Facing +y (yaw 90 deg); target along +x world is to the vehicle's right.
    d = make().compute(State(0, 0, 0, math.pi / 2, 0, 0), Reference(5, 0, 0, math.pi / 2), DT)
    assert d.sway < 0
    assert d.surge == pytest.approx(0.0, abs=1e-9)


def test_speed_demand_is_capped():
    p = ControllerParams()
    c = PositionController(p)
    # Moving at max_speed toward a far target: no velocity error, output is FF only.
    d = c.compute(State(0, 0, 0, 0, p.max_speed, 0), Reference(100, 0, 0, 0), DT)
    curve = SpeedCurve(p.surge_curve_stick, p.surge_curve_speed)
    assert d.surge == pytest.approx(curve.stick_for(p.max_speed))


def test_outputs_are_limited():
    p = ControllerParams()
    d = PositionController(p).compute(State(0, 0, 0, 0, -2, 2), Reference(100, -100, 50, 3.0), DT)
    assert abs(d.surge) <= p.max_thrust and abs(d.sway) <= p.max_thrust
    assert abs(d.heave) <= p.max_heave and abs(d.yaw) <= p.max_yaw


def test_yaw_takes_short_way_round():
    # From 170 deg to -170 deg is +20 deg (CCW), not -340.
    d = make().compute(State(0, 0, 0, math.radians(170), 0, 0),
                       Reference(0, 0, 0, math.radians(-170)), DT)
    assert d.yaw > 0


def test_depth_and_yaw_deadbands_centre_the_stick():
    p = ControllerParams()
    d = PositionController(p).compute(
        State(0, 0, -1.0, 0, 0, 0),
        Reference(0, 0, -1.0 + p.depth_deadband / 2, p.yaw_deadband / 2), DT)
    assert d.heave == 0.0 and d.yaw == 0.0


def test_deeper_target_commands_down():
    d = make().compute(State(0, 0, -1, 0, 0, 0), Reference(0, 0, -2, 0), DT)
    assert d.heave < 0


def test_dvl_loss_centres_horizontal_but_keeps_depth_and_yaw():
    d = make().compute(State(0, 0, 0, 0, 0, 0), Reference(5, 5, -1, 1.0), DT, velocity_ok=False)
    assert d.surge == 0.0 and d.sway == 0.0
    assert d.heave != 0.0 and d.yaw != 0.0


def test_integrator_removes_steady_state_velocity_error():
    # Feedforward alone gives 0.1 m/s, but the vehicle is held at 0 (current).
    c = make()
    s, r = State(0, 0, 0, 0, 0.0, 0), Reference(0, 0, 0, 0, vx_ff=0.1)
    first = c.compute(s, r, DT).surge
    for _ in range(100):
        last = c.compute(s, r, DT).surge
    assert last > first


def test_rc_neutral():
    assert demand_to_channels(0, 0, 0, 0, SIGNS) == [1500] * 6


def test_rc_channel_order_and_signs():
    ch = demand_to_channels(surge=1, sway=1, heave=1, yaw=1, signs=SIGNS)
    assert ch[0] == ch[1] == 1500   # pitch, roll untouched
    assert ch[2] == 1900            # heave up    -> throttle up
    assert ch[3] == 1100            # yaw CCW     -> ArduSub yaw left
    assert ch[4] == 1900            # surge fwd   -> forward
    assert ch[5] == 1100            # sway left   -> ArduSub lateral left


DZ = dict(heave=40, yaw=20)


def test_rc_small_demand_starts_outside_deadzone():
    ch = demand_to_channels(0, 0, heave=0.01, yaw=0.01, signs=SIGNS, deadzones=DZ)
    assert ch[2] > 1540      # just past the 40 us throttle deadzone, upward
    assert ch[3] < 1480      # just past the 20 us yaw deadzone, CCW = below 1500


def test_rc_zero_demand_stays_exactly_neutral_with_deadzones():
    assert demand_to_channels(0, 0, 0, 0, SIGNS, deadzones=DZ) == [1500] * 6


def test_rc_full_demand_unchanged_by_deadzone():
    ch = demand_to_channels(0, 0, heave=-1, yaw=-1, signs=SIGNS, deadzones=DZ)
    assert ch[2] == 1100 and ch[3] == 1900


def test_rc_clamps_out_of_range_demand():
    ch = demand_to_channels(5, -5, 0, 0, SIGNS)
    assert ch[4] == 1900 and ch[5] == 1900


def test_speed_curve_hits_measured_points_and_is_odd():
    c = SpeedCurve([0.0, 0.1, 0.4], [0.0, 0.45, 0.95])
    assert c.stick_for(0.45) == pytest.approx(0.1)
    assert c.stick_for(-0.95) == pytest.approx(-0.4)
    assert c.stick_for(0.0) == 0.0
    assert c.stick_for(0.225) == pytest.approx(0.05)          # linear between points
    assert c.stick_for(1.2) == pytest.approx(0.4 + 0.3 * 0.25 / 0.5)  # extrapolated


def test_speed_curve_rejects_bad_tables():
    for stick, speed in (([0.0, 0.1], [0.0]), ([0.1, 0.2], [0.0, 0.5]), ([0.0, 0.2, 0.1], [0.0, 0.3, 0.5])):
        with pytest.raises(ValueError):
            SpeedCurve(stick, speed)


def test_feedforward_uses_measured_curve():
    # 0.25 m/s cruise needs ~0.034 stick in sim, a third of what the old linear gain gave.
    c = make()
    d = c.compute(State(0, 0, 0, 0, 0.25, 0), Reference(0, 0, 0, 0, vx_ff=0.25), DT)
    assert 0.03 < d.surge < 0.04


def test_acceleration_feedforward_pushes_harder_to_speed_up_and_brakes_to_slow_down():
    p = ControllerParams()
    s = State(0, 0, 0, 0, 0.1, 0)
    cruise = PositionController(p).compute(s, Reference(0, 0, 0, 0, vx_ff=0.1), DT).surge
    up = PositionController(p).compute(s, Reference(0, 0, 0, 0, vx_ff=0.1, ax_ff=0.1), DT).surge
    down = PositionController(p).compute(s, Reference(0, 0, 0, 0, vx_ff=0.1, ax_ff=-0.1), DT).surge
    assert up == pytest.approx(cruise + p.vel_ka * 0.1)
    assert down == pytest.approx(cruise - p.vel_ka * 0.1)


def test_acceleration_feedforward_is_rotated_into_body_frame():
    # Facing +y; reference accelerating along +x world = to the vehicle's right.
    p = ControllerParams()
    d = PositionController(p).compute(State(0, 0, 0, math.pi / 2, 0, 0),
                                      Reference(0, 0, 0, math.pi / 2, ax_ff=0.1), DT)
    assert d.sway == pytest.approx(-p.vel_ka * 0.1)
    assert d.surge == pytest.approx(0.0, abs=1e-9)
