"""Unit tests for the image-based visual-servoing math."""

import numpy as np
import pytest

from auv_control.servo_math import (
    MIN_VALID_DEPTH, build_interaction, is_valid_depth, servo_velocity)

FX = FY = 500.0
CU, CV = 320.0, 240.0
# Same gains as VisualServoingActionServer
LAMBDA_SURGE, LAMBDA_SWAY, LAMBDA_HEAVE = 0.2, 0.18, 0.3


def square(half, du=0.0, dv=0.0):
    """TL, TR, BR, BL corners of a square centred on the principal point (+offset)."""
    return [
        (CU - half + du, CV - half + dv),
        (CU + half + du, CV - half + dv),
        (CU + half + du, CV + half + dv),
        (CU - half + du, CV + half + dv),
    ]


def command(current, target, z, w_yaw=0.0):
    L, e = build_interaction(current, target, z, FX, FY, CU, CV)
    return servo_velocity(L, e, w_yaw, LAMBDA_SWAY, LAMBDA_HEAVE, LAMBDA_SURGE)


def test_interaction_matrix_shape_and_point_at_principal_point():
    pts = [(CU, CV)] * 4
    L, e = build_interaction(pts, pts, 2.0, FX, FY, CU, CV)
    assert L.shape == (8, 4)
    assert e.shape == (8, 1)
    np.testing.assert_allclose(L[0], [-0.5, 0.0, 0.0, -1.0])
    np.testing.assert_allclose(L[1], [0.0, -0.5, 0.0, 0.0])
    assert not e.any()


def test_feature_error_is_in_normalized_coordinates():
    _, e = build_interaction(square(50, du=5.0), square(50), 2.0, FX, FY, CU, CV)
    # 5 px / fx = 0.01 on every u row, 0 on every v row
    np.testing.assert_allclose(e[0::2, 0], [0.01] * 4)
    np.testing.assert_allclose(e[1::2, 0], [0.0] * 4, atol=1e-15)


def test_zero_error_gives_zero_command():
    surge, sway, heave, yaw = command(square(50), square(50), 2.0)
    assert (surge, sway, heave, yaw) == (0.0, 0.0, 0.0, 0.0)


def test_horizontal_offset_drives_sway_only():
    # pure image shift dx = 0.01 is explained exactly by camera vx = -z * dx = -0.02
    surge, sway, heave, yaw = command(square(50, du=5.0), square(50), 2.0)
    assert sway == pytest.approx(LAMBDA_SWAY * 0.02)
    assert surge == pytest.approx(0.0, abs=1e-12)
    assert heave == pytest.approx(0.0, abs=1e-12)
    assert yaw == 0.0


def test_vertical_offset_drives_heave_only():
    surge, sway, heave, _ = command(square(50, dv=-5.0), square(50), 2.0)
    assert heave == pytest.approx(-LAMBDA_HEAVE * 0.02)
    assert sway == pytest.approx(0.0, abs=1e-12)
    assert surge == pytest.approx(0.0, abs=1e-12)


def test_object_too_large_backs_off():
    # current square 10% bigger than target -> camera vz = z (s - 1) / s
    z, s = 2.0, 1.1
    surge, sway, heave, _ = command(square(55), square(50), z)
    assert surge == pytest.approx(-LAMBDA_SURGE * z * (s - 1) / s)
    assert surge < 0
    assert sway == pytest.approx(0.0, abs=1e-12)
    assert heave == pytest.approx(0.0, abs=1e-12)


def test_commands_are_clipped_to_limit():
    surge, sway, heave, yaw = command(square(150, du=200.0, dv=200.0), square(50), 5.0, 0.3)
    for v in (surge, sway, heave, yaw):
        assert abs(v) <= 0.05
    assert yaw == 0.05
    assert abs(sway) == pytest.approx(0.05)


@pytest.mark.parametrize('z', [
    -1.0,          # what the PnP solvers publish when solvePnP fails
    0.0,           # would divide by zero in build_interaction
    -0.5,          # negative depth flips the sign of every command
    MIN_VALID_DEPTH / 2,
    float('nan'),
    float('inf'),
    float('-inf'),
    None,
])
def test_invalid_depths_are_rejected(z):
    assert not is_valid_depth(z)


@pytest.mark.parametrize('z', [MIN_VALID_DEPTH, 0.5, 1.5, 12.0, np.float32(2.0)])
def test_valid_depths_are_accepted(z):
    assert is_valid_depth(z)


def test_negative_depth_would_reverse_the_command():
    """Why the guard matters: z = -1 drives the vehicle the wrong way."""
    current, target = square(40, du=30.0), square(40)
    good = command(current, target, 1.0)
    bad = command(current, target, -1.0)
    assert np.sign(bad[1]) == -np.sign(good[1]) != 0  # sway flips
