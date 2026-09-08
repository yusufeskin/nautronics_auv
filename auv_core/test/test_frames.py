"""Frame conversion tests.

These are cheap and they guard the most expensive bug class in the stack, so
they run in CI on every push. A failure here means something downstream is
silently steering the wrong way.
"""

import math

import numpy as np
import pytest

from auv_core import frames


# --------------------------------------------------------------------------
# The two convention transforms are proper rotations and involutions
# --------------------------------------------------------------------------


@pytest.mark.parametrize("matrix", [frames.ENU_NED, frames.FLU_FRD])
def test_convention_transforms_are_proper_rotations(matrix):
    np.testing.assert_allclose(matrix @ matrix.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(matrix) == pytest.approx(1.0)


@pytest.mark.parametrize("flip", [frames.flip_world_convention, frames.flip_body_convention])
def test_convention_flips_are_involutions(flip):
    rng = np.random.default_rng(0)
    for _ in range(20):
        v = rng.normal(size=3)
        np.testing.assert_allclose(flip(flip(v)), v, atol=1e-12)


def test_enu_to_ned_known_axes():
    # East in ENU is +x; in NED East is +y.
    np.testing.assert_allclose(frames.flip_world_convention([1, 0, 0]), [0, 1, 0], atol=1e-12)
    # North in ENU is +y; in NED North is +x.
    np.testing.assert_allclose(frames.flip_world_convention([0, 1, 0]), [1, 0, 0], atol=1e-12)
    # Up in ENU is +z; in NED that is -z (Down positive).
    np.testing.assert_allclose(frames.flip_world_convention([0, 0, 1]), [0, 0, -1], atol=1e-12)


def test_flu_to_frd_known_axes():
    np.testing.assert_allclose(frames.flip_body_convention([1, 0, 0]), [1, 0, 0], atol=1e-12)
    # Left in FLU is +y; in FRD that is -y (Right positive).
    np.testing.assert_allclose(frames.flip_body_convention([0, 1, 0]), [0, -1, 0], atol=1e-12)
    np.testing.assert_allclose(frames.flip_body_convention([0, 0, 1]), [0, 0, -1], atol=1e-12)


# --------------------------------------------------------------------------
# Quaternion algebra
# --------------------------------------------------------------------------


def _random_quats(n, seed=1):
    rng = np.random.default_rng(seed)
    return [frames.quat_normalize(rng.normal(size=4)) for _ in range(n)]


def test_quat_rot_round_trip():
    for q in _random_quats(50):
        recovered = frames.rot_to_quat(frames.quat_to_rot(q))
        # q and -q are the same rotation, so compare rotation matrices.
        np.testing.assert_allclose(
            frames.quat_to_rot(recovered), frames.quat_to_rot(q), atol=1e-10
        )


def test_rot_to_quat_handles_all_branches():
    # 180 degree rotations about each axis drive the trace negative and hit
    # the three non-trace branches of Shepperd's method; naive extraction
    # divides by ~0 here.
    for axis in range(3):
        rot = -np.eye(3)
        rot[axis, axis] = 1.0
        q = frames.rot_to_quat(rot)
        np.testing.assert_allclose(frames.quat_to_rot(q), rot, atol=1e-10)


def test_quat_multiply_matches_matrix_product():
    for q1, q2 in zip(_random_quats(20, seed=2), _random_quats(20, seed=3)):
        np.testing.assert_allclose(
            frames.quat_to_rot(frames.quat_multiply(q1, q2)),
            frames.quat_to_rot(q1) @ frames.quat_to_rot(q2),
            atol=1e-10,
        )


def test_quat_conjugate_inverts():
    for q in _random_quats(20, seed=4):
        product = frames.quat_multiply(q, frames.quat_conjugate(q))
        np.testing.assert_allclose(frames.quat_to_rot(product), np.eye(3), atol=1e-10)


# --------------------------------------------------------------------------
# Attitude deltas - the reason Euler differencing is banned
# --------------------------------------------------------------------------


def test_quat_delta_recovers_known_yaw_step():
    q0 = frames.quat_from_euler_rpy(0.0, 0.0, 0.0)
    q1 = frames.quat_from_euler_rpy(0.0, 0.0, math.radians(30.0))
    np.testing.assert_allclose(
        frames.quat_delta_to_rotvec(q0, q1), [0.0, 0.0, math.radians(30.0)], atol=1e-9
    )


def test_quat_delta_is_zero_for_identical_attitudes():
    for q in _random_quats(10, seed=5):
        np.testing.assert_allclose(frames.quat_delta_to_rotvec(q, q), np.zeros(3), atol=1e-12)


def test_quat_delta_takes_the_short_way_round():
    q0 = frames.quat_from_euler_rpy(0.0, 0.0, math.radians(-170.0))
    q1 = frames.quat_from_euler_rpy(0.0, 0.0, math.radians(170.0))
    delta = frames.quat_delta_to_rotvec(q0, q1)
    # The short arc is -20 degrees, not +340.
    assert np.linalg.norm(delta) == pytest.approx(math.radians(20.0), abs=1e-9)
    assert delta[2] < 0.0


def test_quat_delta_stays_finite_through_pitch_ninety():
    """The gate-roll regression: Euler yaw/roll are undefined at pitch=+/-90.

    Stepping a small rotation across that attitude must produce a small,
    finite delta. Differencing Euler triples here produces a discontinuity of
    order pi, which is what would kick the estimator during a style manoeuvre.
    """
    step = math.radians(1.0)
    for offset in (-1e-4, 0.0, 1e-4):
        pitch = 0.5 * math.pi + offset
        q0 = frames.quat_from_euler_rpy(0.0, pitch - 0.5 * step, 0.0)
        q1 = frames.quat_from_euler_rpy(0.0, pitch + 0.5 * step, 0.0)
        delta = frames.quat_delta_to_rotvec(q0, q1)
        assert np.all(np.isfinite(delta))
        assert np.linalg.norm(delta) == pytest.approx(step, abs=1e-6)


def test_quat_delta_composes_back_to_the_new_attitude():
    """delta must be the *body* rotation carrying q_prev onto q_now."""
    for q0, q1 in zip(_random_quats(20, seed=6), _random_quats(20, seed=7)):
        rotvec = frames.quat_delta_to_rotvec(q0, q1)
        angle = float(np.linalg.norm(rotvec))
        if angle < 1e-9:
            continue
        axis = rotvec / angle
        dq = np.array(
            [
                axis[0] * math.sin(angle / 2),
                axis[1] * math.sin(angle / 2),
                axis[2] * math.sin(angle / 2),
                math.cos(angle / 2),
            ]
        )
        np.testing.assert_allclose(
            frames.quat_to_rot(frames.quat_multiply(q0, dq)),
            frames.quat_to_rot(q1),
            atol=1e-9,
        )


# --------------------------------------------------------------------------
# Attitude convention flips
# --------------------------------------------------------------------------


def test_full_attitude_flip_matches_vector_transforms():
    """ROS attitude -> MAVLink attitude must move vectors consistently.

    If q maps FLU body vectors into ENU, then the flipped quaternion must map
    the same vectors, expressed FRD, into NED.
    """
    rng = np.random.default_rng(8)
    for q in _random_quats(20, seed=9):
        q_ned_frd = frames.flip_world_quat(frames.flip_body_quat(q))
        v_flu = rng.normal(size=3)
        v_enu = frames.quat_to_rot(q) @ v_flu
        np.testing.assert_allclose(
            frames.quat_to_rot(q_ned_frd) @ frames.flip_body_convention(v_flu),
            frames.flip_world_convention(v_enu),
            atol=1e-10,
        )


# --------------------------------------------------------------------------
# Heading conventions
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "yaw_deg,heading_deg",
    [
        (0.0, 90.0),     # facing East: ENU yaw 0, NED heading 90
        (90.0, 0.0),     # facing North
        (180.0, -90.0),  # facing West
        (-90.0, 180.0),  # facing South
    ],
)
def test_yaw_heading_conversion_known_values(yaw_deg, heading_deg):
    got = frames.yaw_enu_to_heading_ned(math.radians(yaw_deg))
    assert got == pytest.approx(math.radians(heading_deg), abs=1e-9)


def test_yaw_heading_conversion_is_involutive():
    for deg in range(-180, 181, 7):
        rad = math.radians(deg)
        back = frames.heading_ned_to_yaw_enu(frames.yaw_enu_to_heading_ned(rad))
        assert frames.wrap_angle(back - rad) == pytest.approx(0.0, abs=1e-9)


def test_wrap_angle_range():
    for deg in range(-1000, 1001, 13):
        wrapped = frames.wrap_angle(math.radians(deg))
        assert -math.pi - 1e-12 <= wrapped <= math.pi + 1e-12


# --------------------------------------------------------------------------
# Covariance
# --------------------------------------------------------------------------


def test_rotate_covariance_preserves_trace_and_symmetry():
    rng = np.random.default_rng(10)
    a = rng.normal(size=(3, 3))
    cov = a @ a.T  # symmetric positive semi-definite
    rot = frames.quat_to_rot(_random_quats(1, seed=11)[0])
    rotated = frames.rotate_covariance(rot, cov)
    np.testing.assert_allclose(rotated, rotated.T, atol=1e-10)
    assert np.trace(rotated) == pytest.approx(np.trace(cov))


def test_rotate_covariance_accepts_flat_nine_element_input():
    cov = np.diag([1.0, 2.0, 3.0])
    rotated = frames.rotate_covariance(np.eye(3), cov.reshape(9))
    np.testing.assert_allclose(rotated, cov, atol=1e-12)
