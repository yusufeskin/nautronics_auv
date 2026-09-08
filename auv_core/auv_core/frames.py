"""The one place frame conventions are converted. Nowhere else.

Two conventions meet in this vehicle and mixing them up is the single most
expensive class of bug in an AUV stack, because the symptoms (a heading that
drifts the wrong way, a sway command that pushes the wrong direction, a
covariance that is transposed) look like tuning problems for weeks:

  ROS side     REP-103/105: world is ENU (x East, y North, z Up),
               body is FLU (x Forward, y Left, z Up).
  MAVLink /    world is NED (x North, y East, z Down),
  autopilot    body is FRD (x Forward, y Right, z Down).
  DVL          Water Linked A50 reports body velocity in FRD.

The rule this module exists to enforce: every ROS topic in this stack is ENU
and FLU. Conversions happen exactly twice - once in the sensor gate that
brings an FRD sensor in, once in the bridge that talks to the autopilot - and
both call into here. If you find yourself writing a sign flip anywhere else,
that is the bug.

Quaternions are numpy arrays in ROS order ``(x, y, z, w)``, matching
``geometry_msgs/Quaternion``, so a value read off a message needs no
reordering. Vectors are numpy arrays of shape ``(3,)``.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "ENU_NED",
    "FLU_FRD",
    "IDENTITY_QUAT",
    "euler_rpy_from_quat",
    "flip_body_convention",
    "flip_body_quat",
    "flip_world_convention",
    "flip_world_quat",
    "heading_ned_to_yaw_enu",
    "quat_conjugate",
    "quat_delta_to_rotvec",
    "quat_from_euler_rpy",
    "quat_multiply",
    "quat_normalize",
    "quat_to_rot",
    "rot_to_quat",
    "rotate_covariance",
    "wrap_angle",
    "yaw_enu_to_heading_ned",
]


# ENU <-> NED: swap x and y, negate z. A 180 degree rotation about the axis
# (1, 1, 0)/sqrt(2), so it is a proper rotation and its own inverse.
ENU_NED = np.array(
    [
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
    ]
)

# FLU <-> FRD: keep forward, negate left and up. A 180 degree rotation about
# the body x axis, also a proper rotation and its own inverse.
FLU_FRD = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
    ]
)

IDENTITY_QUAT = np.array([0.0, 0.0, 0.0, 1.0])


# --------------------------------------------------------------------------
# Vectors
# --------------------------------------------------------------------------


def flip_world_convention(v) -> np.ndarray:
    """Convert a world-frame vector between ENU and NED.

    The transform is an involution, so one function serves both directions.
    Applies to positions, velocities and accelerations expressed in the
    world frame.
    """
    return ENU_NED @ np.asarray(v, dtype=float)


def flip_body_convention(v) -> np.ndarray:
    """Convert a body-frame vector between FLU and FRD.

    Also an involution. This is the conversion a DVL gate applies to the
    velocity the A50 reports.
    """
    return FLU_FRD @ np.asarray(v, dtype=float)


# --------------------------------------------------------------------------
# Orientations
# --------------------------------------------------------------------------


def flip_world_quat(q) -> np.ndarray:
    """Reinterpret an attitude whose *world* frame changes ENU <-> NED.

    Given ``q`` describing body->ENU, returns the quaternion describing the
    same physical attitude as body->NED, leaving the body convention alone.
    Rarely what you want on its own - see :func:`flip_body_quat` for the
    body half, and note that a full ROS<->MAVLink attitude conversion needs
    both.
    """
    rot = ENU_NED @ quat_to_rot(q)
    return rot_to_quat(rot)


def flip_body_quat(q) -> np.ndarray:
    """Reinterpret an attitude whose *body* frame changes FLU <-> FRD.

    Given ``q`` describing FLU->world, returns the quaternion describing
    FRD->the same world.
    """
    rot = quat_to_rot(q) @ FLU_FRD
    return rot_to_quat(rot)


def yaw_enu_to_heading_ned(yaw: float) -> float:
    """ENU yaw (counter-clockwise from East) -> NED heading (clockwise from North).

    Involutive, so :func:`heading_ned_to_yaw_enu` is the same arithmetic; both
    names exist so call sites read correctly.
    """
    return wrap_angle(0.5 * math.pi - yaw)


def heading_ned_to_yaw_enu(heading: float) -> float:
    """NED heading (clockwise from North) -> ENU yaw (counter-clockwise from East)."""
    return wrap_angle(0.5 * math.pi - heading)


# --------------------------------------------------------------------------
# Quaternion algebra
# --------------------------------------------------------------------------


def quat_normalize(q) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    norm = float(np.linalg.norm(q))
    if norm < 1e-12:
        return IDENTITY_QUAT.copy()
    return q / norm


def quat_conjugate(q) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    return np.array([-q[0], -q[1], -q[2], q[3]])


def quat_multiply(q1, q2) -> np.ndarray:
    """Hamilton product ``q1 * q2``, i.e. apply q2 first, then q1."""
    x1, y1, z1, w1 = np.asarray(q1, dtype=float)
    x2, y2, z2, w2 = np.asarray(q2, dtype=float)
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ]
    )


def quat_delta_to_rotvec(q_prev, q_now) -> np.ndarray:
    """Rotation from ``q_prev`` to ``q_now`` as a body-frame rotation vector.

    This replaces differencing Euler angles, which is what makes an attitude
    delta blow up near pitch = +/-90 degrees - exactly the attitude a RoboSub
    vehicle passes through when it rolls or pitches through the gate for style
    points. The returned vector is axis * angle in radians, always taking the
    shorter arc, and behaves smoothly through any orientation.
    """
    dq = quat_multiply(quat_conjugate(quat_normalize(q_prev)), quat_normalize(q_now))
    # Pick the hemisphere with the shorter rotation; q and -q are the same
    # attitude but differ by 2*pi in the extracted angle.
    if dq[3] < 0.0:
        dq = -dq
    vec = dq[:3]
    sin_half = float(np.linalg.norm(vec))
    if sin_half < 1e-12:
        # Below this the small-angle form is both accurate and singularity
        # free; atan2 would divide by ~0.
        return 2.0 * vec
    angle = 2.0 * math.atan2(sin_half, float(dq[3]))
    return vec / sin_half * angle


def quat_to_rot(q) -> np.ndarray:
    x, y, z, w = quat_normalize(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def rot_to_quat(rot) -> np.ndarray:
    """Rotation matrix -> quaternion (x, y, z, w), Shepperd's method.

    Branching on the largest diagonal term keeps the divisor away from zero,
    which naive trace-only extraction does not for rotations near 180 degrees.
    """
    m = np.asarray(rot, dtype=float)
    trace = m[0, 0] + m[1, 1] + m[2, 2]

    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s

    return quat_normalize(np.array([x, y, z, w]))


# --------------------------------------------------------------------------
# Euler angles - for configuration files and human-readable logs only
# --------------------------------------------------------------------------


def quat_from_euler_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Intrinsic Z-Y-X (yaw, then pitch, then roll), the ROS/URDF convention.

    Use this to read mount orientations out of YAML. Do not use Euler angles
    to carry attitude between nodes or to difference two attitudes - see
    :func:`quat_delta_to_rotvec`.
    """
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return np.array(
        [
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ]
    )


def euler_rpy_from_quat(q) -> tuple[float, float, float]:
    """Inverse of :func:`quat_from_euler_rpy`. Degenerate at pitch = +/-90."""
    x, y, z, w = quat_normalize(q)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


# --------------------------------------------------------------------------
# Misc
# --------------------------------------------------------------------------


def wrap_angle(a: float) -> float:
    """Wrap to (-pi, pi]."""
    return math.atan2(math.sin(a), math.cos(a))


def rotate_covariance(rot, cov) -> np.ndarray:
    """Rotate a 3x3 covariance into another frame: ``R C R^T``.

    Rotating the mean of a measurement without rotating its covariance is a
    quiet way to feed a filter a wrong uncertainty, so every place that
    converts a vector's frame should convert its covariance here too.
    """
    r = np.asarray(rot, dtype=float)
    c = np.asarray(cov, dtype=float).reshape(3, 3)
    return r @ c @ r.T
