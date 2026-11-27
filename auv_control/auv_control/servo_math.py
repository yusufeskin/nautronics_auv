"""Pure image-based visual-servoing math used by visual_servoing_action (no ROS imports)."""

import math

import numpy as np

VELOCITY_LIMIT = 0.05
# Closest depth (m) the servo loop accepts. The PnP solvers publish -1.0 when
# solvePnP fails; 0, negative and non-finite depths must never reach
# build_interaction (1/z flips sign or divides by zero).
MIN_VALID_DEPTH = 0.01


def is_valid_depth(z):
    """Return True if z is a finite depth in metres usable by build_interaction."""
    try:
        z = float(z)
    except (TypeError, ValueError):
        return False
    return math.isfinite(z) and z >= MIN_VALID_DEPTH


def build_interaction(current_points, target_points, z, fx, fy, cu, cv):
    """
    Stack the 4-point interaction matrix and the feature error.

    current_points / target_points are 4 (u, v) pixel pairs; z is the depth
    used for every point. Columns of the returned (8, 4) matrix are
    camera vx, vy, vz and the rotation about the camera y axis.
    Returns (L_total, e_total) with e_total of shape (8, 1).
    """
    L_stacked = []
    error_stacked = []

    for i in range(4):
        curr_u, curr_v = current_points[i]
        x = (curr_u - cu) / fx
        y = (curr_v - cv) / fy

        tar_u, tar_v = target_points[i]
        x_star = (tar_u - cu) / fx
        y_star = (tar_v - cv) / fy

        L_i = np.array([
            [-1/z,    0,      x/z,   -(1 + x**2)],
            [0,     -1/z,    y/z,   -x * y]
        ])

        e_i = np.array([[x - x_star], [y - y_star]])

        L_stacked.append(L_i)
        error_stacked.append(e_i)
    # vertical stacking
    return np.vstack(L_stacked), np.vstack(error_stacked)


def servo_velocity(L_total, e_total, w_yaw_val,
                   lambda_sway, lambda_heave, lambda_surge,
                   limit=VELOCITY_LIMIT):
    """
    Hybrid visual-servoing translation command with a feed-forward yaw rate.

    Returns clipped (v_surge, v_sway, v_heave, v_yaw) before the node applies
    its body-frame sign conventions.
    """
    L_v = L_total[:, 0:3]
    L_w = L_total[:, 3:]

    L_v_inv = np.linalg.pinv(L_v)
    # hybrid visual servoing (https://inria.hal.science/inria-00350638v1/document)
    # v = -lambda * L_v_inv * (error - L_w * w_yaw)
    compensated_error = e_total - (L_w * w_yaw_val)
    v_target_raw = np.dot(L_v_inv, compensated_error).flatten()

    v_sway = np.clip(-lambda_sway * v_target_raw[0], -limit, limit)
    v_heave = np.clip(-lambda_heave * v_target_raw[1], -limit, limit)
    v_surge = np.clip(-lambda_surge * v_target_raw[2], -limit, limit)
    v_yaw = np.clip(w_yaw_val, -limit, limit)
    return v_surge, v_sway, v_heave, v_yaw
