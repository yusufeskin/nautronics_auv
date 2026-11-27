"""Pure PnP geometry helpers (no ROS / OpenCV imports)."""

import numpy as np


def bbox_corners(cx, cy, w, h):
    """
    Return the 4 image corners of an axis-aligned bounding box.

    Order matches OBJECT_REGISTRY: top-left, top-right, bottom-right,
    bottom-left (image y grows downwards). dtype is float32 for cv2.solvePnP.
    """
    return np.array([
        [cx - w / 2.0, cy - h / 2.0],
        [cx + w / 2.0, cy - h / 2.0],
        [cx + w / 2.0, cy + h / 2.0],
        [cx - w / 2.0, cy + h / 2.0]
    ], dtype=np.float32)


def scale_camera_matrix(camera_matrix, width, height, target_size=640.0):
    """
    Scale fx, fy, cx, cy in place from width x height to a square target size.

    Returns the same (modified) matrix for convenience.
    """
    scale_x = target_size / width
    scale_y = target_size / height

    camera_matrix[0, 0] *= scale_x  # fx
    camera_matrix[1, 1] *= scale_y  # fy
    camera_matrix[0, 2] *= scale_x  # cx
    camera_matrix[1, 2] *= scale_y  # cy
    return camera_matrix
