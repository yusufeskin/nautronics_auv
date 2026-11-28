"""Unit tests for the object registry and pure PnP geometry helpers."""

import numpy as np
import pytest

from auv_vision.object_config import OBJECT_REGISTRY
from auv_vision.pnp_geometry import bbox_corners, scale_camera_matrix


@pytest.mark.parametrize('cls_id', sorted(OBJECT_REGISTRY))
def test_registry_entry_is_planar_rectangle_in_keypoint_order(cls_id):
    entry = OBJECT_REGISTRY[cls_id]
    assert isinstance(entry['name'], str) and entry['name']
    pts = np.array(entry['points_3d'], dtype=float)
    assert pts.shape == (4, 3)
    assert np.all(pts[:, 2] == 0.0)                  # planar target, z = 0
    tl, tr, br, bl = pts
    # TL -> TR -> BR -> BL with y growing downwards (same as image keypoints)
    assert tr[0] > tl[0] and tr[1] == tl[1]
    assert br[0] == tr[0] and br[1] > tr[1]
    assert bl[0] == tl[0] and bl[1] == br[1]
    width, height = tr[0] - tl[0], br[1] - tr[1]
    assert 0.0 < width < 3.0 and 0.0 < height < 3.0  # plausible metres


def test_registry_names_are_unique():
    names = [entry['name'] for entry in OBJECT_REGISTRY.values()]
    assert len(names) == len(set(names))


def test_bbox_corners_values_and_order():
    corners = bbox_corners(100.0, 50.0, 40.0, 20.0)
    assert corners.dtype == np.float32
    np.testing.assert_array_equal(
        corners, [[80, 40], [120, 40], [120, 60], [80, 60]])


def test_bbox_corners_follow_registry_winding():
    corners = bbox_corners(320.0, 240.0, 60.0, 30.0)
    obj = np.array(OBJECT_REGISTRY[0]['points_3d'])[:, :2]
    # Edge directions of image and object points must agree for solvePnP
    assert np.array_equal(np.sign(np.diff(corners, axis=0)),
                          np.sign(np.diff(obj, axis=0)))


def test_bbox_corners_centroid_is_bbox_center():
    corners = bbox_corners(12.5, 7.5, 5.0, 3.0)
    np.testing.assert_allclose(corners.mean(axis=0), [12.5, 7.5])


def test_scale_camera_matrix_to_640_square():
    k = np.array([[900.0, 0.0, 640.0],
                  [0.0, 900.0, 360.0],
                  [0.0, 0.0, 1.0]])
    out = scale_camera_matrix(k, 1280, 720, 640.0)
    assert out is k                                  # scaled in place
    np.testing.assert_allclose(k, [[450.0, 0.0, 320.0],
                                   [0.0, 800.0, 320.0],
                                   [0.0, 0.0, 1.0]])


def test_scale_camera_matrix_identity_at_target_size():
    k = np.array([[500.0, 0.0, 320.0], [0.0, 510.0, 330.0], [0.0, 0.0, 1.0]])
    expected = k.copy()
    scale_camera_matrix(k, 640, 640)
    np.testing.assert_array_equal(k, expected)
