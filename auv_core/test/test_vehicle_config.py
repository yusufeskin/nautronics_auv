"""Vehicle geometry tests.

These assert properties of the physical vehicle that the rest of the stack
takes for granted. If one fails, the fix is in vehicle.yaml or on the
vehicle, not in the code that reads it.
"""

import numpy as np
import pytest

from auv_core.vehicle_config import load_vehicle_config


@pytest.fixture(scope="module")
def config():
    return load_vehicle_config()


def test_required_frames_are_declared(config):
    for key in ("odom", "base_link"):
        assert config.frames[key]


def test_no_map_frame_is_declared(config):
    """There is no source of absolute position on this vehicle.

    A map frame would be an alias for odom pretending to be drift-free, which
    is worse than not having one: code would start assuming a pose means the
    same thing at minute 15 as at minute 0.
    """
    assert "map" not in config.frames


def test_every_layer_one_sensor_has_a_mount(config):
    # The gates in auv_sensors look these up by name at startup.
    for name in ("imu", "dvl", "depth", "heading"):
        mount = config.sensor(name)
        assert mount.frame
        assert mount.xyz.shape == (3,)
        assert mount.rpy.shape == (3,)


def test_sensor_frames_are_unique(config):
    frames = [mount.frame for mount in config.sensors.values()]
    assert len(frames) == len(set(frames))


def test_mount_rotations_are_valid(config):
    for mount in config.sensors.values():
        rot = mount.rotation
        np.testing.assert_allclose(rot @ rot.T, np.eye(3), atol=1e-9)
        assert np.linalg.det(rot) == pytest.approx(1.0)


def test_vehicle_is_fully_actuated(config):
    """Rank 6 means arbitrary attitude while translating.

    This is the hardware precondition for the gate roll/pitch style points
    and for holding heading while strafing during the slalom. If a thruster
    is removed or remounted and this drops to 5, no amount of control tuning
    recovers the lost axis.
    """
    b = config.configuration_matrix()
    assert b.shape == (6, len(config.thrusters))
    assert np.linalg.matrix_rank(b, tol=1e-6) == 6


def test_actuation_is_not_near_singular(config):
    """Rank alone can hide an axis that is technically actuated but weak.

    The smallest singular value is the worst-case wrench-per-unit-thrust; if
    it collapses the allocator will demand huge thrust for a small torque.
    """
    singular_values = np.linalg.svd(config.configuration_matrix(), compute_uv=False)
    assert singular_values[-1] > 0.05


def test_thruster_axes_are_unit_length(config):
    for thruster in config.thrusters:
        axis = thruster.axis(config.thruster_axis_local)
        assert np.linalg.norm(axis) == pytest.approx(1.0)


def test_thruster_ids_are_unique_and_contiguous(config):
    ids = sorted(t.id for t in config.thrusters)
    assert ids == list(range(1, len(ids) + 1))


def test_vehicle_is_slightly_positively_buoyant(config):
    """A dead vehicle must surface on its own. This is a safety rule, and
    RoboSub requires a working kill switch to leave the vehicle recoverable."""
    assert 0.0 < config.net_buoyancy_n < 20.0


def test_centre_of_buoyancy_sits_above_centre_of_mass(config):
    """The separation is what rights the vehicle in roll and pitch."""
    com_z = config.body["com_xyz"][2]
    cob_z = config.body["cob_xyz"][2]
    assert cob_z > com_z


def test_unknown_sensor_lookup_names_the_alternatives(config):
    with pytest.raises(KeyError) as excinfo:
        config.sensor("lidar")
    assert "dvl" in str(excinfo.value)
