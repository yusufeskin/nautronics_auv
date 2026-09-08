"""Typed access to ``config/vehicle.yaml``.

Nodes can read the same file through ROS parameters, but allocation, tests
and offline analysis all need it without a running graph, so the parsing and
the derived geometry live here rather than inside any one node.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import yaml

from auv_core import frames

__all__ = [
    "SensorMount",
    "Thruster",
    "VehicleConfig",
    "default_config_path",
    "load_vehicle_config",
]


@dataclass(frozen=True)
class SensorMount:
    """A sensor's rigid mount, expressed in base_link (FLU)."""

    name: str
    frame: str
    xyz: np.ndarray
    rpy: np.ndarray

    @property
    def quat(self) -> np.ndarray:
        """Mount orientation as (x, y, z, w)."""
        return frames.quat_from_euler_rpy(*self.rpy)

    @property
    def rotation(self) -> np.ndarray:
        """Rotation carrying a vector from the sensor frame into base_link."""
        return frames.quat_to_rot(self.quat)


@dataclass(frozen=True)
class Thruster:
    id: int
    xyz: np.ndarray
    rpy: np.ndarray
    direction_sign: float

    def axis(self, axis_local) -> np.ndarray:
        """Unit thrust direction in base_link for a positive command."""
        rot = frames.quat_to_rot(frames.quat_from_euler_rpy(*self.rpy))
        return self.direction_sign * (rot @ np.asarray(axis_local, dtype=float))

    def wrench_column(self, axis_local) -> np.ndarray:
        """This thruster's column of the configuration matrix: [force; torque]."""
        axis = self.axis(axis_local)
        return np.concatenate([axis, np.cross(self.xyz, axis)])


@dataclass(frozen=True)
class VehicleConfig:
    frames: dict
    body: dict
    sensors: dict          # name -> SensorMount
    thrusters: list        # list[Thruster]
    thruster_axis_local: np.ndarray
    max_forward_n: float
    max_reverse_n: float
    hydrodynamics: dict

    def sensor(self, name: str) -> SensorMount:
        try:
            return self.sensors[name]
        except KeyError:
            known = ", ".join(sorted(self.sensors))
            raise KeyError(f"no sensor mount named '{name}' in vehicle.yaml (have: {known})") from None

    def configuration_matrix(self) -> np.ndarray:
        """6xN matrix B mapping per-thruster force to body wrench: ``tau = B u``.

        Layer 5 allocation inverts this. Its rank tells you how many of the
        six degrees of freedom the vehicle can actually command: rank 6 means
        arbitrary attitude while translating, which is what the gate style
        points need.
        """
        axis_local = self.thruster_axis_local
        return np.column_stack([t.wrench_column(axis_local) for t in self.thrusters])

    @property
    def net_buoyancy_n(self) -> float:
        """Positive means the vehicle floats. Should be slightly positive so a
        dead vehicle surfaces on its own - a safety requirement, not a tuning
        preference."""
        g = 9.80665
        displaced = self.body["displaced_volume_m3"] * self.body["water_density_kg_m3"]
        return (displaced - self.body["mass_kg"]) * g


def default_config_path() -> str:
    """Installed share path when available, source tree otherwise.

    Falling back to the source tree keeps unit tests and offline scripts
    working without a colcon install.
    """
    try:
        from ament_index_python.packages import get_package_share_directory

        return os.path.join(get_package_share_directory("auv_core"), "config", "vehicle.yaml")
    except Exception:
        here = os.path.dirname(os.path.abspath(__file__))
        return os.path.normpath(os.path.join(here, "..", "config", "vehicle.yaml"))


def load_vehicle_config(path: str | None = None) -> VehicleConfig:
    path = path or default_config_path()
    with open(path, "r") as handle:
        raw = yaml.safe_load(handle)

    # The file is written in ROS parameter form so a node can load it
    # directly; unwrap that here for plain-Python consumers.
    params = raw
    for key in ("/**", "ros__parameters"):
        if isinstance(params, dict) and key in params:
            params = params[key]

    sensors = {
        name: SensorMount(
            name=name,
            frame=entry["frame"],
            xyz=np.asarray(entry["xyz"], dtype=float),
            rpy=np.asarray(entry["rpy"], dtype=float),
        )
        for name, entry in params["sensors"].items()
    }

    thruster_block = params["thrusters"]
    thrusters = [
        Thruster(
            id=int(unit["id"]),
            xyz=np.asarray(unit["xyz"], dtype=float),
            rpy=np.asarray(unit["rpy"], dtype=float),
            direction_sign=float(unit.get("direction_sign", 1.0)),
        )
        for unit in thruster_block["units"]
    ]

    return VehicleConfig(
        frames=params["frames"],
        body=params["body"],
        sensors=sensors,
        thrusters=thrusters,
        thruster_axis_local=np.asarray(thruster_block["axis_local"], dtype=float),
        max_forward_n=float(thruster_block["max_forward_n"]),
        max_reverse_n=float(thruster_block["max_reverse_n"]),
        hydrodynamics=params.get("hydrodynamics", {}),
    )
