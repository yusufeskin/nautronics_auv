"""Synthesises a pressure sensor in simulation from ground-truth odometry.

The Gazebo model carries no barometer, but the depth gate is load-bearing and
must be exercised by the same stack that runs on the vehicle. This node makes
the sim look like it has one: same message type, same units, same noise
character, so ``depth_gate`` cannot tell the difference.

Simulation-only. It must never appear in a real-vehicle launch file, which is
why it lives behind its own executable rather than a mode flag.
"""

from __future__ import annotations

import random

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import FluidPressure

from auv_sensors.depth_validation import NOMINAL_ATMOSPHERIC_PA, STANDARD_GRAVITY


class SimPressureSource(Node):
    def __init__(self):
        super().__init__("sim_pressure_source")

        self.declare_parameter("ground_truth_topic", "/odom")
        self.declare_parameter("frame_id", "depth_link")
        self.declare_parameter("water_density_kg_m3", 1000.0)
        self.declare_parameter("surface_pressure_pa", NOMINAL_ATMOSPHERIC_PA)
        # Standard deviation of the synthetic noise, in pascals. 30 Pa is
        # about 3 mm of water, in the region of a Bar30.
        self.declare_parameter("noise_stddev_pa", 30.0)
        self.declare_parameter("publish_rate_hz", 20.0)

        self._frame_id = self.get_parameter("frame_id").value
        self._density = self.get_parameter("water_density_kg_m3").value
        self._surface = self.get_parameter("surface_pressure_pa").value
        self._noise = self.get_parameter("noise_stddev_pa").value

        self._latest_z = None

        self._pub = self.create_publisher(FluidPressure, "~/raw", 20)
        self.create_subscription(
            Odometry, self.get_parameter("ground_truth_topic").value, self._on_odom, 10
        )

        rate = max(1.0, self.get_parameter("publish_rate_hz").value)
        self.create_timer(1.0 / rate, self._publish)

    def _on_odom(self, msg: Odometry) -> None:
        self._latest_z = msg.pose.pose.position.z

    def _publish(self) -> None:
        if self._latest_z is None:
            return

        depth = max(0.0, -self._latest_z)
        pressure = self._surface + self._density * STANDARD_GRAVITY * depth
        pressure += random.gauss(0.0, self._noise)

        msg = FluidPressure()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._frame_id
        msg.fluid_pressure = float(pressure)
        msg.variance = float(self._noise ** 2)
        self._pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = SimPressureSource()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
