"""Publishes NavStatus: how much the current state estimate is worth.

Thin wrapper around ``nav_status.NavStatusTracker``. It subscribes to the
Layer 1 health topics and to the filter's own output, and answers one
question for the mission layer: is the estimate aided, coasting, or not to be
trusted.

This exists because ``robot_localization`` will not tell you. It keeps
publishing odometry after the DVL has gone silent, and that odometry looks
identical to odometry backed by real velocity measurements.
"""

from __future__ import annotations

import rclpy
from geometry_msgs.msg import Vector3Stamped
from nav_msgs.msg import Odometry
from rclpy.node import Node

from auv_core_msgs.msg import NavStatus, SensorHealth
from auv_estimation.nav_status import NavMode, NavStatusConfig, NavStatusTracker

_MODE_NAMES = {
    NavMode.INIT: "INIT",
    NavMode.AIDED: "AIDED",
    NavMode.DEAD_RECKONING: "DEAD RECKONING",
    NavMode.FAULT: "FAULT",
}


class NavStatusNode(Node):
    def __init__(self):
        super().__init__("nav_status")

        self.declare_parameter("publish_rate_hz", 5.0)
        self.declare_parameter("odom_timeout_s", 1.0)
        self.declare_parameter("health_timeout_s", 2.0)
        self.declare_parameter("critical_sensors", ["imu"])
        # Logged once when a dead-reckoning stretch passes this. Not a policy:
        # nothing changes behaviour, it just stops the operator from having to
        # watch a number.
        self.declare_parameter("dead_reckoning_warn_s", 5.0)

        self._tracker = NavStatusTracker(
            NavStatusConfig(
                odom_timeout_s=self._param("odom_timeout_s"),
                health_timeout_s=self._param("health_timeout_s"),
                critical_sensors=tuple(self._param("critical_sensors")),
            )
        )
        self._warn_after = self._param("dead_reckoning_warn_s")
        self._last_mode = None
        self._warned_this_gap = False

        self._pub = self.create_publisher(NavStatus, "/nav_status", 10)

        # One subscription per gate. Health topics are remapped in the launch
        # file so that adding a sensor does not mean editing this node.
        for sensor in ("dvl", "imu", "depth", "heading"):
            self.create_subscription(
                SensorHealth,
                f"~/{sensor}_health",
                self._make_health_callback(sensor),
                10,
            )

        self.create_subscription(Odometry, "~/odom", self._on_odom, 10)
        self.create_subscription(Vector3Stamped, "~/gyro_bias", self._on_bias, 10)

        rate = max(0.5, self._param("publish_rate_hz"))
        self.create_timer(1.0 / rate, self._publish)

    # -- callbacks ---------------------------------------------------------

    def _make_health_callback(self, sensor: str):
        def callback(msg: SensorHealth) -> None:
            # The gate's own `fusable` verdict is authoritative; this node
            # deliberately does not second-guess it by re-reading `status`.
            self._tracker.update_health(sensor, msg.fusable, self._now())
        return callback

    def _on_odom(self, msg: Odometry) -> None:
        self._tracker.update_odometry(msg.pose.covariance, self._now())

    def _on_bias(self, msg: Vector3Stamped) -> None:
        self._tracker.update_gyro_bias((msg.vector.x, msg.vector.y, msg.vector.z))

    # -- publishing --------------------------------------------------------

    def _publish(self) -> None:
        report = self._tracker.evaluate(self._now())

        msg = NavStatus()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.mode = report.mode
        msg.dvl_fused = report.dvl_fused
        msg.depth_fused = report.depth_fused
        msg.heading_fused = report.heading_fused
        msg.dead_reckoning_time_s = float(report.dead_reckoning_time_s)
        msg.position_sigma_xy_m = float(report.position_sigma_xy_m)
        msg.position_sigma_z_m = float(report.position_sigma_z_m)
        msg.yaw_sigma_rad = float(report.yaw_sigma_rad)
        msg.gyro_bias.x = float(report.gyro_bias[0])
        msg.gyro_bias.y = float(report.gyro_bias[1])
        msg.gyro_bias.z = float(report.gyro_bias[2])
        self._pub.publish(msg)

        self._log_transitions(report)

    def _log_transitions(self, report) -> None:
        if report.mode != self._last_mode:
            self._last_mode = report.mode
            self._warned_this_gap = False
            text = _MODE_NAMES[report.mode]
            if report.reason:
                text += f": {report.reason}"
            if report.mode == NavMode.AIDED:
                self.get_logger().info(text)
            elif report.mode == NavMode.FAULT:
                self.get_logger().error(text)
            else:
                self.get_logger().warn(text)

        if (
            report.mode == NavMode.DEAD_RECKONING
            and not self._warned_this_gap
            and report.dead_reckoning_time_s > self._warn_after
        ):
            self._warned_this_gap = True
            self.get_logger().warn(
                f"dead reckoning for {report.dead_reckoning_time_s:.1f} s; "
                f"position sigma now {report.position_sigma_xy_m:.2f} m"
            )

    # -- helpers -----------------------------------------------------------

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _param(self, name: str):
        return self.get_parameter(name).value


def main(args=None):
    rclpy.init(args=args)
    node = NavStatusNode()
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
