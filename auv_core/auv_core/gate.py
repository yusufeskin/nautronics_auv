"""Base node for Layer 1 sensor gates.

A gate sits between a driver and the estimator and does exactly three things:
validate, convert conventions, and report health. It never integrates,
never fuses, and never talks to the autopilot - keeping those responsibilities
apart is what lets the estimator be swapped without touching sensor handling
and vice versa.

Subclasses implement the sensor-specific checks and call :meth:`accept` or
:meth:`reject` once per incoming sample.
"""

from __future__ import annotations

from rclpy.node import Node

from auv_core.health import HealthConfig, HealthTracker, Status
from auv_core_msgs.msg import SensorHealth

__all__ = ["SensorGateNode"]


class SensorGateNode(Node):
    def __init__(
        self,
        node_name: str,
        sensor_name: str,
        default_expected_rate_hz: float,
        default_stale_timeout_s: float,
    ):
        super().__init__(node_name)

        self.declare_parameter("expected_rate_hz", default_expected_rate_hz)
        self.declare_parameter("stale_timeout_s", default_stale_timeout_s)
        self.declare_parameter("health_window_s", 5.0)
        self.declare_parameter("health_publish_rate_hz", 2.0)
        self.declare_parameter("min_rate_fraction", 0.6)
        self.declare_parameter("degraded_reject_ratio", 0.25)

        self._tracker = HealthTracker(
            HealthConfig(
                sensor=sensor_name,
                expected_rate_hz=self._param("expected_rate_hz"),
                stale_timeout_s=self._param("stale_timeout_s"),
                window_s=self._param("health_window_s"),
                min_rate_fraction=self._param("min_rate_fraction"),
                degraded_reject_ratio=self._param("degraded_reject_ratio"),
            )
        )
        self._sensor_name = sensor_name
        self._last_status = None

        self._health_pub = self.create_publisher(SensorHealth, "~/health", 10)

        publish_rate = max(0.1, self._param("health_publish_rate_hz"))
        self._health_timer = self.create_timer(1.0 / publish_rate, self._publish_health)

    # -- clock -------------------------------------------------------------

    def now_seconds(self) -> float:
        """Current time on the node clock, in seconds.

        Deliberately the ROS clock and not ``time.monotonic``: under
        ``use_sim_time`` the two run at different speeds, and mixing time
        bases inside one node produces staleness checks that silently never
        fire. Every timestamp a gate compares must come from here or from a
        message header, never from both.
        """
        return self.get_clock().now().nanoseconds * 1e-9

    # -- gate verdicts -----------------------------------------------------

    def accept(self, degraded_reason: str = "") -> None:
        self._tracker.record_accept(self.now_seconds(), degraded_reason)

    def reject(self, reason: str) -> None:
        self._tracker.record_reject(self.now_seconds(), reason)

    @property
    def fusable(self) -> bool:
        return self._tracker.evaluate(self.now_seconds()).fusable

    # -- health reporting --------------------------------------------------

    def _publish_health(self) -> None:
        report = self._tracker.evaluate(self.now_seconds())

        msg = SensorHealth()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.sensor = self._sensor_name
        msg.status = report.status
        msg.fusable = report.fusable
        msg.reason = report.reason
        msg.measured_rate_hz = float(report.measured_rate_hz)
        msg.expected_rate_hz = float(self._tracker.config.expected_rate_hz)
        msg.accepted_count = report.accepted_count
        msg.rejected_count = report.rejected_count
        msg.reject_ratio = float(report.reject_ratio)
        self._health_pub.publish(msg)

        # Log only on transitions. A gate that logs every rejected sample
        # buries the one line that mattered under thousands of duplicates.
        if report.status != self._last_status:
            self._last_status = report.status
            text = f"{self._sensor_name}: {_STATUS_NAMES[report.status]}"
            if report.reason:
                text += f" ({report.reason})"
            if report.status == Status.OK:
                self.get_logger().info(text)
            elif report.status == Status.DEGRADED:
                self.get_logger().warn(text)
            else:
                self.get_logger().error(text)

    # -- helpers -----------------------------------------------------------

    def _param(self, name: str):
        return self.get_parameter(name).value


_STATUS_NAMES = {
    Status.OK: "OK",
    Status.DEGRADED: "DEGRADED",
    Status.REJECTED: "REJECTED",
    Status.STALE: "STALE",
    Status.NO_DATA: "NO DATA",
}
