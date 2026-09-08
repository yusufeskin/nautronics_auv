"""Layer 4 hardware boundary: commanded wrench out to ArduSub.

This is the only node in the new stack that talks to the autopilot. Everything
above it works in forces and torques and knows nothing about MAVLink; ArduSub
below it keeps doing the job it is already good at - stabilising attitude,
holding depth, and failing safe.

Deliberately not this node's job:

  arming and mode changes   Those stay with the operator and the existing
                            mode/arm services. A bridge that can arm the
                            vehicle is a bridge that can arm it by accident.
  absolute heading          ArduSub holds heading with its own EKF3 yaw, which
                            is magnetometer-aided and is not the yaw this
                            stack estimates. Only rates are commanded here;
                            the outer loop owns absolute heading.
  thrust allocation         ArduSub's mixer does it. If that is ever replaced
                            by our own allocator, this node is swapped out and
                            the WrenchStamped contract above it does not move.

Safety behaviour, in order of precedence: output is neutral unless explicitly
enabled; it reverts to neutral if the commanded wrench goes stale; and it is
sent continuously at a fixed rate so ArduSub's own pilot-input failsafe never
sees a gap it has to interpret.
"""

from __future__ import annotations

import numpy as np
import rclpy
from geometry_msgs.msg import WrenchStamped
from rclpy.node import Node
from std_msgs.msg import Bool
from std_srvs.srv import SetBool

from auv_actuation.demand_mapping import DemandLimits, WrenchDemandMapper

try:
    from pymavlink import mavutil
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise ImportError(
        "pymavlink is required by ardusub_bridge; it is listed in requirements.txt"
    ) from exc

LOG_THROTTLE_S = 5.0


class ArduSubBridge(Node):
    def __init__(self):
        super().__init__("ardusub_bridge")

        self.declare_parameter("connection", "udp:127.0.0.1:14551")
        self.declare_parameter("send_rate_hz", 20.0)
        # Longer than a couple of controller periods, short enough that a dead
        # controller cannot keep the vehicle driving into a wall.
        self.declare_parameter("command_timeout_s", 0.5)
        self.declare_parameter("start_enabled", False)

        self.declare_parameter("surge_full_scale_n", 40.0)
        self.declare_parameter("sway_full_scale_n", 30.0)
        self.declare_parameter("heave_full_scale_n", 40.0)
        self.declare_parameter("yaw_full_scale_nm", 8.0)
        self.declare_parameter("neutral_z", 500)
        self.declare_parameter("deadband_counts", 0)

        self._mapper = WrenchDemandMapper(
            limits=DemandLimits(
                surge_full_scale_n=self._param("surge_full_scale_n"),
                sway_full_scale_n=self._param("sway_full_scale_n"),
                heave_full_scale_n=self._param("heave_full_scale_n"),
                yaw_full_scale_nm=self._param("yaw_full_scale_nm"),
                neutral_z=self._param("neutral_z"),
                deadband_counts=self._param("deadband_counts"),
            )
        )

        self._enabled = bool(self._param("start_enabled"))
        self._last_command = None
        self._last_command_time = None
        self._was_stale = False

        self._master = mavutil.mavlink_connection(self._param("connection"))
        self.get_logger().info("waiting for heartbeat...")
        self._master.wait_heartbeat()
        self.get_logger().info(
            f"connected to system {self._master.target_system}, "
            f"component {self._master.target_component}"
        )

        # What the vehicle will actually feel, after clipping and after the
        # axes ArduSub cannot deliver are dropped. The controller compares it
        # against what it asked for and stops integrating on the difference.
        self._realised_pub = self.create_publisher(WrenchStamped, "~/realised_wrench", 10)
        self._saturated_pub = self.create_publisher(Bool, "~/saturated", 10)

        self.create_subscription(WrenchStamped, "~/cmd_wrench", self._on_wrench, 10)
        self.create_service(SetBool, "~/enable", self._on_enable)

        rate = max(1.0, self._param("send_rate_hz"))
        self.create_timer(1.0 / rate, self._send)

        self.get_logger().warn(
            "output disabled; call ~/enable with data:=true to allow commands"
            if not self._enabled else "output ENABLED at startup"
        )

    # -- inputs ------------------------------------------------------------

    def _on_wrench(self, msg: WrenchStamped) -> None:
        self._last_command = (
            np.array([msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z]),
            np.array([msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z]),
        )
        self._last_command_time = self._now()

    def _on_enable(self, request, response):
        self._enabled = bool(request.data)
        state = "ENABLED" if self._enabled else "disabled"
        self.get_logger().warn(f"output {state}")
        response.success = True
        response.message = f"ardusub_bridge output {state}"
        return response

    # -- output ------------------------------------------------------------

    def _send(self) -> None:
        demand = self._resolve_demand()

        try:
            # Sent unconditionally at a fixed rate, even when neutral: a gap in
            # pilot input is something ArduSub has to interpret, and a steady
            # neutral is unambiguous where silence is not.
            self._master.mav.manual_control_send(
                self._master.target_system,
                int(demand.x),
                int(demand.y),
                int(demand.z),
                int(demand.r),
                0,  # buttons
            )
        except Exception as exc:
            self.get_logger().error(
                f"MANUAL_CONTROL send failed: {exc}", throttle_duration_sec=LOG_THROTTLE_S
            )
            return

        self._publish_realised(demand)

    def _resolve_demand(self):
        if not self._enabled:
            return self._mapper.neutral()

        if self._last_command is None or self._last_command_time is None:
            return self._mapper.neutral()

        age = self._now() - self._last_command_time
        if age > self._param("command_timeout_s"):
            if not self._was_stale:
                self._was_stale = True
                self.get_logger().warn(
                    f"no wrench for {age:.2f} s; holding depth and stopping translation"
                )
            return self._mapper.neutral()

        if self._was_stale:
            self._was_stale = False
            self.get_logger().info("wrench command resumed")

        demand = self._mapper.map(*self._last_command)

        if demand.dropped_roll_pitch:
            self.get_logger().warn(
                "roll/pitch torque commanded but not deliverable: ArduSub's "
                "STABILIZE and ALT_HOLD stabilise those axes to level and "
                "accept a lean angle, not a torque",
                throttle_duration_sec=LOG_THROTTLE_S,
            )
        if demand.saturated:
            self.get_logger().warn(
                "demand saturated; the controller should hold its integrators",
                throttle_duration_sec=LOG_THROTTLE_S,
            )
        return demand

    def _publish_realised(self, demand) -> None:
        msg = WrenchStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.wrench.force.x = float(demand.realised_force[0])
        msg.wrench.force.y = float(demand.realised_force[1])
        msg.wrench.force.z = float(demand.realised_force[2])
        msg.wrench.torque.x = float(demand.realised_torque[0])
        msg.wrench.torque.y = float(demand.realised_torque[1])
        msg.wrench.torque.z = float(demand.realised_torque[2])
        self._realised_pub.publish(msg)

        flag = Bool()
        flag.data = bool(demand.saturated)
        self._saturated_pub.publish(flag)

    # -- helpers -----------------------------------------------------------

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _param(self, name: str):
        return self.get_parameter(name).value


def main(args=None):
    rclpy.init(args=args)
    node = ArduSubBridge()
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
