"""Layer 4 output stage: commanded wrench to an ArduSub pilot demand.

This node does the unit conversion and the safety gating, and then stops.
It does **not** speak MAVLink.

That restraint is the whole point. Only one process may hold the link to the
autopilot - a second one cannot bind the same endpoint, and even if it could,
two independent sources of command authority over one vehicle is a worse
problem than a port conflict. auv_hardware/pixhawk_bridge2 owns the link; this
node publishes auv_core_msgs/PilotDemand and that bridge forwards it as
MANUAL_CONTROL without thinking about it.

Everything above works in newtons and knows nothing about stick counts;
everything below works in stick counts and knows nothing about forces. The
conversion, its calibration, and the decision to send nothing at all live
here.
"""

from __future__ import annotations

import numpy as np
import rclpy
from geometry_msgs.msg import WrenchStamped
from rclpy.node import Node
from std_msgs.msg import Bool
from std_srvs.srv import SetBool

from auv_actuation.demand_mapping import DemandLimits, WrenchDemandMapper
from auv_core_msgs.msg import PilotDemand

LOG_THROTTLE_S = 5.0


class PilotDemandNode(Node):
    def __init__(self):
        super().__init__("pilot_demand")

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

        self._demand_pub = self.create_publisher(PilotDemand, "~/demand", 10)
        # What the vehicle will actually feel, after clipping and after the
        # axes ArduSub cannot deliver are dropped. The controller compares it
        # against what it asked for and stops integrating on the difference.
        self._realised_pub = self.create_publisher(WrenchStamped, "~/realised_wrench", 10)
        self._saturated_pub = self.create_publisher(Bool, "~/saturated", 10)

        self.create_subscription(WrenchStamped, "~/cmd_wrench", self._on_wrench, 10)
        self.create_service(SetBool, "~/enable", self._on_enable)

        rate = max(1.0, self._param("send_rate_hz"))
        self.create_timer(1.0 / rate, self._publish)

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
        response.message = f"pilot_demand output {state}"
        return response

    # -- output ------------------------------------------------------------

    def _publish(self) -> None:
        demand = self._resolve_demand()
        stamp = self.get_clock().now().to_msg()

        # Published unconditionally at a fixed rate, even when neutral. A gap
        # in pilot input is something the autopilot has to interpret; a steady
        # neutral is unambiguous.
        msg = PilotDemand()
        msg.header.stamp = stamp
        msg.header.frame_id = "base_link"
        msg.x = int(demand.x)
        msg.y = int(demand.y)
        msg.z = int(demand.z)
        msg.r = int(demand.r)
        msg.buttons = 0
        msg.saturated = bool(demand.saturated)
        self._demand_pub.publish(msg)

        realised = WrenchStamped()
        realised.header.stamp = stamp
        realised.header.frame_id = "base_link"
        realised.wrench.force.x = float(demand.realised_force[0])
        realised.wrench.force.y = float(demand.realised_force[1])
        realised.wrench.force.z = float(demand.realised_force[2])
        realised.wrench.torque.x = float(demand.realised_torque[0])
        realised.wrench.torque.y = float(demand.realised_torque[1])
        realised.wrench.torque.z = float(demand.realised_torque[2])
        self._realised_pub.publish(realised)

        flag = Bool()
        flag.data = bool(demand.saturated)
        self._saturated_pub.publish(flag)

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

    # -- helpers -----------------------------------------------------------

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _param(self, name: str):
        return self.get_parameter(name).value


def main(args=None):
    rclpy.init(args=args)
    node = PilotDemandNode()
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
