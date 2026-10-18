"""Layer 4 output stage: axis demand to ArduSub RC channels.

This node does the unit conversion and the safety gating, and then stops.
It does **not** speak MAVLink.

That restraint is the whole point. Only one process may hold the link to the
autopilot - a second one cannot bind the same endpoint, and even if it could,
two independent sources of command authority over one vehicle is a worse
problem than a port conflict. auv_hardware/pixhawk_bridge2 owns the link; this
node publishes auv_core_msgs/PilotDemand and that bridge forwards it as
RC_CHANNELS_OVERRIDE without thinking about it.

Everything above works in thrust fractions and rates and knows nothing about
sticks; everything below works in PWM and knows nothing about the vehicle.
The conversion, its calibration, and the decision to send nothing at all live
here.

Gating:
  disabled          publishes nothing. ArduSub's override times out and the
                    pilot's own input applies again.
  enabled, stale    publishes neutral on every channel: hold depth, hold
                    heading, stop pushing.
  enabled, fresh    publishes the mapped demand.
"""

from __future__ import annotations

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from std_srvs.srv import SetBool

from auv_actuation.demand_mapping import AxisDemandMapper, ChannelLimits
from auv_core_msgs.msg import AxisDemand, PilotDemand

LOG_THROTTLE_S = 5.0


class PilotDemandNode(Node):
    def __init__(self):
        super().__init__("pilot_demand")

        self.declare_parameter("send_rate_hz", 20.0)
        # Longer than a couple of controller periods, short enough that a dead
        # controller cannot keep the vehicle driving into a wall.
        self.declare_parameter("command_timeout_s", 0.5)
        self.declare_parameter("start_enabled", False)

        defaults = ChannelLimits()
        for name in ("neutral_pwm", "span_pwm", "throttle_deadzone_pwm",
                     "yaw_deadzone_pwm", "forward_deadzone_pwm", "lateral_deadzone_pwm"):
            self.declare_parameter(name, getattr(defaults, name))
        for name in ("max_climb_rate_up", "max_climb_rate_down", "max_yaw_rate"):
            self.declare_parameter(name, getattr(defaults, name))

        self._mapper = AxisDemandMapper(limits=ChannelLimits(
            neutral_pwm=self._param("neutral_pwm"),
            span_pwm=self._param("span_pwm"),
            max_climb_rate_up=self._param("max_climb_rate_up"),
            max_climb_rate_down=self._param("max_climb_rate_down"),
            max_yaw_rate=self._param("max_yaw_rate"),
            throttle_deadzone_pwm=self._param("throttle_deadzone_pwm"),
            yaw_deadzone_pwm=self._param("yaw_deadzone_pwm"),
            forward_deadzone_pwm=self._param("forward_deadzone_pwm"),
            lateral_deadzone_pwm=self._param("lateral_deadzone_pwm"),
        ))

        self._enabled = bool(self._param("start_enabled"))
        self._last_command = None
        self._last_command_time = None
        self._was_stale = False

        self._demand_pub = self.create_publisher(PilotDemand, "~/demand", 10)
        self._saturated_pub = self.create_publisher(Bool, "~/saturated", 10)

        self.create_subscription(AxisDemand, "~/cmd_axis", self._on_axis, 10)
        self.create_service(SetBool, "~/enable", self._on_enable)

        rate = max(1.0, self._param("send_rate_hz"))
        self.create_timer(1.0 / rate, self._publish)

        self.get_logger().warn(
            "output disabled; call ~/enable with data:=true to allow commands"
            if not self._enabled else "output ENABLED at startup"
        )

    # -- inputs ------------------------------------------------------------

    def _on_axis(self, msg: AxisDemand) -> None:
        self._last_command = (msg.surge, msg.sway, msg.climb_rate, msg.yaw_rate)
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
        if not self._enabled:
            return

        demand = self._resolve_demand()

        msg = PilotDemand()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pitch = demand.pitch
        msg.roll = demand.roll
        msg.throttle = demand.throttle
        msg.yaw = demand.yaw
        msg.forward = demand.forward
        msg.lateral = demand.lateral
        self._demand_pub.publish(msg)

        flag = Bool()
        flag.data = bool(demand.saturated)
        self._saturated_pub.publish(flag)

    def _resolve_demand(self):
        if self._last_command is None or self._last_command_time is None:
            return self._mapper.neutral()

        age = self._now() - self._last_command_time
        if age > self._param("command_timeout_s"):
            if not self._was_stale:
                self._was_stale = True
                self.get_logger().warn(
                    f"no axis demand for {age:.2f} s; sticks centred "
                    "(hold depth, hold heading, stop pushing)"
                )
            return self._mapper.neutral()

        if self._was_stale:
            self._was_stale = False
            self.get_logger().info("axis demand resumed")

        demand = self._mapper.map(*self._last_command)
        if demand.saturated:
            self.get_logger().warn(
                "demand beyond full stick, clipped: lower the controller's caps "
                "or check max_climb_rate_* / max_yaw_rate against the vehicle",
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
