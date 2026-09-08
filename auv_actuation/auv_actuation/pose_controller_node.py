"""Layer 4 outer loop: pose or velocity target in, wrench out.

Two ways to drive it, and the more recent one wins:

  ~/target_pose   a pose in odom. The cascade runs in full: position error
                  becomes a desired approach speed, that speed is tracked.
                  This is the "go to this place, turn to this heading" path.
  ~/cmd_vel       a body velocity directly, bypassing the outer loop. This is
                  the socket the guidance layer plugs into later.

The wrench it produces goes to pilot_demand, which converts it to ArduSub's
pilot-input axes; auv_hardware/pixhawk_bridge2 then forwards that as
MANUAL_CONTROL, because it is the one process that owns the MAVLink link.

The autopilot stabilises attitude and this loop decides where to go. Because
the loop closes on this stack's own estimate, the autopilot's internal idea of
heading never has to agree with ours - it only has to hold still when we stop
asking it to move.
"""

from __future__ import annotations

import math

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TwistStamped, WrenchStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Bool
from std_srvs.srv import SetBool

from auv_core import frames
from auv_actuation.control_law import Axis, CascadeGains, PoseController, PoseTarget

LOG_THROTTLE_S = 5.0


class PoseControllerNode(Node):
    def __init__(self):
        super().__init__("pose_controller")

        self.declare_parameter("rate_hz", 20.0)
        self.declare_parameter("odom_timeout_s", 0.5)
        self.declare_parameter("cmd_vel_timeout_s", 0.5)
        self.declare_parameter("start_enabled", False)

        self.declare_parameter("kp_position", 0.8)
        self.declare_parameter("kp_yaw", 1.2)
        self.declare_parameter("max_speed_xy", 0.5)
        self.declare_parameter("max_speed_z", 0.3)
        self.declare_parameter("max_yaw_rate", 0.6)
        self.declare_parameter("position_tolerance_m", 0.1)
        self.declare_parameter("yaw_tolerance_deg", 5.0)

        for axis, limit in (("surge", 40.0), ("sway", 30.0),
                            ("heave", 40.0), ("yaw_rate", 8.0)):
            self.declare_parameter(f"{axis}.kp", 0.0)
            self.declare_parameter(f"{axis}.ki", 0.0)
            self.declare_parameter(f"{axis}.kd", 0.0)
            self.declare_parameter(f"{axis}.output_limit", limit)
            self.declare_parameter(f"{axis}.integral_limit", limit * 0.5)

        self._controller = PoseController(gains=self._load_gains())
        self._enabled = bool(self._param("start_enabled"))

        self._odom = None
        self._odom_time = None
        self._target = None
        self._cmd_vel = None
        self._cmd_vel_time = None
        self._saturated = False
        self._last_step_time = None

        self._wrench_pub = self.create_publisher(WrenchStamped, "~/cmd_wrench", 10)
        self._error_pub = self.create_publisher(PoseStamped, "~/pose_error", 10)
        self._at_target_pub = self.create_publisher(Bool, "~/at_target", 10)

        self.create_subscription(Odometry, "~/odom", self._on_odom, 10)
        self.create_subscription(PoseStamped, "~/target_pose", self._on_target, 10)
        self.create_subscription(TwistStamped, "~/cmd_vel", self._on_cmd_vel, 10)
        self.create_subscription(Bool, "~/saturated", self._on_saturated, 10)
        self.create_service(SetBool, "~/enable", self._on_enable)

        rate = max(1.0, self._param("rate_hz"))
        self.create_timer(1.0 / rate, self._step)

        if not self._enabled:
            self.get_logger().warn(
                "controller disabled; call ~/enable with data:=true to start"
            )
        if self._param("surge.kp") == 0.0:
            self.get_logger().warn(
                "inner-loop gains are all zero: this controller will command "
                "nothing until they are tuned, and they cannot be tuned until "
                "the bridge's full-scale forces are measured"
            )

    # -- inputs ------------------------------------------------------------

    def _on_odom(self, msg: Odometry) -> None:
        self._odom = msg
        self._odom_time = self._now()

    def _on_target(self, msg: PoseStamped) -> None:
        quat = np.array([
            msg.pose.orientation.x, msg.pose.orientation.y,
            msg.pose.orientation.z, msg.pose.orientation.w,
        ])
        # A zero quaternion means "hold whatever heading you have"; a caller
        # that wants a heading has to say so with a real orientation.
        yaw = None
        if float(np.linalg.norm(quat)) > 1e-6:
            yaw = frames.euler_rpy_from_quat(quat)[2]

        self._target = PoseTarget(
            x=msg.pose.position.x,
            y=msg.pose.position.y,
            z=msg.pose.position.z,
            yaw=yaw,
        )
        self._cmd_vel = None
        self._controller.reset()
        self.get_logger().info(
            f"new target: ({self._target.x:.2f}, {self._target.y:.2f}, "
            f"{self._target.z:.2f})"
            + (f" yaw {math.degrees(yaw):.0f} deg" if yaw is not None else "")
        )

    def _on_cmd_vel(self, msg: TwistStamped) -> None:
        self._cmd_vel = np.array([
            msg.twist.linear.x, msg.twist.linear.y,
            msg.twist.linear.z, msg.twist.angular.z,
        ])
        self._cmd_vel_time = self._now()

    def _on_saturated(self, msg: Bool) -> None:
        self._saturated = bool(msg.data)

    def _on_enable(self, request, response):
        self._enabled = bool(request.data)
        self._controller.reset()
        self._last_step_time = None
        state = "ENABLED" if self._enabled else "disabled"
        self.get_logger().warn(f"controller {state}")
        response.success = True
        response.message = f"pose_controller {state}"
        return response

    # -- the loop ----------------------------------------------------------

    def _step(self) -> None:
        now = self._now()
        dt = 0.0 if self._last_step_time is None else now - self._last_step_time
        self._last_step_time = now

        if not self._enabled or dt <= 0.0:
            return

        if self._odom is None or (now - self._odom_time) > self._param("odom_timeout_s"):
            # No estimate means no idea where we are. Stop commanding and let
            # the bridge's own watchdog fall back to neutral.
            self.get_logger().warn(
                "no odometry; not commanding", throttle_duration_sec=LOG_THROTTLE_S
            )
            self._controller.reset()
            return

        measured = np.array([
            self._odom.twist.twist.linear.x,
            self._odom.twist.twist.linear.y,
            self._odom.twist.twist.linear.z,
            self._odom.twist.twist.angular.z,
        ])

        if self._velocity_command_is_current(now):
            # Guidance is driving: skip the outer loop entirely.
            desired = self._cmd_vel
            force, torque = self._controller.velocity_to_wrench(
                desired, measured, dt, self._saturated
            )
            self._publish(force, torque)
            return

        if self._target is None:
            return

        position = np.array([
            self._odom.pose.pose.position.x,
            self._odom.pose.pose.position.y,
            self._odom.pose.pose.position.z,
        ])
        quat = np.array([
            self._odom.pose.pose.orientation.x, self._odom.pose.pose.orientation.y,
            self._odom.pose.pose.orientation.z, self._odom.pose.pose.orientation.w,
        ])
        yaw = frames.euler_rpy_from_quat(quat)[2]

        output = self._controller.step(
            self._target, position, yaw, measured, dt, self._saturated
        )
        self._publish(output.force, output.torque)
        self._publish_diagnostics(output)

    def _velocity_command_is_current(self, now: float) -> bool:
        if self._cmd_vel is None or self._cmd_vel_time is None:
            return False
        return (now - self._cmd_vel_time) <= self._param("cmd_vel_timeout_s")

    # -- outputs -----------------------------------------------------------

    def _publish(self, force, torque) -> None:
        msg = WrenchStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.wrench.force.x = float(force[0])
        msg.wrench.force.y = float(force[1])
        msg.wrench.force.z = float(force[2])
        msg.wrench.torque.x = float(torque[0])
        msg.wrench.torque.y = float(torque[1])
        msg.wrench.torque.z = float(torque[2])
        self._wrench_pub.publish(msg)

    def _publish_diagnostics(self, output) -> None:
        error = PoseStamped()
        error.header.stamp = self.get_clock().now().to_msg()
        error.header.frame_id = "base_link"
        error.pose.position.x = float(output.position_error[0])
        error.pose.position.y = float(output.position_error[1])
        error.pose.position.z = float(output.position_error[2])
        quat = frames.quat_from_euler_rpy(0.0, 0.0, output.yaw_error)
        error.pose.orientation.x = float(quat[0])
        error.pose.orientation.y = float(quat[1])
        error.pose.orientation.z = float(quat[2])
        error.pose.orientation.w = float(quat[3])
        self._error_pub.publish(error)

        flag = Bool()
        flag.data = bool(output.at_target)
        self._at_target_pub.publish(flag)

    # -- helpers -----------------------------------------------------------

    def _load_gains(self) -> CascadeGains:
        def axis(name: str) -> Axis:
            return Axis(
                kp=self._param(f"{name}.kp"),
                ki=self._param(f"{name}.ki"),
                kd=self._param(f"{name}.kd"),
                output_limit=self._param(f"{name}.output_limit"),
                integral_limit=self._param(f"{name}.integral_limit"),
            )

        return CascadeGains(
            kp_position=self._param("kp_position"),
            kp_yaw=self._param("kp_yaw"),
            max_speed_xy=self._param("max_speed_xy"),
            max_speed_z=self._param("max_speed_z"),
            max_yaw_rate=self._param("max_yaw_rate"),
            surge=axis("surge"),
            sway=axis("sway"),
            heave=axis("heave"),
            yaw_rate=axis("yaw_rate"),
            position_tolerance_m=self._param("position_tolerance_m"),
            yaw_tolerance_rad=math.radians(self._param("yaw_tolerance_deg")),
        )

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _param(self, name: str):
        return self.get_parameter(name).value


def main(args=None):
    rclpy.init(args=args)
    node = PoseControllerNode()
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
