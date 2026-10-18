"""Layer 4 control loop: reference pose in, ALT_HOLD axis demand out.

Three ways to give it somewhere to be:

  ~/reference     nav_msgs/Odometry, streamed by guidance: the pose to be at
                  now (odom) and the reference's own velocity as feedforward
                  (base_link). If the stream stops, the controller holds at
                  the last reference rather than chasing a stale one.
  ~/target_pose   geometry_msgs/PoseStamped in odom, a fixed target. For
                  testing by hand and for the mission until guidance exists.
  ~/hold          std_srvs/Trigger: hold exactly where the vehicle is now.

Enabling the controller also holds where the vehicle is, so turning it on
never sends the vehicle after a target left over from earlier.

The demand goes to pilot_demand, which turns it into RC channel PWM;
auv_hardware/pixhawk_bridge2 then forwards that, because it is the one
process that owns the MAVLink link.
"""

from __future__ import annotations

import math

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Bool
from std_srvs.srv import SetBool, Trigger

from auv_core import frames
from auv_core_msgs.msg import AxisDemand, NavStatus
from auv_actuation.control_law import Axis, ControlGains, PoseController, Reference

LOG_THROTTLE_S = 5.0
WORLD_FRAME = "odom"


class PoseControllerNode(Node):
    def __init__(self):
        super().__init__("pose_controller")

        self.declare_parameter("rate_hz", 20.0)
        self.declare_parameter("odom_timeout_s", 0.5)
        self.declare_parameter("reference_timeout_s", 0.5)
        self.declare_parameter("nav_status_timeout_s", 1.0)
        self.declare_parameter("dvl_coast_s", 2.0)
        self.declare_parameter("start_enabled", False)

        self.declare_parameter("kp_position", 0.8)
        self.declare_parameter("max_speed_xy", 0.5)
        self.declare_parameter("kp_depth", 0.8)
        self.declare_parameter("max_climb_rate", 0.3)
        self.declare_parameter("kp_yaw", 1.0)
        self.declare_parameter("max_yaw_rate", 0.5)
        self.declare_parameter("position_tolerance_m", 0.1)
        self.declare_parameter("depth_tolerance_m", 0.1)
        self.declare_parameter("yaw_tolerance_deg", 5.0)

        for axis in ("surge", "sway"):
            self.declare_parameter(f"{axis}.kp", 0.0)
            self.declare_parameter(f"{axis}.ki", 0.0)
            self.declare_parameter(f"{axis}.kd", 0.0)
            self.declare_parameter(f"{axis}.output_limit", 1.0)
            self.declare_parameter(f"{axis}.integral_limit", 0.3)

        self._controller = PoseController(gains=self._load_gains())
        self._enabled = bool(self._param("start_enabled"))

        self._odom = None
        self._odom_time = None
        self._nav_status = None
        self._nav_status_time = None
        self._reference = None
        self._streaming = False
        self._reference_time = None
        self._last_step_time = None

        self._axis_pub = self.create_publisher(AxisDemand, "~/cmd_axis", 10)
        self._error_pub = self.create_publisher(PoseStamped, "~/pose_error", 10)
        self._desired_pub = self.create_publisher(TwistStamped, "~/desired_velocity", 10)
        self._at_target_pub = self.create_publisher(Bool, "~/at_target", 10)

        self.create_subscription(Odometry, "~/odom", self._on_odom, 10)
        self.create_subscription(NavStatus, "~/nav_status", self._on_nav_status, 10)
        self.create_subscription(Odometry, "~/reference", self._on_reference, 10)
        self.create_subscription(PoseStamped, "~/target_pose", self._on_target, 10)
        self.create_service(SetBool, "~/enable", self._on_enable)
        self.create_service(Trigger, "~/hold", self._on_hold)

        rate = max(1.0, self._param("rate_hz"))
        self.create_timer(1.0 / rate, self._step)

        if not self._enabled:
            self.get_logger().warn(
                "controller disabled; call ~/enable with data:=true to start"
            )
        if self._param("surge.kp") == 0.0 and self._param("sway.kp") == 0.0:
            self.get_logger().warn(
                "surge and sway gains are zero: the vehicle will hold depth and "
                "heading but will not move horizontally until they are tuned"
            )

    # -- inputs ------------------------------------------------------------

    def _on_odom(self, msg: Odometry) -> None:
        self._odom = msg
        self._odom_time = self._now()

    def _on_nav_status(self, msg: NavStatus) -> None:
        self._nav_status = msg
        self._nav_status_time = self._now()

    def _on_target(self, msg: PoseStamped) -> None:
        if not self._check_frame(msg.header.frame_id, "target_pose"):
            return
        quat = _quat(msg.pose.orientation)
        # A zero quaternion means "keep whatever heading you have"; a caller
        # that wants a heading has to say so with a real orientation.
        yaw = None
        if float(np.linalg.norm(quat)) > 1e-6:
            yaw = frames.euler_rpy_from_quat(quat)[2]

        self._set_reference(Reference(
            x=msg.pose.position.x, y=msg.pose.position.y,
            z=msg.pose.position.z, yaw=yaw,
        ), streaming=False)
        self.get_logger().info(
            f"new target: ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f}, "
            f"{msg.pose.position.z:.2f})"
            + (f" yaw {math.degrees(yaw):.0f} deg" if yaw is not None else "")
        )

    def _on_reference(self, msg: Odometry) -> None:
        if not self._check_frame(msg.header.frame_id, "reference"):
            return
        pose = msg.pose.pose
        reference = Reference(
            x=pose.position.x, y=pose.position.y, z=pose.position.z,
            yaw=frames.euler_rpy_from_quat(_quat(pose.orientation))[2],
            velocity=np.array([
                msg.twist.twist.linear.x,
                msg.twist.twist.linear.y,
                msg.twist.twist.linear.z,
            ]),
            yaw_rate=msg.twist.twist.angular.z,
        )
        # Integrators are kept across a continuous stream - resetting them
        # every message would throw the trim away 20 times a second.
        self._set_reference(reference, streaming=True, reset=not self._streaming)
        self._reference_time = self._now()

    def _on_enable(self, request, response):
        self._enabled = bool(request.data)
        self._last_step_time = None
        if self._enabled:
            # Never resume toward an old target: start by holding here.
            self._reference = None
            self._streaming = False
            if self._odom is not None:
                self._hold_here()
        self._controller.reset()
        state = "ENABLED" if self._enabled else "disabled"
        self.get_logger().warn(f"controller {state}")
        response.success = True
        response.message = f"pose_controller {state}"
        return response

    def _on_hold(self, request, response):
        if self._odom is None:
            response.success = False
            response.message = "no odometry yet; nothing to hold"
            return response
        self._hold_here()
        response.success = True
        response.message = "holding current pose"
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
            # the output stage's watchdog centre the sticks: ArduSub then
            # holds depth and heading on its own.
            self.get_logger().warn(
                "no odometry; not commanding", throttle_duration_sec=LOG_THROTTLE_S
            )
            self._controller.reset()
            return

        if self._reference is None:
            self._hold_here()

        if self._streaming and (now - self._reference_time) > self._param("reference_timeout_s"):
            self.get_logger().warn("reference stream stopped; holding at its last pose")
            last = self._reference
            self._set_reference(Reference(x=last.x, y=last.y, z=last.z, yaw=last.yaw),
                                streaming=False, reset=False)

        pose = self._odom.pose.pose
        position = np.array([pose.position.x, pose.position.y, pose.position.z])
        yaw = frames.euler_rpy_from_quat(_quat(pose.orientation))[2]
        velocity = np.array([
            self._odom.twist.twist.linear.x,
            self._odom.twist.twist.linear.y,
        ])

        output = self._controller.step(
            self._reference, position, yaw, velocity, dt,
            horizontal_ok=self._horizontal_ok(now),
        )
        self._publish(output)

    def _horizontal_ok(self, now: float) -> bool:
        """Is the horizontal velocity good enough to drive forward and lateral on?

        Short DVL gaps are coasted on the estimator's prediction. Past
        dvl_coast_s, or with no word from nav_status at all, forward and
        lateral are centred: a vehicle that drifts slowly is better than one
        driving on a velocity nobody measured.
        """
        status = self._nav_status
        if status is None or (now - self._nav_status_time) > self._param("nav_status_timeout_s"):
            self.get_logger().warn(
                "no nav_status; horizontal axes centred",
                throttle_duration_sec=LOG_THROTTLE_S,
            )
            return False
        if status.mode == NavStatus.MODE_AIDED:
            return True
        if (status.mode == NavStatus.MODE_DEAD_RECKONING
                and status.dead_reckoning_time_s <= self._param("dvl_coast_s")):
            return True
        self.get_logger().warn(
            f"no velocity aiding (mode {status.mode}, "
            f"{status.dead_reckoning_time_s:.1f} s); horizontal axes centred",
            throttle_duration_sec=LOG_THROTTLE_S,
        )
        return False

    # -- helpers -----------------------------------------------------------

    def _hold_here(self) -> None:
        pose = self._odom.pose.pose
        self._set_reference(Reference(
            x=pose.position.x, y=pose.position.y, z=pose.position.z,
            yaw=frames.euler_rpy_from_quat(_quat(pose.orientation))[2],
        ), streaming=False)
        self.get_logger().info(
            f"holding at ({pose.position.x:.2f}, {pose.position.y:.2f}, {pose.position.z:.2f})"
        )

    def _set_reference(self, reference: Reference, streaming: bool, reset: bool = True) -> None:
        self._reference = reference
        self._streaming = streaming
        if reset:
            self._controller.reset()

    def _check_frame(self, frame_id: str, what: str) -> bool:
        if frame_id == WORLD_FRAME:
            return True
        self.get_logger().error(
            f"{what} in frame '{frame_id}' ignored: it must be in '{WORLD_FRAME}'",
            throttle_duration_sec=LOG_THROTTLE_S,
        )
        return False

    def _publish(self, output) -> None:
        stamp = self.get_clock().now().to_msg()

        demand = AxisDemand()
        demand.header.stamp = stamp
        demand.header.frame_id = "base_link"
        demand.surge = float(output.surge)
        demand.sway = float(output.sway)
        demand.climb_rate = float(output.climb_rate)
        demand.yaw_rate = float(output.yaw_rate)
        self._axis_pub.publish(demand)

        error = PoseStamped()
        error.header.stamp = stamp
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

        desired = TwistStamped()
        desired.header.stamp = stamp
        desired.header.frame_id = "base_link"
        desired.twist.linear.x = float(output.desired_velocity[0])
        desired.twist.linear.y = float(output.desired_velocity[1])
        desired.twist.linear.z = float(output.climb_rate)
        desired.twist.angular.z = float(output.yaw_rate)
        self._desired_pub.publish(desired)

        flag = Bool()
        flag.data = bool(output.at_target)
        self._at_target_pub.publish(flag)

    def _load_gains(self) -> ControlGains:
        def axis(name: str) -> Axis:
            return Axis(
                kp=self._param(f"{name}.kp"),
                ki=self._param(f"{name}.ki"),
                kd=self._param(f"{name}.kd"),
                output_limit=self._param(f"{name}.output_limit"),
                integral_limit=self._param(f"{name}.integral_limit"),
            )

        return ControlGains(
            kp_position=self._param("kp_position"),
            max_speed_xy=self._param("max_speed_xy"),
            kp_depth=self._param("kp_depth"),
            max_climb_rate=self._param("max_climb_rate"),
            kp_yaw=self._param("kp_yaw"),
            max_yaw_rate=self._param("max_yaw_rate"),
            surge=axis("surge"),
            sway=axis("sway"),
            position_tolerance_m=self._param("position_tolerance_m"),
            depth_tolerance_m=self._param("depth_tolerance_m"),
            yaw_tolerance_rad=math.radians(self._param("yaw_tolerance_deg")),
        )

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _param(self, name: str):
        return self.get_parameter(name).value


def _quat(orientation) -> np.ndarray:
    return np.array([orientation.x, orientation.y, orientation.z, orientation.w])


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
