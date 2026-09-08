"""Layer 1 gate for the pressure/depth sensor.

Converts absolute pressure to a z observation the estimator can fuse. Depth
is the only state on this vehicle that never drifts, so this is the smallest
and most important gate in the stack.

Exposes a ``~/zero_surface`` service that captures the current pressure as
the surface reference. Run it on the deck with the vehicle floating, before
arming; until then the gate refuses to publish rather than quietly assuming a
nominal atmosphere.
"""

from __future__ import annotations

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import FluidPressure
from std_msgs.msg import Float64
from std_srvs.srv import Trigger

from auv_core.gate import SensorGateNode
from auv_core.vehicle_config import default_config_path, load_vehicle_config
from auv_sensors.depth_validation import DepthConverter, DepthLimits, DepthSample

# Horizontal position and orientation are not observed by a pressure sensor.
UNMEASURED_VARIANCE = 1.0e6


class DepthGate(SensorGateNode):
    def __init__(self):
        super().__init__(
            node_name="depth_gate",
            sensor_name="depth",
            default_expected_rate_hz=20.0,
            default_stale_timeout_s=0.5,
        )

        self.declare_parameter("vehicle_config", default_config_path())
        self.declare_parameter("water_density_kg_m3", 1000.0)
        self.declare_parameter("min_pressure_pa", 50_000.0)
        self.declare_parameter("max_pressure_pa", 400_000.0)
        self.declare_parameter("max_depth_rate_mps", 3.0)
        self.declare_parameter("max_dt_s", 1.0)
        self.declare_parameter("max_clock_skew_s", 0.5)
        self.declare_parameter("depth_variance_m2", 4.0e-4)
        self.declare_parameter("require_surface_zero", True)

        config = load_vehicle_config(self.get_parameter("vehicle_config").value)
        self._odom_frame = config.frames["odom"]
        self._depth_variance = self._param("depth_variance_m2")

        self._converter = DepthConverter(
            limits=DepthLimits(
                water_density_kg_m3=self._param("water_density_kg_m3"),
                min_pressure_pa=self._param("min_pressure_pa"),
                max_pressure_pa=self._param("max_pressure_pa"),
                max_depth_rate_mps=self._param("max_depth_rate_mps"),
                max_dt_s=self._param("max_dt_s"),
                max_clock_skew_s=self._param("max_clock_skew_s"),
                depth_variance_m2=self._depth_variance,
                require_surface_zero=self._param("require_surface_zero"),
            )
        )
        self._latest_pressure = None

        self._pose_pub = self.create_publisher(PoseWithCovarianceStamped, "~/pose", 20)
        self._depth_pub = self.create_publisher(Float64, "~/depth", 20)
        self.create_subscription(
            FluidPressure, "~/raw", self._on_raw, qos_profile_sensor_data
        )
        self._zero_service = self.create_service(
            Trigger, "~/zero_surface", self._on_zero_surface
        )

        if self._param("require_surface_zero"):
            self.get_logger().info(
                "depth_gate: waiting for a surface zero - call ~/zero_surface with "
                "the vehicle floating before arming"
            )

    def _on_raw(self, msg: FluidPressure) -> None:
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self._latest_pressure = float(msg.fluid_pressure)

        verdict = self._converter.check(
            DepthSample(stamp=stamp, pressure_pa=self._latest_pressure),
            self.now_seconds(),
        )
        if not verdict.accepted:
            self.reject(verdict.reason)
            return

        self.accept(verdict.degraded_reason)

        depth = Float64()
        depth.data = float(verdict.depth_m)
        self._depth_pub.publish(depth)

        pose = PoseWithCovarianceStamped()
        pose.header.stamp = msg.header.stamp
        pose.header.frame_id = self._odom_frame
        # ENU: z is up, so a positive depth is a negative z.
        pose.pose.pose.position.z = -float(verdict.depth_m)
        pose.pose.pose.orientation.w = 1.0

        covariance = [0.0] * 36
        covariance[0] = UNMEASURED_VARIANCE     # x
        covariance[7] = UNMEASURED_VARIANCE     # y
        covariance[14] = float(self._depth_variance)
        for axis in range(3, 6):                # roll, pitch, yaw
            covariance[axis * 6 + axis] = UNMEASURED_VARIANCE
        pose.pose.covariance = covariance

        self._pose_pub.publish(pose)

    def _on_zero_surface(self, request, response):
        del request
        if self._latest_pressure is None:
            response.success = False
            response.message = "no pressure sample received yet"
            return response

        if self._converter.zero_at(self._latest_pressure):
            response.success = True
            response.message = (
                f"surface reference set to {self._converter.surface_pressure_pa:.0f} Pa"
            )
            self.get_logger().info(response.message)
        else:
            response.success = False
            response.message = (
                f"{self._latest_pressure:.0f} Pa is not a plausible surface reading; "
                "is the vehicle floating?"
            )
            self.get_logger().warn(response.message)
        return response


def main(args=None):
    rclpy.init(args=args)
    node = DepthGate()
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
