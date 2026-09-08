"""Layer 1 gate for the Doppler velocity log.

Turns a raw ``marine_acoustic_msgs/Dvl`` stream - from the Water Linked A50
driver or from the simulator bridge, which are shaped identically - into a
validated FLU body velocity the estimator can fuse, plus a bottom-tracking
altitude that most teams leave on the floor.

What this node deliberately does not do: integrate velocity into a position,
interpolate attitude, or send anything to the autopilot. Those belong to the
estimator and the autopilot bridge respectively. Mixing them into the DVL
handler is what makes a DVL stack impossible to test.
"""

from __future__ import annotations

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Range

from auv_core.gate import SensorGateNode
from auv_core.vehicle_config import default_config_path, load_vehicle_config
from auv_sensors.dvl_validation import (
    DvlLimits,
    DvlSample,
    DvlValidator,
    altitude_is_valid,
)

try:
    from marine_acoustic_msgs.msg import Dvl
except ImportError as exc:  # pragma: no cover - build-time dependency
    raise ImportError(
        "marine_acoustic_msgs is required by dvl_gate; it is pulled in by "
        "nautronics.repos, run 'vcs import < nautronics.repos' and rebuild"
    ) from exc

# Angular velocity is not measured by a DVL. Publishing a large covariance
# rather than zero keeps a filter that reads the whole 6x6 from treating an
# unmeasured axis as perfectly known.
UNMEASURED_VARIANCE = 1.0e6


class DvlGate(SensorGateNode):
    def __init__(self):
        super().__init__(
            node_name="dvl_gate",
            sensor_name="dvl",
            default_expected_rate_hz=8.0,
            default_stale_timeout_s=0.75,
        )

        self.declare_parameter("vehicle_config", default_config_path())
        self.declare_parameter("input_is_frd", True)
        self.declare_parameter("min_good_beams", 3)
        self.declare_parameter("beams_for_full_confidence", 4)
        self.declare_parameter("max_speed_mps", 2.0)
        self.declare_parameter("max_accel_mps2", 5.0)
        self.declare_parameter("max_dt_s", 1.0)
        self.declare_parameter("lock_loss_streak", 4)
        self.declare_parameter("lock_regain_streak", 2)
        self.declare_parameter("default_velocity_variance", 0.0025)
        self.declare_parameter("degraded_variance_scale", 4.0)
        self.declare_parameter("allow_water_track", False)
        self.declare_parameter("max_clock_skew_s", 0.5)
        self.declare_parameter("beam_aperture_rad", 0.0366)  # 2.1 deg, A50 spec
        self.declare_parameter("altitude_min_m", 0.05)
        self.declare_parameter("altitude_max_m", 50.0)

        config = load_vehicle_config(self.get_parameter("vehicle_config").value)
        self._odom_frame = config.frames["odom"]
        self._dvl_frame = config.sensor("dvl").frame

        self._validator = DvlValidator(
            limits=DvlLimits(
                min_good_beams=self._param("min_good_beams"),
                beams_for_full_confidence=self._param("beams_for_full_confidence"),
                max_speed_mps=self._param("max_speed_mps"),
                max_accel_mps2=self._param("max_accel_mps2"),
                max_dt_s=self._param("max_dt_s"),
                lock_loss_streak=self._param("lock_loss_streak"),
                lock_regain_streak=self._param("lock_regain_streak"),
                default_velocity_variance=self._param("default_velocity_variance"),
                degraded_variance_scale=self._param("degraded_variance_scale"),
                allow_water_track=self._param("allow_water_track"),
                max_clock_skew_s=self._param("max_clock_skew_s"),
            ),
            input_is_frd=self._param("input_is_frd"),
        )

        self._velocity_pub = self.create_publisher(Odometry, "~/velocity", 10)
        self._altitude_pub = self.create_publisher(Range, "~/altitude", 10)
        self.create_subscription(Dvl, "~/raw", self._on_raw, qos_profile_sensor_data)

        self.get_logger().info(
            f"dvl_gate: '~/raw' -> '~/velocity' in {self._dvl_frame}, "
            f"input convention {'FRD' if self._validator.input_is_frd else 'FLU'}"
        )

    # -- callback ----------------------------------------------------------

    def _on_raw(self, msg: Dvl) -> None:
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        sample = DvlSample(
            stamp=stamp,
            velocity=np.array([msg.velocity.x, msg.velocity.y, msg.velocity.z]),
            num_good_beams=int(msg.num_good_beams),
            beams_valid=bool(msg.beam_velocities_valid),
            velocity_mode=int(msg.velocity_mode),
            covariance=np.asarray(msg.velocity_covar, dtype=float),
            altitude=float(msg.altitude),
        )

        was_locked = self._validator.locked
        verdict = self._validator.check(sample, self.now_seconds())

        if was_locked and not verdict.locked:
            self.get_logger().warn(
                "bottom lock lost; velocity aiding stops until it is regained"
            )
        elif verdict.locked and not was_locked:
            self.get_logger().info("bottom lock acquired")

        if not verdict.accepted:
            self.reject(verdict.reason)
            return

        self.accept(verdict.degraded_reason)

        # Altitude is published whenever the beams report one, even before
        # lock is declared - it is a range measurement, not a velocity, and
        # bottom-following wants it as early as possible.
        if altitude_is_valid(sample.altitude):
            self._publish_altitude(msg, sample.altitude)

        # Velocity waits for the lock hysteresis to settle, so a single good
        # ping in the middle of a dropout does not restart aiding.
        if verdict.locked:
            self._publish_velocity(msg, verdict)

    # -- publishing --------------------------------------------------------

    def _publish_velocity(self, msg: Dvl, verdict) -> None:
        out = Odometry()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self._odom_frame
        # The twist is measured in the DVL's frame; the estimator uses the
        # static base_link->dvl_link transform to bring it to the body. That
        # is why the mount rotation in vehicle.yaml has to be measured.
        out.child_frame_id = self._dvl_frame

        out.twist.twist.linear.x = float(verdict.velocity_flu[0])
        out.twist.twist.linear.y = float(verdict.velocity_flu[1])
        out.twist.twist.linear.z = float(verdict.velocity_flu[2])

        covariance = [0.0] * 36
        for row in range(3):
            for col in range(3):
                covariance[row * 6 + col] = float(verdict.covariance_flu[row, col])
        for axis in range(3, 6):
            covariance[axis * 6 + axis] = UNMEASURED_VARIANCE
        out.twist.covariance = covariance

        self._velocity_pub.publish(out)

    def _publish_altitude(self, msg: Dvl, altitude: float) -> None:
        out = Range()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self._dvl_frame
        out.radiation_type = Range.ULTRASOUND
        out.field_of_view = float(self._param("beam_aperture_rad"))
        out.min_range = float(self._param("altitude_min_m"))
        out.max_range = float(self._param("altitude_max_m"))
        out.range = float(altitude)
        self._altitude_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = DvlGate()
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
