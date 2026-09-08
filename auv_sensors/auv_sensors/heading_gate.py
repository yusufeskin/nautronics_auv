"""Layer 1 gate for an absolute heading source.

Symmetric with ``depth_gate``: that one supplies the absolute z the estimator
cannot get any other way, this one supplies the absolute yaw. Between them
they are the only two sensors on the vehicle that measure a coordinate rather
than a rate, and they are what stop the estimate from drifting away.

The source can be anything that publishes an orientation - a compass, an
AHRS, a gyrocompass. The gate does not care which, but it does cross-check
every heading against the gyro from ``imu_gate``, because a magnetically
disturbed heading is biased rather than noisy and is worse than no heading at
all. See ``heading_validation.py`` for why that check is the heart of this
node.

If no absolute heading source is fitted, do not run this node. The estimator
falls back to integrating the gyro, yaw drifts, and the vehicle navigates
relative to where it started - which is the situation this gate exists to
improve on, not a failure mode.

A note on declination: it does not matter here. The vehicle has no map frame
and no notion of true north, so what is needed from a heading source is that
it be *stable*, not that it be correct. A constant offset from true north is
harmless; an offset that changes as the vehicle moves past the pool's rebar
is exactly what the gyro cross-check is looking for.
"""

from __future__ import annotations

import math

import numpy as np
import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu, MagneticField
from std_msgs.msg import Float64

from auv_core import frames
from auv_core.gate import SensorGateNode
from auv_core.vehicle_config import default_config_path, load_vehicle_config
from auv_sensors.heading_validation import (
    HeadingLimits,
    HeadingSample,
    HeadingValidator,
)

# Position and tilt are not observed by a heading source.
UNMEASURED_VARIANCE = 1.0e6


class HeadingGate(SensorGateNode):
    def __init__(self):
        super().__init__(
            node_name="heading_gate",
            sensor_name="heading",
            default_expected_rate_hz=20.0,
            default_stale_timeout_s=0.5,
        )

        self.declare_parameter("vehicle_config", default_config_path())
        self.declare_parameter("max_clock_skew_s", 0.2)
        self.declare_parameter("max_tilt_deg", 60.0)
        self.declare_parameter("max_yaw_rate_mismatch_deg_s", 8.0)
        self.declare_parameter("max_compare_dt_s", 1.0)
        self.declare_parameter("max_gyro_dt_s", 0.2)
        self.declare_parameter("disturbed_streak", 3)
        self.declare_parameter("recover_streak", 30)
        self.declare_parameter("expected_field_magnitude_t", 0.0)
        self.declare_parameter("field_magnitude_tolerance", 0.25)
        self.declare_parameter("yaw_accuracy_deg", 5.0)
        self.declare_parameter("degraded_variance_scale", 9.0)

        config = load_vehicle_config(self.get_parameter("vehicle_config").value)
        self._odom_frame = config.frames["odom"]
        mount = config.sensor("heading")
        # The device reports its own orientation; the vehicle's is that
        # composed with the inverse of the device's mount rotation. A mount
        # yaw error therefore shows up as a constant heading bias, which is
        # why vehicle.yaml marks it MEASURE.
        self._mount_inverse = frames.quat_conjugate(mount.quat)

        self._validator = HeadingValidator(
            limits=HeadingLimits(
                max_clock_skew_s=self._param("max_clock_skew_s"),
                max_tilt_rad=math.radians(self._param("max_tilt_deg")),
                max_yaw_rate_mismatch_rad_s=math.radians(
                    self._param("max_yaw_rate_mismatch_deg_s")
                ),
                max_compare_dt_s=self._param("max_compare_dt_s"),
                max_gyro_dt_s=self._param("max_gyro_dt_s"),
                disturbed_streak=self._param("disturbed_streak"),
                recover_streak=self._param("recover_streak"),
                expected_field_magnitude_t=self._param("expected_field_magnitude_t"),
                field_magnitude_tolerance=self._param("field_magnitude_tolerance"),
                default_yaw_variance=math.radians(self._param("yaw_accuracy_deg")) ** 2,
                degraded_variance_scale=self._param("degraded_variance_scale"),
            )
        )
        self._latest_field = None
        self._was_disturbed = False

        self._pose_pub = self.create_publisher(PoseWithCovarianceStamped, "~/pose", 20)
        self._yaw_pub = self.create_publisher(Float64, "~/yaw_deg", 20)

        self.create_subscription(Imu, "~/raw", self._on_raw, qos_profile_sensor_data)
        # The gyro reference for the cross-check. This is imu_gate's validated,
        # bias-corrected output, not the raw driver stream.
        self.create_subscription(Imu, "~/gyro", self._on_gyro, qos_profile_sensor_data)
        self.create_subscription(
            MagneticField, "~/magnetic_field", self._on_field, qos_profile_sensor_data
        )

        if self._param("expected_field_magnitude_t") <= 0.0:
            self.get_logger().info(
                "heading_gate: magnetic field magnitude check disabled "
                "(set expected_field_magnitude_t from a reading taken in open "
                "air at the venue to enable it)"
            )

    # -- callbacks ---------------------------------------------------------

    def _on_gyro(self, msg: Imu) -> None:
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self._validator.update_gyro(
            stamp,
            (msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z),
        )

    def _on_field(self, msg: MagneticField) -> None:
        self._latest_field = np.array(
            [msg.magnetic_field.x, msg.magnetic_field.y, msg.magnetic_field.z]
        )

    def _on_raw(self, msg: Imu) -> None:
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        device_quat = np.array(
            [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
        )
        vehicle_quat = frames.quat_multiply(device_quat, self._mount_inverse)

        verdict = self._validator.check(
            HeadingSample(
                stamp=stamp,
                orientation=vehicle_quat,
                magnetic_field=self._latest_field,
            ),
            self.now_seconds(),
        )

        if self._validator.disturbed and not self._was_disturbed:
            self.get_logger().warn(
                "heading source disagrees with the gyro; treating it as "
                "magnetically disturbed and holding it out of the estimate"
            )
        elif self._was_disturbed and not self._validator.disturbed:
            self.get_logger().info("heading source agrees with the gyro again")
        self._was_disturbed = self._validator.disturbed

        if not verdict.accepted:
            self.reject(verdict.reason)
            return

        self.accept(verdict.degraded_reason)
        self._publish(msg, verdict)

    # -- publishing --------------------------------------------------------

    def _publish(self, msg: Imu, verdict) -> None:
        yaw = Float64()
        yaw.data = float(math.degrees(verdict.yaw_rad))
        self._yaw_pub.publish(yaw)

        pose = PoseWithCovarianceStamped()
        pose.header.stamp = msg.header.stamp
        pose.header.frame_id = self._odom_frame

        # Yaw only. Roll and pitch belong to the attitude filter, which gets
        # them from gravity; forwarding this source's tilt as well would
        # double-count the same accelerometer.
        quat = frames.quat_from_euler_rpy(0.0, 0.0, verdict.yaw_rad)
        pose.pose.pose.orientation.x = float(quat[0])
        pose.pose.pose.orientation.y = float(quat[1])
        pose.pose.pose.orientation.z = float(quat[2])
        pose.pose.pose.orientation.w = float(quat[3])

        covariance = [0.0] * 36
        for axis in range(3):                    # x, y, z
            covariance[axis * 6 + axis] = UNMEASURED_VARIANCE
        covariance[21] = UNMEASURED_VARIANCE     # roll
        covariance[28] = UNMEASURED_VARIANCE     # pitch
        covariance[35] = float(verdict.yaw_variance)
        pose.pose.covariance = covariance

        self._pose_pub.publish(pose)


def main(args=None):
    rclpy.init(args=args)
    node = HeadingGate()
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
