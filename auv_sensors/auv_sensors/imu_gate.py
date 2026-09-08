"""Layer 1 gate for the inertial measurement unit.

Validates the IMU stream, estimates and removes the gyro turn-on bias, and -
by default - strips the orientation field.

Dropping orientation is a deliberate policy, not an oversight. A fused
orientation from a consumer IMU is magnetometer-aided, and a competition pool
is a steel box: NTU Mecatron's 2026 report describes IMU data "jumping"
underwater until they moved to a relative heading mode and recalibrated away
from magnetic influence, and NUS Bumblebee simply bolt a fibre-optic gyro to
the vehicle rather than trust a magnetic heading. Feeding a mag-aided
orientation into the estimator imports the pool's rebar into the yaw
solution. The estimator integrates the (bias-corrected) gyro instead, and
yaw drift is bounded by landmark observations from Layer 3.
"""

from __future__ import annotations

import numpy as np
import rclpy
from geometry_msgs.msg import Vector3Stamped
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu

from auv_core.gate import SensorGateNode
from auv_core.vehicle_config import default_config_path, load_vehicle_config
from auv_sensors.imu_validation import (
    GyroBiasEstimator,
    ImuLimits,
    ImuSample,
    ImuValidator,
)

# sensor_msgs/Imu convention: a leading -1 in a covariance means the field is
# not provided at all, which is exactly what we want to say about orientation.
NO_ESTIMATE = -1.0


class ImuGate(SensorGateNode):
    def __init__(self):
        super().__init__(
            node_name="imu_gate",
            sensor_name="imu",
            default_expected_rate_hz=100.0,
            default_stale_timeout_s=0.1,
        )

        self.declare_parameter("vehicle_config", default_config_path())
        self.declare_parameter("use_orientation", False)
        self.declare_parameter("subtract_gyro_bias", True)
        self.declare_parameter("max_gyro_rad_s", 8.0)
        self.declare_parameter("max_accel_mps2", 58.8)
        self.declare_parameter("max_clock_skew_s", 0.2)
        self.declare_parameter("default_gyro_variance", 4.0e-6)
        self.declare_parameter("default_accel_variance", 4.0e-4)
        self.declare_parameter("bias_still_gyro_threshold_rad_s", 0.02)
        self.declare_parameter("bias_still_accel_tolerance_mps2", 0.5)
        self.declare_parameter("bias_settle_time_s", 20.0)
        self.declare_parameter("bias_min_samples", 200)

        config = load_vehicle_config(self.get_parameter("vehicle_config").value)
        self._imu_frame = config.sensor("imu").frame

        self._use_orientation = self._param("use_orientation")
        self._default_gyro_variance = self._param("default_gyro_variance")
        self._default_accel_variance = self._param("default_accel_variance")

        self._validator = ImuValidator(
            limits=ImuLimits(
                max_gyro_rad_s=self._param("max_gyro_rad_s"),
                max_accel_mps2=self._param("max_accel_mps2"),
                max_clock_skew_s=self._param("max_clock_skew_s"),
                default_gyro_variance=self._default_gyro_variance,
                default_accel_variance=self._default_accel_variance,
            ),
            bias_estimator=GyroBiasEstimator(
                still_gyro_threshold_rad_s=self._param("bias_still_gyro_threshold_rad_s"),
                still_accel_tolerance_mps2=self._param("bias_still_accel_tolerance_mps2"),
                settle_time_s=self._param("bias_settle_time_s"),
                min_samples=self._param("bias_min_samples"),
            ),
            subtract_bias=self._param("subtract_gyro_bias"),
        )
        self._bias_was_ready = False

        self._data_pub = self.create_publisher(Imu, "~/data", 20)
        self._bias_pub = self.create_publisher(Vector3Stamped, "~/gyro_bias", 10)
        self.create_subscription(Imu, "~/raw", self._on_raw, qos_profile_sensor_data)

        if self._use_orientation:
            self.get_logger().warn(
                "use_orientation is true: the driver's fused orientation will be "
                "forwarded. If that fusion uses a magnetometer, expect heading "
                "errors near pool structures."
            )
        else:
            self.get_logger().info(
                "imu_gate: forwarding gyro and accelerometer only, orientation dropped"
            )

    def _on_raw(self, msg: Imu) -> None:
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        sample = ImuSample(
            stamp=stamp,
            angular_velocity=np.array(
                [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]
            ),
            linear_acceleration=np.array(
                [
                    msg.linear_acceleration.x,
                    msg.linear_acceleration.y,
                    msg.linear_acceleration.z,
                ]
            ),
        )

        verdict = self._validator.check(sample, self.now_seconds())
        if not verdict.accepted:
            self.reject(verdict.reason)
            return

        self.accept(verdict.degraded_reason)

        estimator = self._validator.bias_estimator
        if estimator.ready and not self._bias_was_ready:
            self._bias_was_ready = True
            bias_deg_s = np.degrees(estimator.bias)
            self.get_logger().info(
                "gyro bias converged: "
                f"[{bias_deg_s[0]:+.3f}, {bias_deg_s[1]:+.3f}, {bias_deg_s[2]:+.3f}] deg/s"
            )

        self._publish_data(msg, verdict)
        self._publish_bias(msg, estimator.bias)

    def _publish_data(self, msg: Imu, verdict) -> None:
        out = Imu()
        out.header.stamp = msg.header.stamp
        # Stamped with the mount frame from vehicle.yaml rather than whatever
        # the driver claims, so the estimator's TF lookup resolves against the
        # extrinsics we actually measured.
        out.header.frame_id = self._imu_frame

        out.angular_velocity.x = float(verdict.angular_velocity[0])
        out.angular_velocity.y = float(verdict.angular_velocity[1])
        out.angular_velocity.z = float(verdict.angular_velocity[2])
        out.angular_velocity_covariance = self._covariance(
            msg.angular_velocity_covariance, self._default_gyro_variance
        )

        out.linear_acceleration = msg.linear_acceleration
        out.linear_acceleration_covariance = self._covariance(
            msg.linear_acceleration_covariance, self._default_accel_variance
        )

        if self._use_orientation:
            out.orientation = msg.orientation
            out.orientation_covariance = msg.orientation_covariance
        else:
            out.orientation_covariance = [NO_ESTIMATE] + [0.0] * 8

        self._data_pub.publish(out)

    def _publish_bias(self, msg: Imu, bias) -> None:
        out = Vector3Stamped()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self._imu_frame
        out.vector.x = float(bias[0])
        out.vector.y = float(bias[1])
        out.vector.z = float(bias[2])
        self._bias_pub.publish(out)

    @staticmethod
    def _covariance(reported, default_variance: float):
        """Fall back to a configured variance when the driver reports none.

        A covariance of all zeros means "infinitely trusted", which is never
        true and will make any filter diverge, so an unset field is replaced
        rather than passed through.
        """
        values = list(reported)
        if len(values) == 9 and any(v != 0.0 for v in values) and values[0] >= 0.0:
            return values
        out = [0.0] * 9
        out[0] = out[4] = out[8] = float(default_variance)
        return out


def main(args=None):
    rclpy.init(args=args)
    node = ImuGate()
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
