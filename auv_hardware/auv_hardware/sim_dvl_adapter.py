#!/usr/bin/env python3
"""Makes the simulated DVL indistinguishable from the real hardware.

Republishes /sim/dvl/raw onto the exact three topics the real WaterLinked A50
driver publishes, so nothing downstream needs to know whether it is talking to
Gazebo or to the device:

    /waterlinked_dvl_driver/velocity_report     marine_acoustic_msgs/Dvl
    /waterlinked_dvl_driver/odom                nav_msgs/Odometry  (twist only)
    /waterlinked_dvl_driver/dead_reckoning_report
                                    geometry_msgs/PoseWithCovarianceStamped

Topic names and message types mirror waterlinked_dvl_driver.cpp:178-181, where
they are declared relative ("~/velocity_report") under the node name
waterlinked_dvl_driver. Keep the two in sync if the driver is ever updated.
"""
import math
import random

import rclpy
from rclpy.node import Node
from marine_acoustic_msgs.msg import Dvl
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped
from geometry_msgs.msg import TransformStamped
from rclpy.time import Time
from tf2_ros import (
    Buffer,
    TransformListener,
    StaticTransformBroadcaster,
    TransformException,
)

# The real driver marks orientation covariance unknown, because the A50 does
# not report one (waterlinked_dvl_driver.cpp:321-325).
UNKNOWN_ORIENTATION_COVAR = -1.0


class SimDvlAdapter(Node):
    def __init__(self):
        super().__init__('sim_dvl_adapter')

        self.declare_parameter('velocity_variance', 0.0025)
        self.declare_parameter('force_invalid', False)
        self.declare_parameter('drop_ratio', 0.0)
        self.declare_parameter('child_frame_id', 'dvl_frd')
        # Growth of the dead-reckoning position std per metre travelled. The
        # A50 datasheet quotes error as a percentage of distance; the device
        # reports the resulting std in its DeadReckoningReport, which the real
        # driver copies straight into pose.covariance[0/7/14].
        self.declare_parameter('dead_reckoning_drift_ratio', 0.02)
        # The real driver stamps every message, dead reckoning included, with
        # its single frame_id parameter (waterlinked_dvl_driver.cpp:145-148).
        self.declare_parameter('dead_reckoning_frame_id', 'dvl_dr')
        # Ground-truth attitude stand-in. On the real vehicle the A50 is given
        # attitude over its own API and integrates internally; in sim we take
        # the vehicle pose Gazebo already publishes rather than modelling an
        # AHRS. This is the one place the emulation is optimistic: real
        # dead-reckoning also drifts with attitude error, this does not.
        self.declare_parameter('attitude_topic', '/odom')

        self.velocity_report_pub = self.create_publisher(
            Dvl, '/waterlinked_dvl_driver/velocity_report', 10
        )
        self.odom_pub = self.create_publisher(
            Odometry, '/waterlinked_dvl_driver/odom', 10
        )
        self.dead_reckoning_pub = self.create_publisher(
            PoseWithCovarianceStamped,
            '/waterlinked_dvl_driver/dead_reckoning_report', 10
        )
        self.raw_sub = self.create_subscription(
            Dvl, '/sim/dvl/raw', self.raw_callback, 10
        )
        self.attitude_sub = self.create_subscription(
            Odometry, self.get_parameter('attitude_topic').value,
            self.attitude_callback, 10
        )


        # Dead-reckoning state
        self._dr_position = [0.0, 0.0, 0.0]
        self._dr_distance = 0.0
        self._dr_prev_stamp = None

        # Current DVL orientation in odom
        self._orientation = None

        # Initial, fixed dead-reckoning reference
        self._dr_ref_q = None

        # DVL mounting transform relative to base_link
        self._sensor_mount = None

        # Read the fixed base_link -> dvl_frd transform
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(
            self._tf_buffer, self
        )

        # Publish the initial odom -> dvl_dr transform
        self._static_broadcaster = StaticTransformBroadcaster(self)

    def raw_callback(self, msg):
        drop_ratio = self.get_parameter('drop_ratio').value
        if drop_ratio > 0.0 and random.random() < drop_ratio:
            return

        force_invalid = self.get_parameter('force_invalid').value
        variance = self.get_parameter('velocity_variance').value
        child_frame_id = self.get_parameter('child_frame_id').value

        covar = self._resolve_covar(msg.velocity_covar, variance)

        out = Dvl()
        out.header = msg.header
        out.velocity_mode = msg.velocity_mode
        # The real driver hard-codes PISTON (waterlinked_dvl_driver.cpp:151),
        # whatever beam model the Gazebo sensor uses.
        out.dvl_type = Dvl.DVL_TYPE_PISTON
        out.velocity = msg.velocity
        out.velocity_covar = covar
        out.course_gnd = msg.course_gnd
        out.speed_gnd = msg.speed_gnd
        out.altitude = msg.altitude
        out.sound_speed = msg.sound_speed
        out.num_good_beams = 0 if force_invalid else msg.num_good_beams
        out.beam_ranges_valid = False if force_invalid else msg.beam_ranges_valid
        out.beam_velocities_valid = False if force_invalid else msg.beam_velocities_valid
        out.beam_unit_vec = msg.beam_unit_vec
        out.range = msg.range
        out.range_covar = msg.range_covar
        out.beam_quality = msg.beam_quality
        out.beam_velocity = msg.beam_velocity
        out.beam_velocity_covar = msg.beam_velocity_covar
        self.velocity_report_pub.publish(out)

        # The real driver still publishes velocity_report for an invalid
        # sample, but skips odom (waterlinked_dvl_driver.cpp:283-286).
        if not out.beam_velocities_valid:
            self.get_logger().warn(
                'DVL reported an invalid velocity; skipping odometry publication.',
                throttle_duration_sec=5.0,
            )
            self.publish_dead_reckoning(msg, force_invalid)
            return

        odom = Odometry()
        odom.header = msg.header
        odom.child_frame_id = child_frame_id
        odom.twist.twist.linear.x = msg.velocity.x
        odom.twist.twist.linear.y = msg.velocity.y
        odom.twist.twist.linear.z = msg.velocity.z
        for i in range(3):
            for j in range(3):
                odom.twist.covariance[i * 6 + j] = covar[i * 3 + j]
        self.odom_pub.publish(odom)

        self.publish_dead_reckoning(msg, force_invalid)


    def attitude_callback(self, msg: Odometry):
        q = msg.pose.pose.orientation

        q_world_base = self._normalize_quaternion(
            (q.x, q.y, q.z, q.w)
        )

        if q_world_base is None:
            return

        # Get the fixed mounting transform once.
        if self._sensor_mount is None:
            try:
                tf = self._tf_buffer.lookup_transform(
                    msg.child_frame_id or 'base_link',
                    self.get_parameter('child_frame_id').value,
                    Time()
                )
            except TransformException:
                self.get_logger().warn(
                    'Waiting for base_link -> dvl_frd TF',
                    throttle_duration_sec=5.0
                )
                return

            t = tf.transform.translation
            r = tf.transform.rotation

            mount_q = self._normalize_quaternion(
                (r.x, r.y, r.z, r.w)
            )

            if mount_q is None:
                return

            self._sensor_mount = (
                (t.x, t.y, t.z),
                mount_q
            )

        mount_t, mount_q = self._sensor_mount

        # Current DVL orientation in the odom frame.
        q_world_dvl = self._quat_multiply(
            q_world_base, mount_q
        )

        self._orientation = q_world_dvl

        if self._dr_ref_q is not None:
            return

        # DVL origin expressed in odom.
        p = msg.pose.pose.position
        offset = self._rotate_vector(
            q_world_base, mount_t
        )

        origin = (
            p.x + offset[0],
            p.y + offset[1],
            p.z + offset[2]
        )

        # Freeze the initial DVL orientation.
        self._dr_ref_q = q_world_dvl

        transform = TransformStamped()
        transform.header.stamp = msg.header.stamp
        transform.header.frame_id = (
            msg.header.frame_id or 'odom'
        )
        transform.child_frame_id = (
            self.get_parameter(
                'dead_reckoning_frame_id'
            ).value
        )

        transform.transform.translation.x = origin[0]
        transform.transform.translation.y = origin[1]
        transform.transform.translation.z = origin[2]

        (
            transform.transform.rotation.x,
            transform.transform.rotation.y,
            transform.transform.rotation.z,
            transform.transform.rotation.w
        ) = self._dr_ref_q

        self._static_broadcaster.sendTransform(transform)

        self.get_logger().info(
            'Initialized DVL dead-reckoning reference frame'
        )


    def publish_dead_reckoning(self, msg: Dvl, force_invalid):
        # Do not integrate until the initial reference exists.
        if self._dr_ref_q is None or self._orientation is None:
            return

        stamp = msg.header.stamp
        t = stamp.sec + stamp.nanosec * 1.0e-9

        if self._dr_prev_stamp is None:
            self._dr_prev_stamp = t
            return

        dt = t - self._dr_prev_stamp
        self._dr_prev_stamp = t

        # Reject invalid or excessively large time steps.
        if dt <= 0.0 or dt > 1.0:
            return

        if not force_invalid and msg.beam_velocities_valid:

            # DVL velocity is expressed in FRD.
            v_frd = (
                msg.velocity.x,
                msg.velocity.y,
                msg.velocity.z
            )

            # Current DVL frame -> odom frame.
            v_world = self._rotate_vector(
                self._orientation, v_frd
            )

            # odom frame -> initial DVL reference frame.
            v_dr = self._rotate_vector(
                self._quat_conjugate(self._dr_ref_q),
                v_world
            )

            # Integrate velocity in the fixed reference frame.
            for i in range(3):
                self._dr_position[i] += v_dr[i] * dt

            self._dr_distance += math.sqrt(
                sum(v * v for v in v_dr)
            ) * dt

        drift_ratio = self.get_parameter(
            'dead_reckoning_drift_ratio'
        ).value

        std = drift_ratio * self._dr_distance
        variance = std * std

        out = PoseWithCovarianceStamped()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self.get_parameter(
            'dead_reckoning_frame_id'
        ).value

        out.pose.pose.position.x = self._dr_position[0]
        out.pose.pose.position.y = self._dr_position[1]
        out.pose.pose.position.z = self._dr_position[2]

        # Current DVL orientation relative to initial DVL.
        q_relative = self._quat_multiply(
            self._quat_conjugate(self._dr_ref_q),
            self._orientation
        )

        (
            out.pose.pose.orientation.x,
            out.pose.pose.orientation.y,
            out.pose.pose.orientation.z,
            out.pose.pose.orientation.w
        ) = q_relative

        # ROS covariance expects variance, not stddev.
        out.pose.covariance[0] = variance
        out.pose.covariance[7] = variance
        out.pose.covariance[14] = variance

        # Preserve the existing driver's unknown-orientation
        # convention for interface compatibility.
        out.pose.covariance[21] = UNKNOWN_ORIENTATION_COVAR
        out.pose.covariance[28] = UNKNOWN_ORIENTATION_COVAR
        out.pose.covariance[35] = UNKNOWN_ORIENTATION_COVAR

        self.dead_reckoning_pub.publish(out)


    @staticmethod
    def _normalize_quaternion(q):
        norm = math.sqrt(sum(x * x for x in q))

        if norm < 1e-12 or not math.isfinite(norm):
            return None

        return tuple(x / norm for x in q)

    @staticmethod
    def _quat_conjugate(q):
        x, y, z, w = q
        return (-x, -y, -z, w)

    @staticmethod
    def _quat_multiply(a, b):
        ax, ay, az, aw = a
        bx, by, bz, bw = b

        return (
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz
        )

    @staticmethod
    def _rotate_vector(q, v):
        qx, qy, qz, qw = q
        vx, vy, vz = v

        tx = 2.0 * (qy * vz - qz * vy)
        ty = 2.0 * (qz * vx - qx * vz)
        tz = 2.0 * (qx * vy - qy * vx)

        return (
            vx + qw * tx + qy * tz - qz * ty,
            vy + qw * ty + qz * tx - qx * tz,
            vz + qw * tz + qx * ty - qy * tx
        )

    @staticmethod
    def _resolve_covar(raw_covar, variance):
        covar = list(raw_covar)
        if len(covar) == 9 and any(covar):
            return covar
        diagonal = [0.0] * 9
        diagonal[0] = variance
        diagonal[4] = variance
        diagonal[8] = variance
        return diagonal


def main(args=None):
    rclpy.init(args=args)
    node = SimDvlAdapter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
