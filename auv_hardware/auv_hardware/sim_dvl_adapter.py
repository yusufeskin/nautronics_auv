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

# The real driver marks orientation covariance unknown, because the A50 does
# not report one (waterlinked_dvl_driver.cpp:321-325).
UNKNOWN_ORIENTATION_COVAR = -1.0


class SimDvlAdapter(Node):
    def __init__(self):
        super().__init__('sim_dvl_adapter')

        self.declare_parameter('velocity_variance', 0.0025)
        self.declare_parameter('force_invalid', False)
        self.declare_parameter('drop_ratio', 0.0)
        self.declare_parameter('child_frame_id', 'dvl_link')
        # Growth of the dead-reckoning position std per metre travelled. The
        # A50 datasheet quotes error as a percentage of distance; the device
        # reports the resulting std in its DeadReckoningReport, which the real
        # driver copies straight into pose.covariance[0/7/14].
        self.declare_parameter('dead_reckoning_drift_ratio', 0.02)
        self.declare_parameter('dead_reckoning_frame_id', 'odom')
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

        # Integrated dead-reckoning state.
        self._dr_position = [0.0, 0.0, 0.0]
        self._dr_distance = 0.0
        self._dr_prev_stamp = None
        self._orientation = None

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
        out.dvl_type = msg.dvl_type
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
        self._orientation = (q.x, q.y, q.z, q.w)

    def publish_dead_reckoning(self, msg: Dvl, force_invalid):
        """Integrate DVL velocity into a position, like the A50 does onboard.

        The real device integrates internally and reports the result; here the
        same integration is done on the (already noisy) simulated velocity, so
        the output drifts the way bottom-track dead reckoning really does.
        """
        stamp = msg.header.stamp
        t = stamp.sec + stamp.nanosec * 1.0e-9

        if self._dr_prev_stamp is None:
            self._dr_prev_stamp = t
            return

        dt = t - self._dr_prev_stamp
        self._dr_prev_stamp = t
        # A backwards or absurd step means the clock jumped (a sim reset, or
        # use_sim_time toggling); integrating across it would corrupt position.
        if dt <= 0.0 or dt > 1.0:
            return

        # An invalid report carries no usable velocity, so hold position and
        # let the covariance keep growing - the same visible behaviour as the
        # real unit losing bottom lock.
        if not force_invalid and msg.beam_velocities_valid:
            if self._orientation is None:
                # No attitude yet: integrate in the body frame. Correct only
                # while the vehicle is level, which is why it is not the
                # steady state - it just avoids dropping the first samples.
                dx, dy, dz = msg.velocity.x, msg.velocity.y, msg.velocity.z
            else:
                dx, dy, dz = self._rotate_to_world(
                    msg.velocity.x, msg.velocity.y, msg.velocity.z
                )
            self._dr_position[0] += dx * dt
            self._dr_position[1] += dy * dt
            self._dr_position[2] += dz * dt
            self._dr_distance += math.sqrt(dx * dx + dy * dy + dz * dz) * dt

        drift_ratio = self.get_parameter('dead_reckoning_drift_ratio').value
        std = drift_ratio * self._dr_distance

        out = PoseWithCovarianceStamped()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self.get_parameter('dead_reckoning_frame_id').value
        out.pose.pose.position.x = self._dr_position[0]
        out.pose.pose.position.y = self._dr_position[1]
        out.pose.pose.position.z = self._dr_position[2]
        if self._orientation is not None:
            (out.pose.pose.orientation.x, out.pose.pose.orientation.y,
             out.pose.pose.orientation.z, out.pose.pose.orientation.w) = self._orientation
        else:
            out.pose.pose.orientation.w = 1.0

        # Same layout the real driver writes: position std on the diagonal,
        # orientation marked unknown (waterlinked_dvl_driver.cpp:317-325).
        out.pose.covariance[0] = std
        out.pose.covariance[7] = std
        out.pose.covariance[14] = std
        out.pose.covariance[21] = UNKNOWN_ORIENTATION_COVAR
        out.pose.covariance[28] = UNKNOWN_ORIENTATION_COVAR
        out.pose.covariance[35] = UNKNOWN_ORIENTATION_COVAR

        self.dead_reckoning_pub.publish(out)

    def _rotate_to_world(self, vx, vy, vz):
        """Rotate a body-frame vector by the current attitude quaternion."""
        qx, qy, qz, qw = self._orientation
        # v' = v + 2 * q_vec x (q_vec x v + w * v)
        tx = 2.0 * (qy * vz - qz * vy)
        ty = 2.0 * (qz * vx - qx * vz)
        tz = 2.0 * (qx * vy - qy * vx)
        return (
            vx + qw * tx + (qy * tz - qz * ty),
            vy + qw * ty + (qz * tx - qx * tz),
            vz + qw * tz + (qx * ty - qy * tx),
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
