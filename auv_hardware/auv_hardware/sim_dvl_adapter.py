#!/usr/bin/env python3
import random

import rclpy
from rclpy.node import Node
from marine_acoustic_msgs.msg import Dvl
from nav_msgs.msg import Odometry


class SimDvlAdapter(Node):
    def __init__(self):
        super().__init__('sim_dvl_adapter')

        self.declare_parameter('velocity_variance', 0.0025)
        self.declare_parameter('force_invalid', False)
        self.declare_parameter('drop_ratio', 0.0)
        self.declare_parameter('child_frame_id', 'dvl_link')

        self.velocity_report_pub = self.create_publisher(
            Dvl, '/waterlinked_dvl_driver/velocity_report', 10
        )
        self.odom_pub = self.create_publisher(
            Odometry, '/waterlinked_dvl_driver/odom', 10
        )
        self.raw_sub = self.create_subscription(
            Dvl, '/sim/dvl/raw', self.raw_callback, 10
        )

    def raw_callback(self, msg):
        drop_ratio = self.get_parameter('drop_ratio').value
        if drop_ratio > 0.0 and random.random() < drop_ratio:
            return

        force_invalid = self.get_parameter('force_invalid').value
        variance = self.get_parameter('velocity_variance').value
        child_frame_id = self.get_parameter('child_frame_id').value
        stamp = self.get_clock().now().to_msg()

        covar = self._resolve_covar(msg.velocity_covar, variance)

        out = Dvl()
        out.header.stamp = stamp
        out.header.frame_id = msg.header.frame_id
        out.velocity = msg.velocity
        out.velocity_covar = covar
        out.num_good_beams = 0 if force_invalid else 4
        out.beam_velocities_valid = self._resolve_beam_validity(
            out.beam_velocities_valid, force_invalid
        )
        self.velocity_report_pub.publish(out)

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = msg.header.frame_id
        odom.child_frame_id = child_frame_id
        odom.twist.twist.linear.x = msg.velocity.x
        odom.twist.twist.linear.y = msg.velocity.y
        odom.twist.twist.linear.z = msg.velocity.z
        for i in range(3):
            for j in range(3):
                odom.twist.covariance[i * 6 + j] = covar[i * 3 + j]
        self.odom_pub.publish(odom)

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

    @staticmethod
    def _resolve_beam_validity(default_value, force_invalid):
        valid = not force_invalid
        if isinstance(default_value, (list, tuple)):
            length = len(default_value) if len(default_value) else 4
            return [valid] * length
        return valid


def main(args=None):
    rclpy.init(args=args)
    node = SimDvlAdapter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
