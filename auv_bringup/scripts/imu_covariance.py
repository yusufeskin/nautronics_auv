#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu


class ImuCovariance(Node):
    def __init__(self):
        super().__init__('imu_covariance_node')

        self.publisher = self.create_publisher(
            Imu, '/imu0/with_covariance', 10
        )

        self.subscriber = self.create_subscription(
            Imu, '/imu0', self.callback, 10
        )

    def callback(self, msg):

        # Orientation covariance (rad^2)
        msg.orientation_covariance = [
            0.0012, 0.0, 0.0,
            0.0, 0.0012, 0.0,
            0.0, 0.0, 0.0076
        ]

        # Angular velocity covariance ((rad/s)^2)
        msg.angular_velocity_covariance = [
            0.000076, 0.0, 0.0,
            0.0, 0.000076, 0.0,
            0.0, 0.0, 0.000076
        ]

        # Linear acceleration covariance ((m/s^2)^2)
        msg.linear_acceleration_covariance = [
            0.01, 0.0, 0.0,
            0.0, 0.01, 0.0,
            0.0, 0.0, 0.01
        ]

        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ImuCovariance()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
