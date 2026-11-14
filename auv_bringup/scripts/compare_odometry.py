#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry


class CompareOdometry(Node):
    def __init__(self):
        super().__init__('compare_odometry')

        self.start = [None, None]
        self.current = [None, None]
        self.stamps = [None, None]

        self.create_subscription(
            Odometry, '/odom',
            lambda msg: self.callback(msg, 0), 10
        )

        self.create_subscription(
            Odometry, '/odometry/filtered',
            lambda msg: self.callback(msg, 1), 10
        )

        self.create_timer(1.0, self.compare)

    def callback(self, msg, index):
        p = msg.pose.pose.position
        position = (p.x, p.y, p.z)

        if self.start[index] is None:
            self.start[index] = position

        self.current[index] = position
        self.stamps[index] = (
            msg.header.stamp.sec +
            msg.header.stamp.nanosec * 1e-9
        )

    def compare(self):
        if None in self.current:
            return

        if abs(self.stamps[0] - self.stamps[1]) > 0.5:
            return

        gazebo = [
            self.current[0][i] - self.start[0][i]
            for i in range(3)
        ]

        ekf = [
            self.current[1][i] - self.start[1][i]
            for i in range(3)
        ]

        error = math.sqrt(sum(
            (gazebo[i] - ekf[i]) ** 2
            for i in range(3)
        ))

        print(f"Gazebo: {[round(x, 3) for x in gazebo]}")
        print(f"EKF:    {[round(x, 3) for x in ekf]}")
        print(f"Error:  {error:.3f} m\n")


def main():
    rclpy.init()
    node = CompareOdometry()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
