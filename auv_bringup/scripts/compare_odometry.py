#!/usr/bin/env python3

import math
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry


class CompareOdometry(Node):
    def __init__(self):
        super().__init__('compare_odometry')

        self.start = [None, None]
        self.start_yaw = [None, None]
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

    @staticmethod
    def get_yaw(q):
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny, cosy)

    def callback(self, msg, index):
        p = msg.pose.pose.position
        position = (p.x, p.y, p.z)

        if self.start[index] is None:
            self.start[index] = position
            self.start_yaw[index] = self.get_yaw(
                msg.pose.pose.orientation
            )

        self.current[index] = position
        self.stamps[index] = (
            msg.header.stamp.sec +
            msg.header.stamp.nanosec * 1e-9
        )

    def displacement(self, index):
        return [
            self.current[index][i] - self.start[index][i]
            for i in range(3)
        ]

    def aligned_displacement(self, index):
        dx, dy, dz = self.displacement(index)
        yaw = self.start_yaw[index]

        c = math.cos(yaw)
        s = math.sin(yaw)

        return [
            c * dx + s * dy,
            -s * dx + c * dy,
            dz
        ]

    @staticmethod
    def distance(a, b):
        return math.sqrt(
            sum((a[i] - b[i]) ** 2 for i in range(3))
        )

    def compare(self):
        if None in self.current:
            return

        if abs(self.stamps[0] - self.stamps[1]) > 0.5:
            return

        gazebo = self.displacement(0)
        ekf = self.displacement(1)

        gazebo_aligned = self.aligned_displacement(0)
        ekf_aligned = self.aligned_displacement(1)

        print(f"Gazebo: {[round(x, 3) for x in gazebo]}")
        print(f"EKF:    {[round(x, 3) for x in ekf]}")
        print(f"Raw error: {self.distance(gazebo, ekf):.3f} m")

        print(
            "Yaw-aligned error: "
            f"{self.distance(gazebo_aligned, ekf_aligned):.3f} m\n"
        )


def main():
    rclpy.init()
    node = CompareOdometry()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
