#!/usr/bin/env python3
"""Guidance: goal pose -> time-parameterised reference for the position controller.

Inputs
  /guidance/goal           geometry_msgs/PoseStamped (odom)  where to end up
  /odometry/filtered       nav_msgs/Odometry                 start of each trajectory
  /controller/at_setpoint  std_msgs/Bool                     used to report ARRIVED
Outputs
  /controller/reference    nav_msgs/Odometry   pose + feedforward velocity, both odom
  /guidance/path           nav_msgs/Path       the planned line, for RViz
  /guidance/status         std_msgs/String     IDLE | ACTIVE | DONE | ARRIVED
                                               DONE = trajectory finished, vehicle still
                                               settling; ARRIVED = controller at_setpoint
Services
  /guidance/cancel         std_srvs/Trigger    stop where the reference is now; the
                                               controller holds that pose

The goal is the contract perception will drive later: the same PoseStamped,
whether a person typed it or a camera found it.

A new goal while one is active replans from the current EKF pose, starting
from rest. When the trajectory ends the final pose is sent with zero velocity
and publishing stops; the controller keeps holding that pose.
"""
import math

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger

from auv_control.trajectory import Limits, StraightLineTrajectory


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def set_yaw(q, yaw):
    q.x, q.y, q.z, q.w = 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)


class GuidanceNode(Node):
    def __init__(self):
        super().__init__('guidance')
        d = Limits()
        self.declare_parameter('rate', 20.0)
        self.declare_parameter('world_frame', 'odom')
        self.declare_parameter('odom_topic', '/odometry/filtered')
        self.declare_parameter('odom_timeout', 0.5)
        self.declare_parameter('max_speed', d.max_speed)
        self.declare_parameter('max_accel', d.max_accel)
        self.declare_parameter('max_vertical_speed', d.max_vertical_speed)
        self.declare_parameter('max_yaw_rate_deg', math.degrees(d.max_yaw_rate))

        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.limits = Limits(g('max_speed'), g('max_accel'), g('max_vertical_speed'),
                             math.radians(g('max_yaw_rate_deg')))
        self.frame = g('world_frame')
        self.odom_timeout = g('odom_timeout')

        self.odom = None
        self.odom_time = None
        self.at_setpoint = False
        self.traj = None
        self.t_start = None
        self.status = 'IDLE'

        self.create_subscription(Odometry, g('odom_topic'), self.odom_cb, 10)
        self.create_subscription(PoseStamped, '/guidance/goal', self.goal_cb, 10)
        self.create_subscription(Bool, '/controller/at_setpoint', self.at_cb, 10)
        self.ref_pub = self.create_publisher(Odometry, '/controller/reference', 10)
        # Latched, so RViz opened after the goal still shows the plan.
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.path_pub = self.create_publisher(Path, '/guidance/path', latched)
        self.status_pub = self.create_publisher(String, '/guidance/status', 10)
        self.create_service(Trigger, '/guidance/cancel', self.cancel_cb)
        self.create_timer(1.0 / g('rate'), self.tick)
        self.create_timer(0.5, lambda: self.status_pub.publish(String(data=self.status)))
        self.get_logger().info(
            f'guidance ready: v={self.limits.max_speed} m/s, a={self.limits.max_accel} m/s^2')

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def odom_cb(self, msg):
        self.odom = msg
        self.odom_time = self.now()

    def at_cb(self, msg):
        self.at_setpoint = msg.data
        if self.status == 'DONE' and msg.data:
            self.status = 'ARRIVED'
            self.get_logger().info('Arrived at goal.')

    def goal_cb(self, msg):
        if msg.header.frame_id and msg.header.frame_id != self.frame:
            self.get_logger().error(f"Goal frame '{msg.header.frame_id}' is not '{self.frame}'; ignored.")
            return
        if self.odom is None or self.now() - self.odom_time > self.odom_timeout:
            self.get_logger().error('No fresh EKF odometry; cannot plan.')
            return
        p0, gp = self.odom.pose.pose.position, msg.pose.position
        yaw0, yaw1 = yaw_of(self.odom.pose.pose.orientation), yaw_of(msg.pose.orientation)
        self.traj = StraightLineTrajectory((p0.x, p0.y, p0.z), yaw0, (gp.x, gp.y, gp.z), yaw1, self.limits)
        self.t_start = self.now()
        self.status = 'ACTIVE'
        self.get_logger().info(
            f'New goal x={gp.x:.2f} y={gp.y:.2f} z={gp.z:.2f} yaw={math.degrees(yaw1):.1f}: '
            f'{self.traj.length:.2f} m, {self.traj.duration:.1f} s')
        self.publish_path()

    def cancel_cb(self, _req, response):
        if self.traj is None:
            response.success, response.message = True, 'Nothing to cancel.'
            return response
        # Freeze the reference where it is now, with zero velocity.
        s = self.traj.sample(self.now() - self.t_start)
        self.publish_reference(s, zero_velocity=True)
        self.traj = None
        self.status = 'IDLE'
        response.success, response.message = True, 'Cancelled; holding current reference.'
        self.get_logger().info(response.message)
        return response

    def publish_reference(self, s, zero_velocity=False):
        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame
        msg.child_frame_id = self.frame   # twist below is in odom, not base_link
        msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z = s.x, s.y, s.z
        set_yaw(msg.pose.pose.orientation, s.yaw)
        if not zero_velocity:
            v = msg.twist.twist
            v.linear.x, v.linear.y, v.linear.z, v.angular.z = s.vx, s.vy, s.vz, s.yaw_rate
        self.ref_pub.publish(msg)

    def publish_path(self):
        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = self.frame
        n = max(2, int(self.traj.duration * 2))
        for i in range(n + 1):
            s = self.traj.sample(self.traj.duration * i / n)
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = s.x, s.y, s.z
            set_yaw(ps.pose.orientation, s.yaw)
            path.poses.append(ps)
        self.path_pub.publish(path)

    def tick(self):
        if self.traj is None:
            return
        t = self.now() - self.t_start
        s = self.traj.sample(t)
        if t >= self.traj.duration:
            self.publish_reference(s, zero_velocity=True)
            self.traj = None
            self.status = 'DONE'   # becomes ARRIVED once the controller reports at_setpoint
            self.get_logger().info('Trajectory finished; holding goal.')
            return
        self.publish_reference(s)


def main(args=None):
    rclpy.init(args=args)
    node = GuidanceNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
