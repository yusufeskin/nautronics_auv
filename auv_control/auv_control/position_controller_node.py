#!/usr/bin/env python3
"""Jetson position controller: EKF pose + reference -> AxisDemand.

Inputs
  /odometry/filtered        nav_msgs/Odometry   EKF state (pose in odom, twist in base_link)
  /waterlinked_dvl_driver/odom                  only its arrival time, to know the DVL is alive
  /controller/setpoint      geometry_msgs/PoseStamped  hold this pose (odom frame)
  /controller/reference     auv_interfaces/MotionReference  moving reference from
                                                guidance: pose + feedforward velocity
                                                and acceleration, all in odom
Outputs
  /axis_demand              auv_interfaces/AxisDemand  (only while enabled)
  /controller/at_setpoint   std_msgs/Bool
Services
  /controller/enable        std_srvs/SetBool   starts disabled; enabling with no
                                               setpoint holds the current pose
  /controller/hold          std_srvs/Trigger   hold the current pose from now on
"""
import math

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool
from std_srvs.srv import SetBool, Trigger
from auv_interfaces.msg import AxisDemand, MotionReference

from auv_control.position_control import (
    ControllerParams, PositionController, Reference, State,
)


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class PositionControllerNode(Node):
    def __init__(self):
        super().__init__('position_controller')

        defaults = ControllerParams()
        self.declare_parameter('rate', 20.0)
        self.declare_parameter('odom_topic', '/odometry/filtered')
        self.declare_parameter('dvl_topic', '/waterlinked_dvl_driver/odom')
        self.declare_parameter('odom_timeout', 0.5)
        self.declare_parameter('dvl_timeout', 1.0)
        self.declare_parameter('reference_timeout', 0.5)
        self.declare_parameter('world_frame', 'odom')
        for name, value in vars(defaults).items():
            if name in ('yaw_deadband', 'yaw_tolerance'):
                self.declare_parameter(name + '_deg', math.degrees(value))
            else:
                self.declare_parameter(name, value)

        params = ControllerParams(**{
            name: (math.radians(self.get_parameter(name + '_deg').value)
                   if name in ('yaw_deadband', 'yaw_tolerance')
                   else self.get_parameter(name).value)
            for name in vars(defaults)
        })
        self.controller = PositionController(params)
        self.world_frame = self.get_parameter('world_frame').value
        self.odom_timeout = self.get_parameter('odom_timeout').value
        self.dvl_timeout = self.get_parameter('dvl_timeout').value
        self.reference_timeout = self.get_parameter('reference_timeout').value

        self.enabled = False
        self.state = None
        self.state_time = None
        self.dvl_time = None
        self.reference = None
        self.ff_time = None
        self.last_tick = None

        self.create_subscription(Odometry, self.get_parameter('odom_topic').value, self.odom_cb, 10)
        self.create_subscription(Odometry, self.get_parameter('dvl_topic').value, self.dvl_cb, 10)
        self.create_subscription(PoseStamped, '/controller/setpoint', self.setpoint_cb, 10)
        self.create_subscription(MotionReference, '/controller/reference', self.reference_cb, 10)
        self.demand_pub = self.create_publisher(AxisDemand, '/axis_demand', 10)
        self.at_pub = self.create_publisher(Bool, '/controller/at_setpoint', 10)
        self.create_service(SetBool, '/controller/enable', self.enable_cb)
        self.create_service(Trigger, '/controller/hold', self.hold_cb)
        self.create_timer(1.0 / self.get_parameter('rate').value, self.tick)

        self.get_logger().info('position_controller ready (disabled). '
                               'Call /controller/enable to start.')

    # ---------------------------------------------------------------- inputs
    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def odom_cb(self, msg):
        p, q, v = msg.pose.pose.position, msg.pose.pose.orientation, msg.twist.twist.linear
        self.state = State(p.x, p.y, p.z, yaw_of(q), v.x, v.y)
        self.state_time = self.now()

    def dvl_cb(self, _msg):
        self.dvl_time = self.now()

    def _frame_ok(self, frame_id):
        if frame_id and frame_id != self.world_frame:
            self.get_logger().error(
                f"Setpoint frame '{frame_id}' is not '{self.world_frame}'; ignored.")
            return False
        return True

    def setpoint_cb(self, msg):
        if not self._frame_ok(msg.header.frame_id):
            return
        p, q = msg.pose.position, msg.pose.orientation
        self.reference = Reference(p.x, p.y, p.z, yaw_of(q))
        self.ff_time = None
        self.get_logger().info(
            f'Setpoint: x={p.x:.2f} y={p.y:.2f} z={p.z:.2f} '
            f'yaw={math.degrees(self.reference.yaw):.1f} deg')

    def reference_cb(self, msg):
        if not self._frame_ok(msg.header.frame_id):
            return
        p, q = msg.pose.position, msg.pose.orientation
        v, a = msg.velocity.linear, msg.acceleration.linear
        self.reference = Reference(p.x, p.y, p.z, yaw_of(q), v.x, v.y, a.x, a.y)
        self.ff_time = self.now()

    def _latch_current_pose(self):
        s = self.state
        self.reference = Reference(s.x, s.y, s.z, s.yaw)
        self.ff_time = None

    def enable_cb(self, request, response):
        if request.data:
            if not self._state_fresh():
                response.success = False
                response.message = 'No fresh /odometry/filtered; not enabling.'
                return response
            if self.reference is None:
                self._latch_current_pose()
            self.controller.reset()
            self.last_tick = None
            self.enabled = True
            response.message = 'Controller enabled.'
        else:
            self.enabled = False
            self.reference = None
            self.demand_pub.publish(self._demand_msg(0.0, 0.0, 0.0, 0.0))
            response.message = 'Controller disabled.'
        response.success = True
        self.get_logger().info(response.message)
        return response

    def hold_cb(self, _request, response):
        if self.state is None:
            response.success = False
            response.message = 'No odometry yet.'
            return response
        self._latch_current_pose()
        response.success = True
        response.message = 'Holding current pose.'
        self.get_logger().info(response.message)
        return response

    # ------------------------------------------------------------------ loop
    def _state_fresh(self):
        return self.state_time is not None and self.now() - self.state_time < self.odom_timeout

    def _demand_msg(self, surge, sway, heave, yaw):
        msg = AxisDemand()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.surge, msg.sway, msg.heave, msg.yaw = surge, sway, heave, yaw
        return msg

    def tick(self):
        if not self.enabled:
            return
        now = self.now()
        dt = 0.0 if self.last_tick is None else min(max(now - self.last_tick, 0.0), 0.2)
        self.last_tick = now

        if not self._state_fresh():
            # Centre every stick: ArduSub holds depth and heading on its own.
            self.controller.reset()
            self.demand_pub.publish(self._demand_msg(0.0, 0.0, 0.0, 0.0))
            self.get_logger().warn('EKF odometry stale - all axes centred.',
                                   throttle_duration_sec=2.0)
            return

        ref = self.reference
        if self.ff_time is not None and now - self.ff_time > self.reference_timeout:
            # Guidance went quiet: keep its last pose, drop its feedforward.
            ref.vx_ff = ref.vy_ff = ref.ax_ff = ref.ay_ff = 0.0
            self.ff_time = None

        dvl_ok = self.dvl_time is not None and now - self.dvl_time < self.dvl_timeout
        if not dvl_ok:
            self.get_logger().warn('DVL stale - surge/sway centred.', throttle_duration_sec=2.0)

        d = self.controller.compute(self.state, ref, dt, velocity_ok=dvl_ok)
        self.demand_pub.publish(self._demand_msg(d.surge, d.sway, d.heave, d.yaw))
        self.at_pub.publish(Bool(data=self.controller.at_reference(self.state, ref)))


def main(args=None):
    rclpy.init(args=args)
    node = PositionControllerNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
