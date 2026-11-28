#!/usr/bin/env python3
"""Measure the ALT_HOLD stick -> steady velocity curve for surge and sway.

For each axis and each PWM offset it holds the stick for hold_time, averages
the EKF body velocity over the last measure_time, then centres the stick for
rest_time. Every offset is driven +, then -, so the vehicle ends near where
it started. The result feeds the controller's feedforward table.

Run with position_controller and rc_output DISABLED (or not running): this
node publishes pwm_router itself. ArduSub must be armed in ALT_HOLD at depth.

  ros2 run auv_control stick_calibration --ros-args -p use_sim_time:=true \
      -p out:=$HOME/bags/stick_calibration.csv

Channels are raw ArduSub: ch5 > 1500 forward, ch6 > 1500 right. Velocities
are reported FLU (sway + left), as the EKF publishes them.
"""
import csv
import math
import os

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import UInt16MultiArray

NEUTRAL = 1500
CHANNEL = {'surge': 4, 'sway': 5}   # index into [ch1..ch6]


class StickCalibration(Node):
    def __init__(self):
        super().__init__('stick_calibration')
        self.declare_parameter('axes', ['surge', 'sway'])
        self.declare_parameter('offsets_us', [20, 40, 60, 80, 120, 160])
        self.declare_parameter('pwm_range', 400)
        self.declare_parameter('hold_time', 7.0)
        self.declare_parameter('measure_time', 3.0)
        self.declare_parameter('rest_time', 4.0)
        self.declare_parameter('odom_topic', '/odometry/filtered')
        self.declare_parameter('out', os.path.expanduser('~/bags/stick_calibration.csv'))
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.pwm_range = g('pwm_range')
        self.hold, self.measure, self.rest = g('hold_time'), g('measure_time'), g('rest_time')
        self.out = g('out')

        # Step list: (axis, signed offset). Rest between every step.
        self.steps = [(a, s * o) for a in g('axes') for o in g('offsets_us') for s in (1, -1)]
        self.i = -1
        self.phase = 'rest'
        self.phase_start = None
        self.samples = []
        self.results = []
        self.odom = None
        self.odom_time = None
        self.done = False

        self.pub = self.create_publisher(UInt16MultiArray, 'pwm_router', 10)
        self.create_subscription(Odometry, g('odom_topic'), self.odom_cb, 10)
        self.timer = self.create_timer(0.05, self.tick)
        total = len(self.steps) * (self.hold + self.rest)
        self.get_logger().info(f'{len(self.steps)} steps, about {total / 60:.1f} min. '
                               'Waiting for odometry...')

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def odom_cb(self, msg):
        self.odom = msg
        self.odom_time = self.now()

    def send(self, axis=None, offset=0):
        ch = [NEUTRAL] * 6
        if axis is not None:
            ch[CHANNEL[axis]] = NEUTRAL + offset
        self.pub.publish(UInt16MultiArray(data=ch))

    def tick(self):
        now = self.now()
        if self.odom_time is None or now - self.odom_time > 0.5:
            self.send()
            if self.phase_start is not None:
                self.get_logger().warn('Odometry stale - sticks centred.', throttle_duration_sec=2.0)
            return
        if self.phase_start is None:
            self.phase_start = now
            self.get_logger().info('Odometry OK, starting.')
        t = now - self.phase_start

        if self.phase == 'rest':
            self.send()
            if t >= self.rest:
                self.i += 1
                if self.i >= len(self.steps):
                    self.finish()
                    return
                self.phase, self.phase_start, self.samples = 'hold', now, []
            return

        axis, offset = self.steps[self.i]
        self.send(axis, offset)
        if t >= self.hold - self.measure:
            v = self.odom.twist.twist.linear
            self.samples.append((v.x, v.y))
        if t >= self.hold:
            n = len(self.samples)
            vx = sum(s[0] for s in self.samples) / n
            vy = sum(s[1] for s in self.samples) / n
            along = vx if axis == 'surge' else vy
            cross = vy if axis == 'surge' else vx
            std = math.sqrt(sum(((s[0] if axis == 'surge' else s[1]) - along) ** 2
                                for s in self.samples) / n)
            u = offset / self.pwm_range
            self.results.append(dict(axis=axis, offset_us=offset, stick=u, v_along=along,
                                     v_std=std, v_cross=cross, n=n))
            self.get_logger().info(f'{axis:5} {offset:+4d} us (stick {u:+.2f}): '
                                   f'v={along:+.3f} m/s (std {std:.3f}, cross {cross:+.3f})')
            self.phase, self.phase_start = 'rest', now

    def finish(self):
        self.send()
        self.timer.cancel()
        os.makedirs(os.path.dirname(self.out) or '.', exist_ok=True)
        with open(self.out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(self.results[0]))
            w.writeheader()
            w.writerows(self.results)
        lines = ['axis   us    stick   v[m/s]  m/s per stick']
        for r in self.results:
            gain = r['v_along'] / r['stick'] if r['stick'] else float('nan')
            lines.append(f"{r['axis']:5} {r['offset_us']:+4d}  {r['stick']:+.2f}  "
                         f"{r['v_along']:+.3f}  {gain:6.2f}")
        self.get_logger().info('Done. Saved ' + self.out + '\n' + '\n'.join(lines))
        self.done = True


def main(args=None):
    rclpy.init(args=args)
    node = StickCalibration()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if rclpy.ok():
            node.pub.publish(UInt16MultiArray(data=[0] * 6))   # release override
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
