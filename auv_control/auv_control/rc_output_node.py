#!/usr/bin/env python3
"""RC output stage: AxisDemand -> RC channel PWM on pwm_router.

pixhawk_bridge2 turns pwm_router into RC_CHANNELS_OVERRIDE. This node is the
only thing that should publish pwm_router while the new stack runs - do not
start the legacy thruster_mixer alongside it.

  ch1 pitch  ch2 roll  : always 1500 (ALT_HOLD keeps the vehicle level)
  ch3 throttle         : heave  (climb rate)
  ch4 yaw              : yaw    (turn rate)
  ch5 forward          : surge  (thrust fraction)
  ch6 lateral          : sway   (thrust fraction)

Mapping is linear outside ArduSub's stick deadzone:
    demand 0      -> 1500 exactly (stick centred: ArduSub holds depth/heading)
    demand u != 0 -> 1500 + sign * (deadzone + (pwm_range - deadzone) * |u|)
so the smallest non-zero demand already leaves the deadzone. Without this a
P loop stalls with a steady error of deadzone / gain (seen in sim: depth
stuck 10 cm off, yaw 1.5 deg off). No thrust curve: ArduSub already shapes
the stick input; a second nonlinearity here would fight it.

ArduSub ALT_HOLD deadzones (source: ArduSub/radio.cpp, Attitude.cpp,
mode_althold.cpp), all scaled by the pilot gain JS_GAIN_DEFAULT = 0.5:
    throttle (ch3): THR_DZ 100 of 1000 over 800 us -> 80 us * gain = 40 us
    yaw      (ch4): RC4_DZ 40 us                    -> 40 us * gain = 20 us
    forward/lateral (ch5/ch6): norm_input(), no deadzone

Safety
  * Starts disabled. /rc_output/enable must be called on purpose.
  * While enabled it publishes at a fixed rate; a demand older than
    demand_timeout is replaced by neutral (all sticks centred).
  * On disable it sends 0 on every channel, which releases the override and
    hands the vehicle back to the RC/GCS.
"""
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import UInt16MultiArray
from std_srvs.srv import SetBool
from auv_interfaces.msg import AxisDemand

NEUTRAL = 1500
RELEASE = 0
RELEASE_REPEATS = 3


def demand_to_channels(surge, sway, heave, yaw, signs, pwm_range=400, pwm_min=1100, pwm_max=1900,
                       deadzones=None):
    """Return [ch1..ch6].

    signs     dict(surge=, sway=, heave=, yaw=), each +1 or -1
    deadzones dict(surge=, sway=, heave=, yaw=) in microseconds; missing = 0
    """
    deadzones = deadzones or {}

    def pwm(axis, value, sign):
        value = max(-1.0, min(1.0, value))
        if value == 0.0:
            return NEUTRAL
        dz = deadzones.get(axis, 0)
        offset = dz + (pwm_range - dz) * abs(value)
        out = NEUTRAL + sign * offset * (1 if value > 0 else -1)
        return int(round(max(pwm_min, min(pwm_max, out))))

    return [
        NEUTRAL,                              # ch1 pitch
        NEUTRAL,                              # ch2 roll
        pwm('heave', heave, signs['heave']),  # ch3 throttle
        pwm('yaw', yaw, signs['yaw']),        # ch4 yaw
        pwm('surge', surge, signs['surge']),  # ch5 forward
        pwm('sway', sway, signs['sway']),     # ch6 lateral
    ]


class RcOutputNode(Node):
    def __init__(self):
        super().__init__('rc_output')
        self.declare_parameter('rate', 20.0)
        self.declare_parameter('demand_timeout', 0.5)
        self.declare_parameter('pwm_range', 400)
        self.declare_parameter('pwm_min', 1100)
        self.declare_parameter('pwm_max', 1900)
        # ArduSub: ch5 >1500 forward, ch6 >1500 right, ch3 >1500 up,
        # ch4 >1500 clockwise. AxisDemand is FLU (sway + left, yaw + CCW),
        # hence the two minus signs.
        self.declare_parameter('surge_sign', 1)
        self.declare_parameter('sway_sign', -1)
        self.declare_parameter('heave_sign', 1)
        self.declare_parameter('yaw_sign', -1)
        self.declare_parameter('start_enabled', False)
        # ArduSub stick deadzones in microseconds (see module docstring).
        self.declare_parameter('surge_deadzone_us', 0)
        self.declare_parameter('sway_deadzone_us', 0)
        self.declare_parameter('heave_deadzone_us', 40)
        self.declare_parameter('yaw_deadzone_us', 20)

        g = lambda n: self.get_parameter(n).value  # noqa: E731
        axes = ('surge', 'sway', 'heave', 'yaw')
        self.signs = {k: g(k + '_sign') for k in axes}
        self.pwm_args = dict(pwm_range=g('pwm_range'), pwm_min=g('pwm_min'), pwm_max=g('pwm_max'),
                             deadzones={k: g(k + '_deadzone_us') for k in axes})
        self.timeout = g('demand_timeout')
        self.enabled = g('start_enabled')
        self.release_left = 0
        self.demand = None
        self.demand_time = None

        self.pub = self.create_publisher(UInt16MultiArray, 'pwm_router', 10)
        self.create_subscription(AxisDemand, '/axis_demand', self.demand_cb, 10)
        self.create_service(SetBool, '/rc_output/enable', self.enable_cb)
        self.create_timer(1.0 / g('rate'), self.tick)
        self.get_logger().info(
            f"rc_output ready ({'ENABLED' if self.enabled else 'disabled'}). signs={self.signs}")

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def demand_cb(self, msg):
        self.demand = msg
        self.demand_time = self.now()

    def enable_cb(self, request, response):
        self.enabled = request.data
        if not self.enabled:
            self.release_left = RELEASE_REPEATS
        response.success = True
        response.message = 'RC output ' + ('enabled' if self.enabled else 'disabled, override released')
        self.get_logger().info(response.message)
        return response

    def publish(self, channels):
        self.pub.publish(UInt16MultiArray(data=channels))

    def tick(self):
        if not self.enabled:
            if self.release_left > 0:
                self.publish([RELEASE] * 6)
                self.release_left -= 1
            return

        fresh = self.demand_time is not None and self.now() - self.demand_time < self.timeout
        if fresh:
            d = self.demand
            channels = demand_to_channels(d.surge, d.sway, d.heave, d.yaw, self.signs, **self.pwm_args)
        else:
            channels = [NEUTRAL] * 6
        self.publish(channels)


def main(args=None):
    rclpy.init(args=args)
    node = RcOutputNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
