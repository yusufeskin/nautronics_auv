# auv_actuation

Layer 4 of the autonomy stack: the boundary between our control loop and
ArduSub.

New to the project? Start with [docs/ONBOARDING.md](../docs/ONBOARDING.md).

## The split

ArduSub keeps the **inner loop** — attitude stabilisation and depth hold,
running at 400 Hz on real-time hardware, with arming logic and failsafes that
already work. This stack owns the **outer loop** — where the vehicle is, where
it should be, and how fast it should get there.

```
Layer 2   /odom/filtered ──┐
                           ▼
Layer 4   pose_controller ──▶ WrenchStamped ──▶ ardusub_bridge ──▶ MANUAL_CONTROL
             ▲       ▲                                                    │
   target_pose    cmd_vel (Layer 3, later)      STABILIZE / ALT_HOLD ◀─────┘
                                                  └─▶ mixer ─▶ PWM
```

`pose_controller` runs two nested loops. The outer one turns a position or
heading error into a **sensible speed to approach at**, saturated so the
vehicle never lunges. The inner one tracks that speed against what the DVL
actually measures and produces a wrench. Having a DVL is what makes the inner
loop possible, and it is where the integral lives — buoyancy trim, a standing
current, one thruster pushing harder.

Turning to a heading is not a special case: heading error produces a yaw rate,
the rate is tracked, and when the error reaches zero the command stops. The
autopilot underneath then holds whatever attitude it is at.

The contract into this package is `WrenchStamped` — forces and torques, not
stick counts. That is deliberate: the controller stays in physical units so a
model-based feedforward term can be added later, and if ArduSub's mixer is
ever replaced by our own thrust allocation, only this node is swapped and
nothing above it moves.

## Two things this architecture cannot do

**Roll and pitch torque.** In STABILIZE and ALT_HOLD those axes are ArduSub's:
the sticks command a lean angle inside a configured limit, not a torque. The
mapper reports such a request as undelivered rather than dropping it silently,
so the controller above can stop integrating against a wrench it will never
get. RoboSub's 90° gate style points need a temporary switch to ACRO or
MANUAL.

**Absolute heading, sent to ArduSub.** Nothing here ever commands a heading —
only a yaw **rate**. That is not caution, it is the design: ArduSub's job on
this axis is to *hold still when left alone*, and its own idea of which way
north is never gets read by anything in this stack. So the number in the
autopilot does not have to agree with the number in the estimator, and there
is no "two yaws" problem to reconcile.

The loop is the ordinary cascade: our estimate says the heading is off by 15°,
we command a proportional yaw rate, the vehicle turns, our estimate says the
error is zero, we stop asking. If the autopilot's hold then wanders a degree,
our estimate sees it and a small correction follows. Pixhawk stabilises; the
external sensors navigate.

One thing worth watching for in the pool: if the autopilot's compass is yanked
by a magnetic disturbance, its heading hold will actively rotate the vehicle to
chase it, and this loop will rotate it back. This loop wins — it commands rate
directly — so the net effect is a slow disturbance it rejects. If it ever
becomes large enough to be annoying, `EK3_SRC1_YAW` makes ArduSub's hold a
pure gyro hold instead. Only worth doing if you actually see it.

## Running

```bash
ros2 launch auv_actuation actuation.launch.py
```

Nothing moves until four deliberate steps, none of which these nodes take on
their own:

```bash
ros2 service call /change_mode  auv_interfaces/srv/SetVehicleMode "{mode_name: 'ALT_HOLD'}"
ros2 service call /arm          std_srvs/srv/SetBool "{data: true}"
ros2 service call /ardusub_bridge/enable  std_srvs/srv/SetBool "{data: true}"
ros2 service call /pose_controller/enable std_srvs/srv/SetBool "{data: true}"
```

A node that can arm the vehicle is a node that will arm it in the lab. Two
independent gates on the path to the thrusters is also deliberate.

Then give it somewhere to be:

```bash
ros2 topic pub --once /pose_controller/target_pose geometry_msgs/msg/PoseStamped \
  "{header: {frame_id: odom}, pose: {position: {x: 2.0, y: 0.0, z: -1.5}}}"
```

Watch `/pose_controller/pose_error` shrink and `/pose_controller/at_target` go
true. A zero quaternion in the target means "hold whatever heading you have";
send a real orientation to command one.

`~/cmd_vel` bypasses the outer loop and commands a body velocity directly.
That is the socket the guidance layer plugs into later, and it takes priority
over a pose target while it is fresh.

## Safety behaviour

| Situation | What happens |
|---|---|
| Not enabled | neutral |
| No wrench received yet | neutral |
| Wrench older than `command_timeout_s` | neutral, warned once |
| Normal | mapped demand, sent at a fixed rate |

Neutral means **hold depth and stop translating**, not cut thrust. On a
positively buoyant vehicle, cutting thrust means surfacing.

Commands are sent continuously even when neutral. A gap in pilot input is
something ArduSub has to interpret; a steady neutral is unambiguous.

## Tuning order, which cannot be skipped

1. **Measure the bridge's full-scale forces.** Until then the controller asks
   for newtons and the vehicle receives stick deflections through an unknown
   conversion, and no gain means anything.
2. **Inner loop alone.** Publish a constant `~/cmd_vel`, watch the DVL. Raise
   `kp` until velocity settles quickly without oscillating, then add just
   enough `ki` to remove the steady offset.
3. **Outer loop.** Send a target a few metres out and watch the approach.
   `kp_position` sets how briskly it arrives; the `max_speed_*` caps set the
   cruise and are the first thing to lower if a manoeuvre looks violent.

## Calibration is not optional

`config/ardusub_bridge.yaml` holds the full-scale force per axis — the
conversion between newtons and stick counts. Until those are measured, every
control gain above this node is meaningless, because the controller is asking
for forces and the vehicle is receiving stick deflections.

The file explains the procedure: one axis at a time, fixed demand, wait for
terminal velocity on the DVL. The same runs fill in the drag coefficients in
`vehicle.yaml`.

## Anti-windup

`~/realised_wrench` publishes what the vehicle will **actually** feel, after
clipping and after the roll/pitch axes are dropped. The controller compares it
against what it asked for and holds its integrators on the difference.
`~/saturated` carries the same information as a flag.

## Tests

```bash
colcon test --packages-select auv_actuation
```

Sign conventions for the mapping, and loop behaviour for the controller. They
earn their place: the first run caught the vertical axis being flipped.
ArduSub's pilot axes are not one consistent frame — lateral and yaw invert
against our FLU convention, the throttle stick does not — so each is spelled
out and tested rather than derived from a frame rule.
