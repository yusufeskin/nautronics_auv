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
Layer 2  /odom/filtered        where we are
Layer 3  TwistStamped          desired body velocity
Layer 4  WrenchStamped ─▶ ardusub_bridge ─▶ MANUAL_CONTROL
                                                  │
ArduSub          STABILIZE / ALT_HOLD inner loop ─┘ ─▶ mixer ─▶ PWM
```

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

**Absolute heading.** ArduSub holds heading with its own EKF3 yaw, which is
magnetometer-aided and is *not* the yaw this stack estimates. So nothing here
ever commands a heading — only a yaw **rate**. The outer loop measures the
result with our own estimate and closes the absolute loop itself, which turns
ArduSub's slow heading-hold drift into an ordinary disturbance rather than a
competing opinion about where north is.

Worth testing on the vehicle: setting `EK3_SRC1_YAW` so ArduSub stops fusing
its compass too. Its heading hold then becomes a pure gyro hold — stable in
the short term, drifting in the long term, and no longer yanked around by the
pool's steel. Check it still arms.

## Running

```bash
ros2 launch auv_actuation actuation.launch.py
```

Nothing moves until three deliberate steps, none of which this node takes on
its own:

```bash
ros2 service call /change_mode  auv_interfaces/srv/SetVehicleMode "{mode_name: 'ALT_HOLD'}"
ros2 service call /arm          std_srvs/srv/SetBool "{data: true}"
ros2 service call /ardusub_bridge/enable std_srvs/srv/SetBool "{data: true}"
```

A bridge that can arm the vehicle is a bridge that will arm it in the lab.

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

Mostly sign conventions, and they earn their place: the first run caught the
vertical axis being flipped. ArduSub's pilot axes are not one consistent
frame — lateral and yaw invert against our FLU convention, the throttle stick
does not — so each is spelled out and tested rather than derived.
