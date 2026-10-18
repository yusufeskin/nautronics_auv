# Autonomy stack architecture

The reference for how this vehicle's software is split up and why. Read this
before adding a node: it says which layer your code belongs in and what it is
allowed to talk to.

New to the project? Read [ONBOARDING.md](ONBOARDING.md) first — it explains
the underlying problem and walks through the code. This document assumes you
already know that and want the map.

## Principles

**Layers talk through contracts, not through each other.** Every boundary
below is a message type on a topic. A layer may be rewritten, replaced with an
off-the-shelf package, or swapped between simulation and hardware without any
other layer noticing. This is what lets two people work on estimation and
control in the same week, and what lets perception be added after everything
else already works.

**Standard messages wherever one exists.** Pose, twist, odometry, IMU, range
and detections stay on `geometry_msgs` / `nav_msgs` / `sensor_msgs` /
`vision_msgs` so that rviz, `robot_localization`, rosbag tooling and Nav2 keep
working. `auv_core_msgs` holds only the interfaces with no standard
equivalent.

**One conversion point per convention.** Frame conversions live in
`auv_core/frames.py` and nowhere else. A sign flip written anywhere else is a
bug, not a shortcut.

**Nothing reaches the estimator ungated.** Every sensor passes through a
Layer 1 gate that validates it and reports its health. Both of the RoboSub
teams whose 2025/2026 technical design reports informed this design lost pool
time to sensors that degraded silently.

**One estimate of where the vehicle is.** The companion computer's estimator
is the only thing that knows the vehicle's position. The autopilot is not fed
the DVL and is never asked to hold a position; two estimators fed the same
data drift apart, and then two parts of the system disagree about where "here"
is.

## Layers

| # | Layer | Package | Status |
|---|-------|---------|--------|
| 0 | Frames, geometry, contracts | `auv_core`, `auv_core_msgs` | built |
| 1 | Sensor gates | `auv_sensors` | built |
| 2 | State estimation (`odom`) | `auv_estimation` | built, untuned |
| P | Perception and object map | — | later; contract fixed below |
| 3 | Guidance: goal pose → reference trajectory | — | not started |
| 4 | Control, and the boundary to ArduSub | `auv_actuation` | built for ALT_HOLD, untuned |
| 5 | Mission (behaviour tree) | — | not started |
| — | Simulation harness | `auv_description` | inherited |

**Perception is added last, and nothing waits for it.** Guidance takes a goal
pose — "be here, facing this way" — and does not care where it came from.
Until perception exists an operator or the behaviour tree supplies that goal
by hand; afterwards the object map supplies it. Neither guidance nor control
changes when that happens.

There is no global (`map`-frame) correction: this vehicle has no source of
absolute position, so there is nothing to correct against. See
[Frame conventions](#frame-conventions).

## Contracts

| Boundary | Message | Frame |
|---|---|---|
| sensor → gate | driver-native | sensor |
| gate → estimator | `sensor_msgs/Imu`, `nav_msgs/Odometry` (twist only), `geometry_msgs/PoseWithCovarianceStamped`, `sensor_msgs/Range` | sensor |
| gate → operators | `auv_core_msgs/SensorHealth` | — |
| estimator → guidance, control | `nav_msgs/Odometry` on `/odom/filtered` | `odom` |
| estimator → mission | `auv_core_msgs/NavStatus` | — |
| perception → object map | `vision_msgs/Detection3DArray`, stamped with the **image** time | camera optical |
| object map → mission | tracked objects: class, track id, pose with covariance, time last seen | `odom` |
| mission → guidance | `geometry_msgs/PoseStamped` goal | `odom` |
| guidance → control | `nav_msgs/Odometry` reference: pose to be at now, velocity feedforward | `odom` / `base_link` |
| control → output stage | `auv_core_msgs/AxisDemand` *(new)*: surge, sway, climb rate, yaw rate | `base_link` (FLU) |
| output stage → autopilot bridge | `auv_core_msgs/PilotDemand` *(redefined)*: six RC channels in PWM | — |

Fixing these now is what lets the layers be built in any order. In
particular the `mission → guidance` row is the socket perception eventually
drives: a goal pose in `odom` is the same message whether a person typed it or
a camera found it.

The last two rows replace the earlier `WrenchStamped` boundary. ArduSub in
ALT_HOLD does not accept a force on every axis — its vertical and yaw sticks
are rate demands — so a wrench contract claimed something the vehicle cannot
deliver. See [Control](#control-ardusub-in-alt_hold-driven-like-a-joystick).

## Frame conventions

Everything on a ROS topic is REP-103: **ENU** world, **FLU** body. The
autopilot and the DVL speak NED/FRD, and those are converted at exactly two
places — the DVL gate on the way in, and the autopilot bridge in both
directions. Both call `auv_core/frames.py`.

```
odom ─(estimator, Layer 2)─▶ base_link
 │                               │
 │   (static, from config/vehicle.yaml) ─┬─▶ imu_link
 │                                       ├─▶ heading_link
 │                                       ├─▶ dvl_link
 │                                       ├─▶ depth_link
 │                                       └─▶ *_camera_link ─▶ *_camera_optical
 │
 └─(object map, Layer P)─▶ object_<class>_<id> ─(fixed offset)─▶ approach_<class>_<id>
```

`base_link → sensor` transforms come from `auv_core/config/vehicle.yaml` and
are published by the `static_frames` node. That file is the single source of
truth for vehicle geometry: the static TFs and the simulator's URDF both have
to agree with it.

Three rules that are easy to get wrong:

- **Camera optical frames are separate frames.** OpenCV and every PnP solver
  use z-forward, x-right, y-down; a ROS `camera_link` is x-forward, z-up.
  `*_camera_optical` is the fixed rotation between them. A pose from PnP is in
  the optical frame; reading it as `camera_link` puts the object 90° off.
- **Objects hang off `odom`, never off a camera or `base_link`.** An object
  attached to the vehicle moves when the vehicle moves.
- **Detections are transformed at the image timestamp.** The camera pose is
  looked up in TF at the moment the frame was captured, not when the detector
  finished. Otherwise a vehicle that is turning writes every object down in
  the wrong place.

`approach_*` is the pose the vehicle should stop at for that object — for
example 1.5 m in front of it, facing it. That offset belongs to the task, not
to the detector, and it is what the mission hands guidance as a goal.

**There is no `map` frame, deliberately.** REP-105 reserves one for a global,
drift-free world frame, but this vehicle has no source of absolute position:
no GPS underwater, no surveyed acoustic beacons, no prior map. A `map` frame
would be an alias for `odom` pretending to be something better, and code
would start assuming a pose means the same thing at minute 15 as at minute 0.

So `odom` is the only world frame. Its origin is wherever the vehicle was when
the estimator started, and it drifts in yaw and horizontal position. That
drift is slow — over the few metres of an approach it is centimetres — and
every object is re-observed before the vehicle commits to it, so an object's
pose in `odom` is good for as long as the mission needs it.

## How a task runs

The mission is a loop over task objects. The camera is fixed to the hull, so
the vehicle moves to look; there is no continuous visual servoing on the way
in.

```
SEARCH ─▶ DETECT ─▶ ESTIMATE ─▶ APPROACH ─▶ RE-OBSERVE ─▶ FINAL APPROACH ─▶ HOLD ─▶ EXECUTE ─▶ next object
  │          │          │           │             │              │             │
 sweep     detector   PnP, filter  to a staging   stop, look,    to approach_*  our own
 heading              over frames  point ~2 m     update pose                   position hold
                      in odom      out
```

**Look, move, look.** The final position error is the PnP error at the moment
the pose was last measured plus the odometry drift since. PnP error grows
quickly with range; DVL drift over a couple of metres is small. So the
vehicle approaches in two legs and measures again at the staging point, where
the object is close enough to be measured well.

**Fine positioning is open.** Whether the last few centimetres come from a
repeated look–move cycle, a short closed loop on the image, or something
else is undecided. Any of them drives control through the same `guidance →
control` reference, so the choice does not move any boundary.

**Planar targets have two PnP solutions.** A flat four-corner target seen
from a distance is ambiguous between leaning left and leaning right. The
legacy detector already asks for both (`solvePnPGeneric`); choosing between
them across frames is the object map's job, and the goal is not issued until
it has.

## What exists today

```
                    ┌──────────────────────────────────────┐
/waterlinked/... ──▶│ dvl_gate    │ FRD→FLU, beam/speed/    │──▶ /dvl_gate/velocity   (Odometry, twist)
   or /sim/dvl/raw  │             │ accel/lock checks       │──▶ /dvl_gate/altitude   (Range)
                    ├─────────────┼─────────────────────────┤──▶ /dvl_gate/health
/imu/raw or /imu0 ─▶│ imu_gate    │ range checks, gyro bias │──▶ /imu_gate/data       (Imu, no orientation)
                    │             │ estimation, mag dropped │──▶ /imu_gate/gyro_bias  (Vector3Stamped)
                    ├─────────────┼─────────────────────────┤──▶ /imu_gate/health
/pressure/raw ─────▶│ depth_gate  │ surface zero, rate and  │──▶ /depth_gate/pose     (PoseWithCovarianceStamped)
                    │             │ plausibility checks     │──▶ /depth_gate/depth    (Float64)
                    ├─────────────┼─────────────────────────┤──▶ /depth_gate/health
/autopilot/attitude▶│heading_gate │ gyro cross-check, tilt  │──▶ /heading_gate/pose   (PoseWithCovarianceStamped)
  + /imu_gate/data  │  (optional) │ limit                   │──▶ /heading_gate/yaw_deg (Float64)
                    └─────────────┴─────────────────────────┘──▶ /heading_gate/health
```

`depth_gate` and `heading_gate` are the two **absolute** sensors: everything
else measures a rate, and rates have to be integrated. Depth is always
available. Heading currently comes from ArduSub's own attitude estimate
(see [Known open items](#known-open-items)), so `heading_gate` is off unless
launched with `use_heading:=true`, which in turn needs `autopilot:=true`.

Layer 2 sits on top of those:

```
/imu_gate/data ──▶ imu_filter_madgwick ──▶ /imu/attitude ──┐
                      (use_mag: false)                     ▼
/dvl_gate/velocity ──────────────────────────────▶ ekf_filter_node ──▶ /odom/filtered
/depth_gate/pose  ───────────────────────────────▶ (robot_localization)  + TF odom→base_link
/heading_gate/pose ──────────────────────────────▶

all */health, /odom/filtered ──────────────────▶ nav_status ──▶ /nav_status
```

`robot_localization` and `imu_filter_madgwick` come from apt via rosdep; the
only code in `auv_estimation` is `nav_status`. The fusion design — z from
pressure, roll and pitch from gravity, velocity from the DVL, yaw integrated
from the gyro with an absolute correction when available — lives in
`auv_estimation/config/ekf.yaml`, which is commented at length.

There is one entry point for the whole stack, and the only thing that
changes between the simulator and the vehicle is which sensor sources feed it:

```bash
ros2 launch auv_bringup autonomy.launch.py sim:=true                     # simulator
ros2 launch auv_bringup autonomy.launch.py sim:=true autopilot:=true     # + SITL control
ros2 launch auv_bringup autonomy.launch.py autopilot:=true               # vehicle
```

ArduSub is a separate process this launch does not start — `sim_vehicle.py -v
ArduSub -f JSON --model JSON` in simulation, already running on the Pixhawk for
real. Without it the stack still runs: sensors flow and the estimator produces
a pose, the vehicle just cannot move and has no absolute heading.

Everything from Layer 0 upward is identical in both cases — same code, same
thresholds, same message flow. Values that genuinely differ live in small
overlay files (`sensors_sim.yaml`, `pilot_demand_sim.yaml`) loaded on top of
the real configuration, so a threshold is changed in one place and simulation
cannot quietly drift away from the vehicle.

The pre-run checklist and the debugging walkthrough are in
[ONBOARDING.md](ONBOARDING.md). The first end-to-end test of control in
simulation, step by step, is [TESTING.md](TESTING.md).

## Control: ArduSub in ALT_HOLD, driven like a joystick

ArduSub's ALT_HOLD runs at 400 Hz on real-time hardware, with arming logic,
failsafes and motor mixing that already work on this vehicle. This stack does
not replace any of that. It drives ArduSub exactly the way a pilot with a
joystick would: one PWM value per axis, sent as RC channel overrides.

So the split is:

| ArduSub owns | This stack owns |
|---|---|
| motor mixing — no thrust allocation is written here | where the vehicle should be: x, y, depth, yaw in `odom` |
| roll and pitch held level | the path there (guidance) |
| depth held whenever the throttle stick is centred | position hold at the end of it |
| heading held whenever the yaw stick is centred | the depth and heading targets themselves |

### What each channel means in ALT_HOLD

This table is the reason the controller looks the way it does. The channels
are not all forces:

| Ch | Axis | In ALT_HOLD it is | At 1500 (centred) | The controller sends |
|---|---|---|---|---|
| 5 | forward | a thrust fraction | no thrust | velocity PID on the DVL |
| 6 | lateral | a thrust fraction | no thrust | velocity PID on the DVL |
| 3 | throttle | a **climb rate** | hold this depth | depth error × gain, capped. **No PID** — ArduSub closes the rate loop |
| 4 | yaw | a **turn rate** | hold this heading | yaw error × gain, capped. **No PID** — ArduSub closes the rate loop |
| 1, 2 | pitch, roll | a lean **angle** | level | always 1500 |

Running a PID on the throttle or yaw channel would close the same rate loop
twice — once here, once inside ArduSub — and tune neither. On those two axes
the controller is a proportional position loop that hands ArduSub a rate and
lets it do the rest; when the error reaches zero the stick centres and ArduSub
holds what it has.

Only forward and lateral need calibrating from thrust fraction to effect. The
climb-rate and turn-rate scales are ArduSub's own (`PILOT_SPEED_UP`,
`PILOT_SPEED_DN`, and its yaw-rate scaling); read them off the vehicle's
parameters and verify by measurement rather than trusting either alone.

### Position hold is ours

There is no separate hold mode. Holding a pose is the position controller
with a target that does not move: on hold, the current pose from
`/odom/filtered` is latched as the target and the same loops keep the vehicle
on it. Guidance reaching the end of a trajectory leaves the vehicle in exactly
that state. This is what fine positioning and task execution stand on.

ArduSub's POS_HOLD is **not** used. It holds against its own EKF3 position,
which needs the DVL fed into the autopilot — the second estimator this
architecture refuses — and even then it would fight our position loop every
time the stick is centred.

### When things go wrong

- **The companion stops sending.** ArduSub's RC override timeout
  (`RC_OVERRIDE_TIME`) expires and every channel returns to neutral. In
  ALT_HOLD neutral means hold depth, hold heading, stop pushing horizontally —
  the vehicle stays roughly where it was. This parameter must be set
  deliberately, not left at its default.
- **The DVL drops out.** Short gaps are coasted on the estimator's
  prediction. Past the limit `nav_status` reports, the controller centres the
  forward and lateral channels and keeps depth and heading: the vehicle
  drifts slowly instead of driving on a stale velocity.
- **The output stage starts gated off** and has to be enabled deliberately.
  A node that drives thrusters the moment it is launched will do it in the
  lab.

### Permanent consequences of this choice

- **No roll or pitch torque.** Those sticks take a lean angle inside
  ArduSub's limit. RoboSub's 90° gate style points need a temporary switch to
  MANUAL.
- **Two parties hold heading.** Between commands ArduSub holds its own EKF3
  heading; our outer loop closes on ours. Today both come from the same
  source, so they agree. If a better heading source is fitted, ArduSub's slow
  hold drift simply becomes a disturbance our loop corrects.

If either becomes intolerable, the escapes are MANUAL mode with this stack
stabilising attitude, or driving the thrusters directly with our own
allocation. Both change the `control → output stage` contract to a full
wrench; `vehicle.yaml` already carries the rank-6 configuration matrix that
allocation would need.

### One process owns the MAVLink link

That process is `auv_hardware/pixhawk_bridge2`. Nothing else opens a
connection: a second process cannot bind the same endpoint, and two
independent sources of command authority over one vehicle is a worse problem
than a port conflict. The output stage publishes `PilotDemand`, and the bridge
forwards it as `RC_CHANNELS_OVERRIDE`. Exactly one kind of pilot input is ever
sent — RC override, not `MANUAL_CONTROL` alongside it — because the autopilot
acts on whichever arrived last.

The bridge also carries command paths left over from the previous
architecture (`set_depth`, `set_attitude`, `set_position*`, `set_speed`), and
the DVL forwarding to ArduSub as `VISION_POSITION_DELTA`. Neither belongs to
this design. They are switched by the bridge's `command_mode` parameter:
`legacy` (the default, so the old mission launches keep working) or
`rc_override`, which `autonomy.launch.py` sets and which leaves only
`/pilot_demand`. They are deleted when the legacy stack is retired.

## Tests

```bash
colcon test --packages-select auv_core auv_sensors auv_actuation
colcon test-result --verbose
```

The logic in every gate and in the controller is a plain Python class with no
ROS dependency, so it is tested directly rather than through a running graph.
The tests are the specification of what each component does.

`test_vehicle_config.py` also asserts properties of the physical vehicle:
that thruster geometry has rank 6 (fully actuated), that the vehicle is
slightly positively buoyant, and that the centre of buoyancy sits above the
centre of mass. If one of those fails, the fix is on the vehicle or in
`vehicle.yaml`, not in the code that reads it.

## Remaining build order

Each step is testable on its own, in simulation first.

1. ~~Frames, contracts, sim harness~~ — done
2. ~~Sensor gates~~ — done
3. ~~Estimator~~ — built as `auv_estimation`, but **untuned**: the process
   noise is `robot_localization`'s default and nobody has compared the output
   against a known trajectory. The simulator's ground truth is the reference.
4. ~~Control for ALT_HOLD~~ — built in `auv_actuation`: velocity PID on
   forward and lateral, proportional rate demands on depth and yaw, our own
   position hold, `PilotDemand` as six RC PWM channels, and the bridge
   sending `RC_CHANNELS_OVERRIDE` in `rc_override` mode. **Untuned, and not
   yet run against SITL.**
5. **Guidance** — goal pose in, reference trajectory out. The vehicle is
   fully actuated and the pool is open water, so this is trajectory
   generation rather than path search: a straight line with a trapezoidal
   speed profile and yaw interpolated along it, ending in hold. Obstacles, if
   they ever matter, slot in behind the same contract.
6. **Calibration** — forward and lateral thrust fraction against the speed
   it produces (terminal velocity off the DVL, one steady run per axis), and
   ArduSub's climb- and turn-rate scales checked against measurement. In
   simulation first, then half an hour of pool time per axis.
7. **Estimator tuning** against simulator ground truth.
8. **Mission behaviour tree**, driving guidance with hand-given goals.
9. **Perception and object map** — detector, PnP, tracking in `odom`, and
   the `approach_*` goals the mission already knows how to use.
10. **Fine positioning** — method still to be chosen.

## Known open items

**Yaw is the only unbounded error, and its source is a stopgap.** Depth is
measured absolutely, roll and pitch are pinned by gravity, velocity comes
straight from the DVL. Yaw comes from integrating the gyro, and horizontal
position error then grows as roughly `distance travelled × heading error`.

There is no dedicated heading sensor yet, so `heading_gate` takes ArduSub's
own attitude (`ATTITUDE`, republished as `/autopilot/attitude`). That is
EKF3 output and it is **compass-aided**: in a steel-reinforced pool it can
carry a magnetic bias. Two limits follow:

- `heading_gate` cross-checks the heading's *rate* against the external
  gyro. That catches a heading that jumps; it cannot catch one that is
  smoothly wrong, and an EKF output is smooth by construction. Treat this
  source as approximate.
- `max_yaw_rate_mismatch_deg_s` in `config/sensors.yaml` still has to be set
  from a pool swing test.

The object-relative mission softens this a great deal. Every final approach
is measured against the object itself, so heading error only matters during
search and during transit between objects. The proper fix remains hardware —
an AHRS with good disturbance rejection, or a gyro stable enough that drift
over one run is tolerable — and `heading_gate` is the socket it plugs into.

**The hydrodynamic model is placeholders.** With ArduSub closing the depth
and yaw rate loops, it matters even less than before: only forward and
lateral are open-loop thrust, and those loops close on measured DVL velocity.
Do not let anyone spend a month here.

**The previous architecture is still in the tree.** `auv_control`, `auv_bt`,
`auv_navigation`, the legacy handlers in `auv_hardware` and the old
`pixhawk_bridge` are not part of this design. Some of them open their own
MAVLink connections. They are not launched by `autonomy.launch.py` and must
not be run alongside it. Worth carrying over when Layer P is built: the
keypoint detector in `auv_vision` and the object registry in
`auv_vision/object_config.py`. Its `DetectedObject` message is not — it
carries a distance and a yaw angle with no frame and no covariance, where
guidance needs a full pose.

## Where the design decisions came from

- **No magnetometer in the companion's own attitude filter.** NTU Mecatron's
  RoboSub 2026 report describes IMU data "jumping" underwater until they moved
  to a relative heading mode and recalibrated the IMU away from magnetic
  influence. NUS Bumblebee mount a dedicated fibre-optic gyro alongside a
  VectorNav VN-100 rather than trust a magnetic heading at all. The interim
  use of ArduSub's compass-aided heading is a known compromise, not an
  exception to this — see [Known open items](#known-open-items).
- **Estimation on the companion, not in the autopilot EKF.** Mecatron traced
  unstable depth control to ArduPilot's EKF3 fusing DVL vertical velocity
  despite that fusion being disabled in the parameters, and had to patch
  firmware. ALT_HOLD holds depth on EKF3's estimate, so the autopilot is given
  no DVL at all and that whole class of problem cannot occur.
- **Dead-reckoning through DVL dropouts.** Bumblebee report beam reflections
  and dropouts during and after high-rate manoeuvres and near pool walls and
  floors, bridged with dead-reckoning fallbacks. The lock hysteresis in
  `dvl_validation.py` is the first half of that; the coasting logic belongs to
  the estimator.
- **Quaternion attitude deltas, never Euler.** RoboSub 2026 awards style
  points for 90-degree roll and pitch through the gate, which is exactly where
  Euler differencing becomes discontinuous.
- **Keep the autopilot's inner loop.** Bumblebee replaced it with their own
  on the companion computer — with 54 people and 300 hours of in-water
  testing. Replacing something that works is the wrong place to spend a
  one-year budget.
