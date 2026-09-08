# Onboarding: the navigation stack from zero

You have just joined the software team and someone pointed you at
`auv_core` and `auv_sensors`. This document explains what problem those
packages solve, why they are shaped the way they are, and how to run and
debug them. It assumes you can write Python and have used ROS 2 a little.
It does not assume you know anything about underwater vehicles.

Read it top to bottom once. After that, [ARCHITECTURE.md](ARCHITECTURE.md)
is the short reference you will actually keep open.

---

## Table of contents

1. [The problem: knowing where you are underwater](#1-the-problem-knowing-where-you-are-underwater)
2. [The sensors, and what each one really tells you](#2-the-sensors-and-what-each-one-really-tells-you)
3. [Drift: the one idea everything else follows from](#3-drift-the-one-idea-everything-else-follows-from)
4. [Frames: the thing that breaks projects](#4-frames-the-thing-that-breaks-projects)
5. [Covariance: every number needs an error bar](#5-covariance-every-number-needs-an-error-bar)
6. [Why the code is split into layers](#6-why-the-code-is-split-into-layers)
7. [Code tour, in reading order](#7-code-tour-in-reading-order)
8. [Running it](#8-running-it)
9. [Debugging: reading the health topics](#9-debugging-reading-the-health-topics)
10. [Recipe: adding a new sensor](#10-recipe-adding-a-new-sensor)
11. [Traps that will bite you](#11-traps-that-will-bite-you)
12. [What is not built yet](#12-what-is-not-built-yet)
13. [Glossary](#13-glossary)

---

## 1. The problem: knowing where you are underwater

A robot that has to swim to a specific place needs to answer one question
continuously: **where am I?** On land or in the air you answer it with GPS.
Underwater you cannot: GPS is a radio signal, and radio does not travel
through water. A metre down, there is nothing.

So the vehicle has to work out its position from its own motion. It knows how
fast it is going and which way it is pointing, and it adds those up over time.
That is called **dead reckoning**, and it is the whole game.

The catch is in the word "adds up". Every measurement has a small error. When
you integrate a measurement over time, you integrate its error too. A velocity
that is wrong by 1 cm/s does not sound like much, but over a ten minute run it
puts you 6 metres from where you think you are. That is the difference between
passing through the competition gate and swimming into the wall next to it.

Everything in this stack exists to make that accumulated error as small as
possible, and — just as importantly — to **know how big it is**, so the rest
of the software can behave sensibly when it grows.

---

## 2. The sensors, and what each one really tells you

New people usually assume a sensor tells you where you are. None of ours do.
Each one measures something much narrower, and knowing exactly what is the
difference between using it correctly and fooling yourself.

### DVL (Doppler Velocity Log)

A Water Linked A50. It fires four acoustic beams at the sea floor, listens for
the echo, and measures the Doppler shift. From those four shifts it computes
**how fast the vehicle is moving relative to the ground**, in three axes.

- It gives you **velocity**, not position. Position only comes from
  integrating it, which is where drift enters.
- It only works when it can see the bottom. This is called **bottom lock**.
  Too far from the floor, too close to it, tilted so the beams miss, or with
  the echo bouncing off a wall — no lock, no velocity. This happens constantly
  in a pool.
- It also reports **altitude**: how far the bottom is below you. Free and
  useful, and most teams forget it exists.
- It runs slowly — about 8 samples per second.

### IMU (Inertial Measurement Unit)

Contains a **gyroscope**, which measures rotation rate (how fast you are
turning, in rad/s), and an **accelerometer**, which measures specific force
(gravity plus whatever acceleration you are undergoing).

- The gyro gives **rate of rotation**, not orientation. Orientation comes from
  integrating it — drift again.
- The accelerometer can tell you which way is down, because gravity is always
  there. That pins **roll** and **pitch** absolutely: they do not drift.
- Nothing in the IMU can tell you which way is **north**. Yaw has no absolute
  reference from these two sensors.
- It runs fast — 100 samples per second or more.

Many IMUs also contain a **magnetometer**, a compass. In open air that solves
yaw. In a competition pool it does not: the building is full of steel, the
vehicle is full of thruster magnets and current-carrying wires, and the
reported heading is wrong in a way that changes as you move. We do not use it.
See section 11.

### Depth sensor (pressure)

Measures absolute water pressure. Pressure rises linearly with depth, so
`depth = (pressure - surface_pressure) / (density * gravity)`.

- This is the **only sensor that measures a position directly**. Not a rate,
  not a change — an actual coordinate.
- It does not drift. At all. The reading at minute 15 is as good as at
  minute 0.
- It is cheap and accurate to about a centimetre.

That makes depth the backbone of the whole estimate, and it is why
`depth_gate` is fussy about its zero reference (section 8).

### Absolute heading source (optional, not fitted yet)

A compass, an AHRS, or a gyrocompass — something that reports **which way the
vehicle is pointing**, absolutely, rather than how fast it is turning.

- Like depth, this measures a coordinate rather than a rate, so it does not
  drift. It is the only thing that can stop yaw drifting (section 3).
- Magnetic ones are cheap and are disturbed indoors. See section 11.
- The vehicle does not have one yet. `heading_gate` exists and is wired up,
  but is off by default until the hardware is chosen.

---

## 3. Drift: the one idea everything else follows from

Put the three sensors together and you get this picture:

| Quantity | Where it comes from | Does it drift? |
|---|---|---|
| Depth (z) | pressure, directly | **No.** Absolute measurement. |
| Roll, pitch | accelerometer sees gravity | **No.** Gravity is an absolute reference. |
| Velocity (u, v, w) | DVL, directly | **No.** Measured, not integrated. |
| **Yaw (heading)** | integrating the gyro | **Yes**, unless an absolute heading source is fitted. |
| **Horizontal position (x, y)** | integrating velocity, rotated by yaw | **Yes**, and worse than yaw. |

Read the last two rows carefully, because they drive most of the design
decisions in this repository.

**Yaw drifts** because a gyro measures rate, you integrate it to get angle,
and every gyro has a small constant offset called **bias**. If the bias is
0.5 deg/s and you do not remove it, after 15 minutes you believe you are
pointing 450 degrees away from where you actually are. Even after removing it,
what remains (gyro noise, and slow bias change with temperature) keeps
accumulating.

**Horizontal position drifts worse**, because it inherits the yaw error. If
you swim 20 metres believing you are heading north when you are actually
heading 5 degrees off, you end up 1.7 metres sideways of where you think — and
the DVL velocity was perfectly correct the whole time. The position error is
roughly `distance travelled × heading error`.

So:

> The DVL is not there to fix drift. It is there to make the velocity term
> almost error-free, so that **heading** becomes the dominant error. Once
> heading is dominant, the way to improve navigation is to improve heading,
> not to buy a better DVL.

This is why `imu_gate` spends effort on gyro bias, why `heading_gate` exists,
and why yaw is written down as the top open item in
[ARCHITECTURE.md](ARCHITECTURE.md#known-open-items) rather than buried.

Note the asymmetry this creates. Depth has an absolute sensor, so it is solved
for free. Yaw needs one too, and the plan is to buy it — but until that part
is chosen and its cross-check tuned, yaw drift is what limits how long a
mission can be.

---

## 4. Frames: the thing that breaks projects

A vector like `[1.0, 0.0, 0.0]` means nothing on its own. One metre in which
direction? Measured from where? A **frame** answers that: it is an origin plus
three axes. Every position, velocity and orientation in the system belongs to
exactly one frame, and moving a quantity between frames requires a
transformation.

Getting this wrong is the most expensive mistake in this kind of project,
because the symptoms do not look like frame bugs. They look like tuning
problems. The vehicle drifts left when commanded straight; you spend a week on
control gains; the actual cause is a sign flip three layers away.

### The two conventions

There are two competing standards for naming axes, and this vehicle contains
both.

**ROS uses ENU / FLU** (this is REP-103, the ROS standard):

- World frame **ENU**: x points East, y points North, z points **Up**.
- Body frame **FLU**: x points **F**orward, y points **L**eft, z points **U**p.

**Autopilots and marine sensors use NED / FRD**:

- World frame **NED**: x points North, y points East, z points **Down**.
- Body frame **FRD**: x points **F**orward, y points **R**ight, z points **D**own.

Notice that both use "z" and both use "y", but they mean opposite directions.
Feed an FRD velocity into code expecting FLU and the vehicle will strafe the
wrong way and dive when told to climb — with no error message anywhere.

The DVL reports FRD. ArduPilot speaks NED/FRD. Everything else in ROS is
ENU/FLU.

### The rule in this repository

> **Every ROS topic in this stack is ENU and FLU.** Conversions happen in
> exactly two places: the DVL gate on the way in, and the autopilot bridge on
> the way out. Both call `auv_core/frames.py`.

If you catch yourself writing `-msg.velocity.y` anywhere else, stop. That is
the bug, not the fix.

### The frame tree

ROS tracks the relationships between frames in a structure called **TF**. Ours
looks like this:

```
odom  ─(the estimator, Layer 2)─▶  base_link
                                       │
           (static, from vehicle.yaml) ─┬─▶ imu_link
                                        ├─▶ heading_link
                                        ├─▶ dvl_link
                                        ├─▶ depth_link
                                        └─▶ front_camera_link ...
```

- **`base_link`** is the vehicle itself. Its origin is a fixed reference point
  on the hull. Everything bolted to the vehicle is at a fixed offset from it.
- **`imu_link`, `dvl_link`, ...** are where each sensor physically sits. The
  DVL is 15 cm below `base_link`, so a velocity measured "at the DVL" is not
  quite a velocity "of the vehicle" when the vehicle is rotating. TF handles
  that, *provided* the offsets in `vehicle.yaml` are correct.
- **`odom`** is a fixed frame in the world that the vehicle moves around in.
  Its origin is wherever the vehicle happened to be when the estimator
  started. It is **continuous** — it never jumps — and it **drifts**: after
  ten minutes, "2 metres north in odom" is not quite the place it was at the
  start.

**There is no `map` frame here**, and that is a deliberate decision rather
than something missing. REP-105 reserves `map` for a global, drift-free world
frame, but a frame is only drift-free if something measures absolute position
to correct it — GPS, surveyed acoustic beacons, a prior map. This vehicle has
none of those and will not have any. Declaring a `map` frame anyway would give
you an alias for `odom` wearing a badge that says "trustworthy", and code
would start assuming a pose means the same thing an hour later.

The practical consequence: **every mission has to be relative.** Relative to
where the run started, or relative to something the vehicle can currently see.
Nothing may be expressed as an absolute pool coordinate, because the vehicle
has no way to know one.

The `base_link → sensor` transforms are published by the `static_frames` node,
which reads them from [`auv_core/config/vehicle.yaml`](../auv_core/config/vehicle.yaml).
That file is the single source of truth. **If you move a sensor on the
vehicle, you change that file and nothing else.**

Anything in that file marked `MEASURE` is a number that must come from a tape
measure on the actual vehicle, not from CAD and not from a guess. The DVL
mount rotation in particular: a 45-degree error there turns every straight
transit into a diagonal one, and it will look like a control problem.

---

## 5. Covariance: every number needs an error bar

A measurement without an uncertainty is only half a measurement. If the DVL
says "0.42 m/s", the estimator needs to know whether that means 0.42 ± 0.01 or
0.42 ± 0.30, because it has to decide how much to believe it relative to
everything else it knows.

**Variance** is that uncertainty squared. If a sensor is typically wrong by
0.05 m/s, its variance is 0.05² = 0.0025. **Covariance** is the same idea for
several quantities at once: a 3×3 matrix whose diagonal holds the variance of
each axis, and whose off-diagonal terms say how the axes' errors are related.

Three rules that come up constantly in this code:

**A covariance of zero means "perfectly known".** No sensor is perfectly
known. A driver that publishes an all-zero covariance is really saying "I did
not fill this in", and passing it through will make any filter diverge. Both
`dvl_gate` and `imu_gate` detect this and substitute a configured default —
see `_resolve_covariance` in
[`dvl_validation.py`](../auv_sensors/auv_sensors/dvl_validation.py).

**A covariance of -1 means "not provided".** This is the ROS convention.
`marine_acoustic_msgs/Dvl` uses it, and `sensor_msgs/Imu` uses a leading -1 in
`orientation_covariance` to mean "this message carries no orientation" — which
is exactly what `imu_gate` sets when it drops the magnetometer-aided
orientation.

**If you rotate a vector, you must rotate its covariance too.** Converting a
velocity from FRD to FLU flips two axes; the uncertainty ellipsoid has to flip
with it. The maths is `R · C · Rᵀ`, and it lives in `frames.rotate_covariance`.
Rotating the mean and forgetting the covariance is a silent bug that quietly
feeds the filter a wrong error model.

---

## 6. Why the code is split into layers

The stack is a stack of layers, each talking to its neighbours only through a
fixed message type on a fixed topic. That fixed message type is called a
**contract**.

```
Layer 0   frames, geometry, contracts          auv_core, auv_core_msgs   ← built
Layer 1   sensor gates                         auv_sensors               ← built
Layer 2   state estimation                     auv_estimation            ← built, untuned
Layer 3   guidance and trajectory              —                         ← next
Layer 4   control + boundary to ArduSub        auv_actuation             ← built, untuned
Layer 5   mission behaviour tree               —
```

Three reasons this matters more than it might look:

**You can replace a layer without touching the others.** Layer 2 will start as
an off-the-shelf `robot_localization` filter. If it is not good enough, it can
be replaced with a custom one later. Because the contract is
`nav_msgs/Odometry` on `/odom/filtered`, nothing above it needs to change.

**Two people can work at once.** The person writing thrust allocation and the
person writing the estimator never have to talk, because the boundary between
their work is already written down. NTU Mecatron's 2026 report gives exactly
this reason for standardising their interfaces.

**Simulation and hardware become interchangeable.** The gates in Layer 1 take
raw sensor topics as launch arguments. Point them at Gazebo topics and you are
in simulation; point them at driver topics and you are on the vehicle.
Everything above Layer 1 is identical and cannot tell the difference. That is
why a bug reproducible in simulation can be fixed without booking pool time —
which is how Mecatron found an ArduPilot EKF bug without burning a test day.

### What a "gate" is, specifically

A gate is a node that sits between one sensor driver and the estimator. Its
entire job is:

1. **Validate.** Is this sample physically possible? Is it fresh? Did the
   sensor actually have a lock?
2. **Convert.** Put it in this stack's conventions — FLU, correct frame id,
   sensible covariance.
3. **Report health.** Publish, at a fixed rate, whether this sensor is usable
   right now and why not if it isn't.

And, just as importantly, what a gate must **never** do: integrate anything,
maintain a pose, fuse two sensors together, or send a command to the
autopilot. The moment a gate does any of those, it stops being testable in
isolation and the layer boundary is gone.

---

## 7. Code tour, in reading order

Read these files in this order. Together they are about 1,500 lines and they
will make sense as a sequence.

### `auv_core/auv_core/frames.py`

Start here. It is the vocabulary the rest of the code speaks. Every frame and
orientation conversion in the project is one of these functions.

The one function worth understanding properly is `quat_delta_to_rotvec`. It
answers "how much did the vehicle rotate between these two attitudes?" without
using Euler angles. Euler angles (roll, pitch, yaw) are convenient for humans
but they break down when pitch approaches ±90°: at that point roll and yaw
become the same axis and the numbers jump wildly. That is not a theoretical
concern here — RoboSub 2026 awards style points for rolling and pitching 90°
through the gate, so the vehicle deliberately passes through the singularity.
The function uses quaternions, which have no such point.

There is a test for exactly this in
[`test_frames.py`](../auv_core/test/test_frames.py):
`test_quat_delta_stays_finite_through_pitch_ninety`.

### `auv_core/config/vehicle.yaml`

Not code, but read it early. It is where the vehicle physically is: sensor
positions, thruster positions, mass, buoyancy. Everything else derives from
it.

### `auv_core/auv_core/vehicle_config.py`

Loads that YAML into typed objects, and derives the **configuration matrix**:
a 6×N matrix mapping "how hard each thruster pushes" to "what force and torque
the vehicle feels". Layer 4 will invert it to answer the reverse question.

Its rank tells you how many of the six degrees of freedom the vehicle can
independently command. Ours is 6, which means the vehicle can hold any
attitude while translating in any direction. That is checked in
[`test_vehicle_config.py`](../auv_core/test/test_vehicle_config.py) — a test
that is really about the hardware, not the software.

### `auv_core/auv_core/health.py` and `gate.py`

`health.py` is the bookkeeping every gate shares: how fast is this sensor
arriving, when did we last hear from it, what fraction of samples are we
throwing away. It is plain Python with no ROS in it, which is why it can be
tested exhaustively in [`test_health.py`](../auv_core/test/test_health.py).

`gate.py` is the thin ROS wrapper: a `Node` base class that owns a tracker and
publishes `SensorHealth` on a timer.

One detail in `gate.py` worth internalising — `now_seconds()` uses the **ROS
clock**, not `time.monotonic()`. Under simulation, ROS time runs at whatever
speed Gazebo runs at. Mixing the two clocks inside one node gives you
comparisons between numbers on completely different scales, and staleness
checks that silently never fire. Always use the node clock or a message
header, never both.

### `auv_sensors/auv_sensors/dvl_validation.py`

The heart of Layer 1, and the best single file to read to understand the
design. Every check is a few lines and each names itself in the rejection
reason. Read `DvlValidator.check` from top to bottom.

Two subtleties worth pausing on:

**Lock hysteresis.** Losing bottom lock takes four consecutive bad samples;
regaining it takes two consecutive good ones. Without this, a single reflected
ping would tear velocity aiding away from the estimator and hand it straight
back, repeatedly. Bumblebee's 2026 report describes exactly this class of
dropout near pool walls and after fast manoeuvres.

**The acceleration check compares against the last *accepted* sample, not the
last sample.** If a bad ping gets rejected and then a good one arrives, the
good one is compared against the last known-good velocity. Compare it against
the rejected one instead and you reject the recovery too, turning one bad ping
into a lasting dropout. There is a test named after this:
`test_acceleration_check_compares_against_the_last_accepted_sample`.

### `auv_sensors/auv_sensors/imu_validation.py`

Range checks, plus `GyroBiasEstimator`. That class watches for the vehicle
being genuinely still — low rotation rate *and* specific force close to
gravity — and averages the gyro over that window. The average is the bias.

Note the deliberate asymmetry: motion resets the accumulation in progress, but
never discards a bias that has already converged. Somebody bumping the vehicle
on the deck must not throw away a good calibration.

### `auv_sensors/auv_sensors/heading_validation.py`

The newest and, if the hardware lands, the most consequential gate. An
absolute heading source is the only thing that can stop yaw drifting, so this
is where that hole gets closed.

It does not simply forward headings. Every sample is cross-checked against the
gyro — which cannot be magnetically disturbed — by comparing how fast the
heading source says the vehicle is turning against how fast the gyro says it
is. When they disagree beyond a threshold, the source is being disturbed and
the gate refuses it.

Two design points worth understanding:

**Why refuse rather than inflate the covariance.** A disturbed compass is not
noisier, it is *biased*: it reports a consistent wrong answer. Handing a
filter a biased measurement with a wide error bar still pulls the estimate
toward the wrong value, just more slowly. Refusing is the correct model.

**Why it goes quiet at high tilt.** Past about 60° of roll or pitch an
absolute heading is untrustworthy, and past 90° of pitch yaw is not even well
defined — it degenerates onto the same axis as roll. This vehicle deliberately
rolls 90° through the gate for style points, so it *will* spend time there.
The gate stops publishing rather than forwarding a meaningless number, and
drops its comparison history so the sample after the manoeuvre is not
differenced across it.

There is no declination handling and there should not be. The vehicle has no
map frame and no notion of true north, so what is needed from a heading source
is that it be **stable**, not that it be correct. A constant offset from true
north is harmless. An offset that changes as the vehicle swims past the pool's
rebar is exactly what the cross-check is hunting.

### `auv_sensors/auv_sensors/depth_validation.py`

The simplest file, guarding the most important sensor. The interesting part is
`zero_at`, which refuses an implausible surface reference — including one
taken with the vehicle already underwater. A run started with a bad zero
carries a constant depth offset that nothing downstream can detect, so it is
worth being rude about.

### The gate nodes

[`dvl_gate.py`](../auv_sensors/auv_sensors/dvl_gate.py),
[`imu_gate.py`](../auv_sensors/auv_sensors/imu_gate.py) and
[`depth_gate.py`](../auv_sensors/auv_sensors/depth_gate.py) are thin. They
convert a ROS message into a plain dataclass, call the validator, and convert
the verdict back into ROS messages. That thinness is on purpose: all the logic
worth testing lives in the files above, where testing it does not require a
running graph.

### `auv_sensors/auv_sensors/sim_pressure_source.py`

The Gazebo model has no barometer, but `depth_gate` is load-bearing and must
be exercised by the same code that runs on the vehicle. This node fakes a
pressure sensor from Gazebo's ground-truth position, with realistic noise.
Simulation only — it must never appear in a real-vehicle launch file, which is
why it is a separate executable rather than a flag.

---

## 8. Running it

### Build

From the workspace root (the directory containing `src/`):

```bash
vcs import src < src/nautronics_auv/nautronics.repos   # first time only
rosdep install --from-paths src --ignore-src -r -y
colcon build --packages-select auv_core_msgs auv_core auv_sensors --symlink-install
source install/setup.bash
```

`auv_core_msgs` must build before the other two; colcon works that out from
the `package.xml` dependencies.

### Run the tests

```bash
colcon test --packages-select auv_core auv_sensors
colcon test-result --verbose
```

You can also run them straight from a clone without building, which is faster
while you are iterating:

```bash
cd src/nautronics_auv/auv_core    && python3 -m pytest
cd src/nautronics_auv/auv_sensors && python3 -m pytest
```

(`auv_sensors/conftest.py` puts the sibling `auv_core` on the path for exactly
this case. Tests that need generated messages still need a build.)

### Run against the simulator

Start the simulator the way you normally do, then:

```bash
ros2 launch auv_estimation navigation.launch.py sim:=true   # layers 0+1+2
ros2 launch auv_sensors    sensors_sim.launch.py            # layers 0+1 only
```

That brings up the static frames, all three gates wired to simulator topics,
and the synthetic pressure source. It assumes `sim_dvl_bridge` from
`auv_hardware` is already publishing `/sim/dvl/raw` — that is part of the
simulator bringup, not of this launch file.

Check it is alive:

```bash
ros2 topic hz /dvl_gate/velocity      # should be around 8 Hz
ros2 topic hz /imu_gate/data          # should be around 100 Hz
ros2 topic hz /depth_gate/pose        # should be around 20 Hz
ros2 topic echo /dvl_gate/health
ros2 run tf2_tools view_frames        # writes a PDF of the frame tree

# with Layer 2 running:
ros2 topic hz /odom/filtered          # should be around 30 Hz
ros2 topic echo /nav_status           # mode 1 = AIDED
```

### Run on the vehicle

```bash
ros2 launch auv_estimation navigation.launch.py   # layers 0+1+2
```

or, to bring up the layers separately:

```bash
ros2 launch auv_core       frames.launch.py
ros2 launch auv_sensors    sensors.launch.py
ros2 launch auv_estimation estimation.launch.py
```

Raw topic names default to the real drivers and can be overridden:

```bash
ros2 launch auv_sensors sensors.launch.py \
  dvl_raw_topic:=/waterlinked_dvl_driver/velocity_report \
  imu_raw_topic:=/imu/raw \
  pressure_raw_topic:=/pressure/raw
```

`heading_gate` is off by default because the vehicle has no absolute heading
source yet. Once one is fitted:

```bash
ros2 launch auv_sensors sensors.launch.py \
  use_heading:=true \
  heading_raw_topic:=/ahrs/imu \
  magnetic_field_topic:=/ahrs/mag
```

Before trusting it, do the **swing test**: put the vehicle in the water, turn
it slowly through a full 360° while recording, and compare
`/heading_gate/yaw_deg` against the gyro. That is what sets
`max_yaw_rate_mismatch_deg_s`, and until it is set the gate is either useless
(too tight) or waved through (too loose).

### Pre-run checklist

Do these in order, every single run. They take under a minute and each one
prevents a failure that is invisible afterwards.

**1. Put the vehicle in the water, floating, and take the surface zero.**

```bash
ros2 service call /depth_gate/zero_surface std_srvs/srv/Trigger
```

`depth_gate` refuses to publish depth until this succeeds. If you skip it,
every depth for the whole run is offset by a constant that no amount of
downstream cleverness can detect. If the service returns failure saying the
reading is not plausible, the vehicle is probably already submerged — float
it and try again.

**2. Hold the vehicle still for about 30 seconds.**

The gyro bias estimator needs an uninterrupted quiet window (20 s by default)
to converge. Watch for it:

```bash
ros2 topic echo /imu_gate/health
```

While it is uncalibrated, `status` is `1` (DEGRADED) and `reason` says so.
Once it converges the log prints the bias in deg/s and status goes to `0`.
If the numbers are much larger than a few tenths of a deg/s, something is
wrong with the mounting or the sensor.

**3. Confirm all three sensors report OK.**

```bash
ros2 topic echo /dvl_gate/health
ros2 topic echo /imu_gate/health
ros2 topic echo /depth_gate/health
```

The DVL will be unhappy until the vehicle is deep enough for the beams to
reach the floor. That is expected on the surface.

**4. Check the frame tree exists.**

```bash
ros2 run tf2_ros tf2_echo base_link dvl_link
```

If this errors, `static_frames` is not running and nothing downstream will
work.

---

## 9. Debugging: reading the health topics

Every gate publishes `auv_core_msgs/SensorHealth` at 2 Hz on `<node>/health`,
whether or not samples are arriving. That last part is the point: a sensor
that has gone silent and a node that has crashed look completely different.

| `status` | Name | Fusable? | What it means |
|---|---|---|---|
| 0 | OK | yes | Everything passing. |
| 1 | DEGRADED | **yes** | Usable but lower quality. Read `reason`. |
| 2 | REJECTED | no | The most recent sample failed a check. |
| 3 | STALE | no | Nothing has arrived within the timeout. |
| 4 | NO_DATA | no | Nothing has *ever* arrived. |

Other fields worth watching:

- `reason` — a sentence saying what is wrong. Always read this first.
- `measured_rate_hz` vs `expected_rate_hz` — a sensor delivering half its
  expected rate is a problem even if every sample passes.
- `reject_ratio` — fraction of samples thrown away in the trailing 5 seconds.
  Deliberately windowed, not lifetime, so a sensor that failed early and
  recovered reads healthy now.
- `accepted_count` / `rejected_count` — lifetime totals, for post-run analysis.

Gates log only on **transitions**, never per sample. A gate that logged every
rejection would bury the one line that mattered under thousands of duplicates.

### Symptom → cause

| What you see | Likely cause |
|---|---|
| `NO_DATA` forever | Wrong topic remap. `ros2 topic list` and check the driver is publishing. |
| `STALE` intermittently | Sensor rate below expectation, or a busy network/USB link. |
| DVL: `"timestamp N s ahead of local clock (check NTP sync...)"` | The DVL and the Jetson disagree about the time. The A50 syncs over NTP; this breaks on a fresh image. Fix the clock, not the threshold. |
| DVL: `"N good beams, need 3"` | No bottom lock. Too shallow, too deep, tilted, or over a bad surface. Normal on the surface. |
| DVL: `"water-track velocity, not bottom lock"` | The DVL fell back to measuring the water instead of the ground. That velocity is relative to the water and is not what you want. |
| DVL: `"implausible acceleration..."` | Usually an acoustic artefact. If it happens constantly, `max_accel_mps2` may be set below what the vehicle actually does. |
| IMU: `"gyro bias not yet estimated"` | Normal until the vehicle has been still long enough. If it never clears, the vehicle is never still enough — check `bias_still_gyro_threshold_rad_s`. |
| IMU: `"...at or beyond full scale"` | The configured range does not match the actual IMU. Set `max_gyro_rad_s` / `max_accel_mps2` from the datasheet. |
| Depth: `"no surface zero taken"` | You skipped checklist step 1. |
| Depth: `"pressure ... outside the plausible band"` | Sensor unplugged or failed. An unplugged sensor reads zero, which is not "very shallow". |
| Depth: `"depth ... is above the surface reference"` | The zero was taken underwater, or atmospheric pressure has moved a lot since. |
| Heading: `"disagrees with gyro by N deg/s"` | The source is being magnetically disturbed — or `max_yaw_rate_mismatch_deg_s` is too tight for its latency. Run the swing test before assuming the sensor is bad. |
| Heading: `"tilt N deg beyond the limit"` | Normal during a roll or pitch manoeuvre. An absolute heading means nothing at that attitude. |
| Heading: `"recovering from disturbance"` | Working as intended. Recovery is slow on purpose. |
| Heading: `"field magnitude N% off expected"` | Something ferrous or electrical is near the magnetometer. Check `expected_field_magnitude_t` was measured at this venue. |
| `/nav_status` mode 3 (FAULT), `"filter silent"` | `ekf_filter_node` has died or is not receiving anything. Check its log first. |
| `/nav_status` mode 3, `"critical sensor unavailable: imu"` | The IMU gate is stale or rejecting. The IMU drives the prediction step, so nothing works without it. |
| `/nav_status` mode 2 (DEAD_RECKONING) | Normal during a DVL dropout. Watch `dead_reckoning_time_s` and `position_sigma_xy_m` grow. |
| TF jitters, two publishers on `odom → base_link` | Something other than `ekf_filter_node` is publishing it. `imu_filter_madgwick` must run with `publish_tf: false`. |
| z, roll and pitch are pinned to zero | `two_d_mode` got set true. It must be false on a six-degree-of-freedom vehicle. |
| Everything looks fine but the vehicle moves wrong | Frames. Check the mount rotations in `vehicle.yaml` against the physical vehicle. See section 11. |

### Recording a run

Always record. Debugging from memory does not work.

```bash
ros2 bag record -o run_$(date +%F_%H%M) \
  /dvl_gate/velocity /dvl_gate/altitude /dvl_gate/health \
  /imu_gate/data /imu_gate/gyro_bias /imu_gate/health \
  /depth_gate/pose /depth_gate/depth /depth_gate/health \
  /heading_gate/pose /heading_gate/yaw_deg /heading_gate/health \
  /imu/attitude /odom/filtered /nav_status \
  /tf /tf_static
```

Because the validation logic is plain Python, you can replay a bag and run the
validators over it offline without any of ROS running.

---

## 10. Recipe: adding a new sensor

Say a forward-looking sonar arrives and you want it in the stack. The pattern
is the same every time.

**1. Add its mount to `vehicle.yaml`.**

```yaml
    sensors:
      forward_sonar:
        frame: "sonar_link"
        xyz: [0.19, -0.04, 0.10]    # MEASURE this on the vehicle
        rpy: [0.0, 0.0, 0.0]        # MEASURE this too
```

`static_frames` picks it up automatically — no code change.

**2. Write the validation logic as a plain Python class.**

Create `auv_sensors/auv_sensors/sonar_validation.py` with a `Sample`
dataclass, a `Limits` dataclass, a `Verdict`, and a `check(sample, now)`
method. No `import rclpy` in this file. Copy the shape from
`depth_validation.py`, which is the smallest example.

Ask yourself, for this sensor specifically:

- What makes a sample physically impossible? (range beyond the transducer's
  capability, non-finite values)
- What makes it meaningless even though it is in range? (no return, wrong
  operating mode)
- What is the reference the measurement is relative to, and can it be wrong?
  (the depth sensor's surface zero is the example)
- What does "degraded but usable" look like?

**3. Write the tests before the node.**

One test per rule, named after the rule. Look at
[`test_dvl_validation.py`](../auv_sensors/test/test_dvl_validation.py) for the
style: each test is three or four lines and its name is a sentence.

**4. Write the node, subclassing `SensorGateNode`.**

```python
class SonarGate(SensorGateNode):
    def __init__(self):
        super().__init__(
            node_name="sonar_gate",
            sensor_name="sonar",
            default_expected_rate_hz=10.0,
            default_stale_timeout_s=0.4,
        )
        ...

    def _on_raw(self, msg):
        verdict = self._validator.check(self._to_sample(msg), self.now_seconds())
        if not verdict.accepted:
            self.reject(verdict.reason)
            return
        self.accept(verdict.degraded_reason)
        self._publish(msg, verdict)
```

Health reporting comes free from the base class.

**5. Publish a standard message** if one exists for what the sensor measures.
A range goes out as `sensor_msgs/Range`, a pose as
`geometry_msgs/PoseWithCovarianceStamped`. Only reach for a custom message
when nothing standard fits.

**6. Register the entry point** in `auv_sensors/setup.py`, **add tuning** to
`config/sensors.yaml`, and **add the node** to both launch files.

**7. Give it a simulation source** if the Gazebo model does not have one, the
way `sim_pressure_source.py` does for the barometer. If a sensor cannot be
exercised in simulation, every bug in it costs pool time.

---

## 11. Traps that will bite you

**Trusting a magnetic heading unconditionally.** Yaw is our biggest problem
and a compass appears to solve it, so the temptation is enormous. The trap is
not using one — the trap is believing it.

A competition pool is a steel box: reinforced concrete, metal gantries
overhead, and the vehicle itself is full of thruster magnets and
current-carrying wire whose field changes with throttle. A disturbed compass
does not get noisier, it gets **biased**: it reports a consistent wrong answer
while still looking healthy. Handing that to a filter is worse than handing it
nothing, because the filter has no way to tell.

NTU Mecatron's 2026 report describes IMU data "jumping" underwater until they
switched to a relative heading mode and recalibrated away from magnetic
influence. NUS Bumblebee — who won that year — bolt a dedicated fibre-optic
gyro to the vehicle rather than trust a magnetic heading at all.

This stack's answer is `heading_gate`: use a heading source, but cross-check
every sample against the gyro, which cannot be magnetically disturbed, and
refuse it while the two disagree. Two rules follow:

- **`imu_gate` still drops the driver's fused orientation.** Absolute heading
  is `heading_gate`'s job; forwarding it from two nodes would fuse the same
  measurement twice. The `use_orientation` parameter exists and logs a
  warning — leave it alone.
- **Never raise `max_yaw_rate_mismatch_deg_s` to make the warnings go away.**
  That threshold is the whole defence. If the gate keeps rejecting, either the
  source really is being disturbed, or the threshold was never set from a
  swing test. Find out which.

**Euler angles for anything but display.** Roll, pitch and yaw are fine for
printing to a screen and for writing mount rotations in a config file. They
are not fine for differencing two attitudes, interpolating between them, or
doing anything else mathematically, because they become degenerate at pitch
±90° — and this vehicle deliberately goes there for style points. Use
quaternions and `frames.quat_delta_to_rotvec`.

**Mixing clocks.** `time.monotonic()`, `time.time()`, the ROS clock and a
message header stamp are four different numbers. Under `use_sim_time` the ROS
clock runs at Gazebo's pace while the others run at wall pace, so comparing
across them gives you nonsense that does not error — it just produces
staleness checks that never fire. Use `self.now_seconds()` from
`SensorGateNode`, or a header stamp, and never compare one against the other.

**Trusting a covariance of zero.** It means "the driver did not fill this in",
not "this measurement is perfect". Substitute a default.

**Rotating a vector without its covariance.** Silent, and it corrupts the
filter's idea of how much to trust the sensor.

**Guessing sensor mount rotations.** Every `MEASURE` in `vehicle.yaml` is
there because that number cannot be derived from CAD or from the datasheet. A
DVL mounted 45° off in yaw makes every straight-line transit come out
diagonal, and it looks exactly like a control tuning problem. Measure it.

**Putting logic in a gate node.** If you find yourself wanting a gate to
remember a pose, integrate a velocity, or combine two sensors, that work
belongs in Layer 2. The moment a gate does it, it stops being testable without
a running graph, and the layer boundary — the thing that lets two people work
in parallel — is gone.

**Skipping the surface zero or the still period.** Both are 30 seconds. Both
produce errors that are undetectable after the fact.

---

## 12. What is not built yet

Layers 3 through 5 do not exist. The contracts they will use are already fixed
in [ARCHITECTURE.md](ARCHITECTURE.md#contracts), which is what lets them be
built in parallel.

Layer 2 exists but is **untuned**: the filter's process noise is
`robot_localization`'s default, and nobody has compared its output against a
known trajectory. Tuning needs ground truth, which needs either the simulator
or a synthetic trajectory generator.

The build order, with reasoning, is in
[ARCHITECTURE.md](ARCHITECTURE.md#remaining-build-order). The next item is
**system identification**: drive the thrusters with step inputs, log the DVL's
velocity response, and fit a drag model into `vehicle.yaml`. It needs no new
software architecture, it makes both the controller and the simulator better,
and having a DVL makes it nearly free. Most student teams skip it and pay for
it in control tuning later.

### When the IMU arrives

The part is not chosen yet, so every IMU number in `config/sensors.yaml` is a
deliberately pessimistic placeholder. Filling them in is a ten minute job off
the datasheet, and the file lists which specification maps to which parameter.

The one specification to choose the part on does not appear as a parameter at
all: **gyro bias instability**. With no absolute heading source, yaw is a gyro
integral, and that number decides how fast it drifts. Roughly, over a fifteen
minute run: a consumer IMU (tens of °/hr) drifts more than 10° and is
unusable; a good tactical MEMS unit (5–10 °/hr) drifts 1–3°, which is
workable; better than that and yaw stops being the limiting factor.

That number is also what decides whether a separate heading source is a
convenience or a requirement.

### The open item that matters

**Yaw is still the only unbounded error.** Section 3 explains why. Depth is
measured absolutely, roll and pitch are pinned by gravity, velocity comes
straight from the DVL — but heading comes from integrating a gyro, and
horizontal position error grows as roughly `distance travelled × heading
error`. It is what decides whether the vehicle can navigate back to where it
started.

The plan is to close it in hardware with an absolute heading source, and
`heading_gate` is the socket it plugs into. Two things still have to happen:
pick the part, and tune the gyro cross-check from a pool swing test. Until
both are done, missions should be short and relative rather than depending on
a heading that holds for fifteen minutes.

### The hydrodynamic model is placeholders, and that is fine

`vehicle.yaml` carries drag and added-mass numbers that nobody has measured.
Only the translational drag terms are realistically obtainable with the
equipment on hand — constant thrust, wait for terminal velocity, read it off
the DVL. Added mass and rotational damping would need CFD.

This is not blocking anything. Those numbers are used as **feedforward only**:
a head start the controller gets for free. The feedback loop does the actual
work and closes on the DVL's measured velocity, so a missing or wrong model
costs some crispness and nothing else. Each entry records its own `source`
field so nobody has to guess whether a number was measured or invented. Do not
let anyone spend a month here.

---

## 13. Glossary

**AHRS** — Attitude and Heading Reference System. An IMU plus enough
processing to output an orientation rather than raw rates.

**Bias** — a constant offset in a sensor's output. A gyro reading 0.5 deg/s
while perfectly still has a 0.5 deg/s bias. Integrating it turns it into a
heading error that grows linearly with time.

**Bottom lock** — the DVL successfully tracking the sea floor and therefore
able to report ground-relative velocity. Without it, no velocity.

**Contract** — the fixed message type on a fixed topic that forms the boundary
between two layers.

**Covariance** — the uncertainty of a measurement, expressed as a matrix. See
section 5.

**Dead reckoning** — computing position by integrating velocity and heading
over time, with no external position fix.

**Feedforward** — a control term computed from a model of the vehicle rather
than from measured error. It gives the controller a head start; the feedback
loop corrects whatever the model got wrong.

**Degraded** — a gate's verdict meaning "usable, but trust it less". Distinct
from rejected, which means "do not use at all".

**DVL** — Doppler Velocity Log. Measures ground-relative velocity acoustically.

**ENU / NED** — East-North-Up / North-East-Down. Two conventions for world
frame axes. ROS uses ENU; autopilots use NED.

**FLU / FRD** — Forward-Left-Up / Forward-Right-Down. Two conventions for body
frame axes. ROS uses FLU; autopilots and the DVL use FRD.

**Gate** — a node validating one sensor and reporting its health. Layer 1.

**IMU** — Inertial Measurement Unit. Gyroscope plus accelerometer.

**Hysteresis** — requiring several consecutive samples before changing state,
so that a single outlier cannot flip a decision back and forth.

**Magnetic disturbance** — nearby ferrous metal or current distorting the
local magnetic field, so a compass reports a heading that is wrong by an
amount that changes as the vehicle moves. Biased, not noisy, which is what
makes it dangerous.

**Odometry** — an estimate of motion derived from the vehicle's own sensors.
Continuous, and it drifts.

**REP-103 / REP-105** — the ROS standards defining units, axis conventions,
and the `map` / `odom` / `base_link` frame hierarchy.

**Swing test** — turning the vehicle slowly through a full 360 degrees in the
water while recording, to characterise a heading source against the gyro.

**TF** — the ROS system that tracks relationships between frames and
transforms data between them.

**Wrench** — a force and a torque together, as a six-element vector. The
output of the controller and the input to thrust allocation.
