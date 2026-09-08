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
allocation in the same week.

**Standard messages wherever one exists.** Pose, twist, wrench, odometry,
IMU and range stay on `geometry_msgs` / `nav_msgs` / `sensor_msgs` so that
rviz, `robot_localization`, rosbag tooling and Nav2 keep working.
`auv_core_msgs` holds only the interfaces with no standard equivalent.

**One conversion point per convention.** Frame conversions live in
`auv_core/frames.py` and nowhere else. A sign flip written anywhere else is a
bug, not a shortcut.

**Nothing reaches the estimator ungated.** Every sensor passes through a
Layer 1 gate that validates it and reports its health. Both of the RoboSub
teams whose 2025/2026 technical design reports informed this design lost pool
time to sensors that degraded silently.

## Layers

| # | Layer | Package | Status |
|---|-------|---------|--------|
| 0 | Frames, geometry, contracts | `auv_core`, `auv_core_msgs` | built |
| 1 | Sensor gates | `auv_sensors` | built |
| 2 | State estimation (`odom`) | `auv_estimation` | built, untuned |
| 3 | Guidance and trajectory | — | not started |
| 4 | Control, and the boundary to ArduSub | `auv_actuation` | built, untuned |
| 5 | Mission (behaviour tree) | — | not started |
| — | Simulation harness | `auv_description` | inherited |

Perception is out of scope for now, and so is any global (`map`-frame)
correction — this vehicle has no source of absolute position, so there is
nothing to correct against. See [Frame conventions](#frame-conventions) and
[Known open items](#known-open-items).

## Contracts

| Boundary | Message | Frame |
|---|---|---|
| sensor → gate | driver-native | sensor |
| gate → estimator | `sensor_msgs/Imu`, `nav_msgs/Odometry` (twist only), `geometry_msgs/PoseWithCovarianceStamped`, `sensor_msgs/Range` | sensor |
| gate → operators | `auv_core_msgs/SensorHealth` | — |
| estimator → control | `nav_msgs/Odometry` on `/odom/filtered` | `odom` |
| estimator → mission | `auv_core_msgs/NavStatus` | — |
| guidance → control | `geometry_msgs/TwistStamped` | `base_link` |
| control → output stage | `geometry_msgs/WrenchStamped` | `base_link` |
| output stage → control | `geometry_msgs/WrenchStamped` (realised) | `base_link` |
| output stage → autopilot bridge | `auv_core_msgs/PilotDemand` | — |

Fixing these now is what lets Layer 2 and Layer 4 be built in parallel: the
estimator author and the allocation author never have to agree on anything
beyond the table above.

## Frame conventions

Everything on a ROS topic is REP-103: **ENU** world, **FLU** body. The
autopilot and the DVL speak NED/FRD, and those are converted at exactly two
places — the DVL gate on the way in, and the autopilot bridge on the way out.
Both call `auv_core/frames.py`.

```
odom ─(estimator, Layer 2)─▶ base_link
                                 │
     (static, from config/vehicle.yaml) ─┬─▶ imu_link
                                         ├─▶ heading_link
                                         ├─▶ dvl_link
                                         ├─▶ depth_link
                                         └─▶ *_camera_link
```

`base_link → sensor` transforms come from `auv_core/config/vehicle.yaml` and
are published by the `static_frames` node. That file is the single source of
truth for vehicle geometry: the static TFs, the thrust allocation
configuration matrix and the simulator's URDF all have to agree with it.

**There is no `map` frame, deliberately.** REP-105 reserves one for a global,
drift-free world frame, but this vehicle has no source of absolute position:
no GPS underwater, no surveyed acoustic beacons, no prior map. A `map` frame
would be an alias for `odom` pretending to be something better, and code
would start assuming a pose means the same thing at minute 15 as at minute 0.

So `odom` is the only world frame. Its origin is wherever the vehicle was when
the estimator started, and it drifts in yaw and horizontal position. Every
mission has to be expressed relative to that origin, or relative to something
the vehicle can currently see.

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
/heading/raw ──────▶│heading_gate │ gyro cross-check, tilt  │──▶ /heading_gate/pose   (PoseWithCovarianceStamped)
  + /imu_gate/data  │  (optional) │ limit, field magnitude  │──▶ /heading_gate/yaw_deg (Float64)
                    └─────────────┴─────────────────────────┘──▶ /heading_gate/health
```

`depth_gate` and `heading_gate` are the two **absolute** sensors: everything
else measures a rate, and rates have to be integrated. Depth is always
available; heading needs hardware the vehicle does not have yet, so
`heading_gate` is off unless launched with `use_heading:=true`.

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
[ONBOARDING.md](ONBOARDING.md).

## Tests

```bash
colcon test --packages-select auv_core auv_sensors
colcon test-result --verbose
```

The validation logic in every gate is a plain Python class with no ROS
dependency, so it is tested directly rather than through a running graph. The
tests are the specification of what each gate lets through.

`test_vehicle_config.py` also asserts properties of the physical vehicle:
that thruster geometry has rank 6 (fully actuated, so arbitrary attitude
while translating is possible), that the vehicle is slightly positively
buoyant, and that the centre of buoyancy sits above the centre of mass. If one
of those fails, the fix is on the vehicle or in `vehicle.yaml`, not in the
code that reads it.

## Remaining build order

Each step is testable on its own. Steps 4 and 5 can run in parallel because
the contract between them is already fixed.

1. ~~Frames, contracts, sim harness~~ — done
2. ~~Sensor gates~~ — done
3. ~~Estimator~~ — built as `auv_estimation`, but **untuned**: the process
   noise is `robot_localization`'s default and nobody has compared the output
   against a known trajectory. Tuning needs ground truth, which needs either
   the simulator or a synthetic trajectory generator.
4. **Thruster and drag calibration** — the measurable part only. Command a
   constant thrust, wait for the vehicle to stop accelerating, read terminal
   velocity off the DVL: at terminal velocity thrust equals drag, so this
   gives the translational quadratic drag coefficient directly, one steady run
   per axis. Half an hour of pool time. Added mass and rotational damping are
   *not* worth chasing this way — take them from CFD if you have it, leave the
   placeholders if you do not. Nothing downstream is blocked either way,
   because the model is only feedforward and the feedback loop closes on
   measured DVL velocity.
5. ~~Wrench interface to ArduSub~~ — built as `auv_actuation`. The full-scale
   calibration in its config is **not measured**; every gain above it is
   meaningless until it is.
6. ~~Velocity and position control loops~~ — built as `pose_controller`. Gains
   are a starting point sized off placeholder forces; nothing has run on a
   vehicle. Tune in the order given in its config: bridge calibration, then
   inner loop, then outer loop.
7. **Guidance**: trajectory generation and LOS, publishing `cmd_vel` into the
   socket the controller already exposes.
8. **Mission behaviour tree.**

## Known open items

**Yaw is the only unbounded error, and it is not solved yet.** Depth is
measured absolutely, roll and pitch are pinned by gravity, velocity comes
straight from the DVL. Yaw comes from integrating the gyro, and horizontal
position error then grows as roughly `distance travelled × heading error`. It
is the error that decides whether the vehicle can navigate back to where it
started.

The plan is to close it in hardware, with an absolute heading source, and
`heading_gate` is the socket that source plugs into. Two things still have to
happen:

- **pick the part.** The specification that matters is heading stability
  indoors, not accuracy on a bench. A magnetic compass is cheap and will be
  disturbed by the pool; an AHRS with good disturbance rejection is better; a
  gyrocompass or fibre-optic gyro sidesteps magnetism entirely and costs
  accordingly. The fallback if none is fitted is a gyro whose bias instability
  is low enough that the drift over one run is tolerable.
- **tune the cross-check.** `heading_gate` refuses any heading that disagrees
  with the gyro, because a disturbed magnetic heading is *biased* rather than
  noisy and a bias delivered with an honest-looking covariance is worse than
  no measurement. The threshold has to be set from a pool swing test before
  the source can be trusted. See `max_yaw_rate_mismatch_deg_s` in
  `config/sensors.yaml`.

Until then the vehicle navigates relative to where it started, and missions
should be designed to be short and relative rather than to depend on a
heading that holds for fifteen minutes.

**The hydrodynamic model is placeholders.** Only the translational drag terms
are realistically measurable with the equipment on hand; added mass and
rotational damping would need CFD. This is not blocking: the model is used as
feedforward only, and the feedback loop closes on the DVL's measured velocity.
A missing model costs some crispness in the controller, not correctness. Do
not let anyone spend a month here.

## Inner loop: why ArduSub keeps it

ArduSub's STABILIZE and ALT_HOLD run at 400 Hz on real-time hardware, with
arming logic and failsafes that are already proven on this vehicle. Rewriting
that on a companion computer over a non-real-time link is what NUS Bumblebee
do — with 54 people and 300 hours of in-water testing. Replacing something
that works is the wrong place to spend a one-year budget.

So the split is: ArduSub owns attitude and depth, this stack owns position and
velocity. `auv_actuation` is the boundary, and it converts a commanded wrench
into pilot-input axes.

Two consequences are permanent features of this choice rather than bugs:

- **No roll or pitch torque.** Those axes take a lean angle inside ArduSub's
  limit, not a torque, so RoboSub's 90° gate style points need a temporary
  switch to ACRO or MANUAL.
- **Absolute heading stays ours.** Only yaw *rates* are ever commanded. The
  autopilot's job on that axis is to hold still when left alone; its own idea
  of where north is never gets read by anything in this stack, so the two
  numbers do not have to agree and there is no "two yaws" problem to
  reconcile. The outer loop closes on our estimate: error becomes a rate, the
  vehicle turns, the error reaches zero, the command stops. Pixhawk
  stabilises, the external sensors navigate.

If either becomes intolerable, the escape is already designed: replace
`pilot_demand` with a thrust allocator driving the thrusters directly. The
`WrenchStamped` contract above it does not move, and `vehicle.yaml` already
carries the rank-6 configuration matrix that allocation needs.

**One process owns the MAVLink link**, and it is `auv_hardware/pixhawk_bridge2`.
Nothing in the new stack opens its own connection: a second process cannot
bind the same endpoint, and two independent sources of command authority over
one vehicle is a worse problem than a port conflict. `auv_actuation` publishes
`auv_core_msgs/PilotDemand` and that bridge forwards it as `MANUAL_CONTROL`.
Note that this and the existing `pwm_router` RC-override path are both pilot
input — run one at a time.

## Where the design decisions came from

- **No magnetometer in the yaw solution.** NTU Mecatron's RoboSub 2026 report
  describes IMU data "jumping" underwater until they moved to a relative
  heading mode and recalibrated the IMU away from magnetic influence. NUS
  Bumblebee mount a dedicated fibre-optic gyro alongside a VectorNav VN-100
  rather than trust a magnetic heading at all.
- **Estimation on the companion, not in the autopilot EKF.** Mecatron traced
  unstable depth control to ArduPilot's EKF3 fusing DVL vertical velocity
  despite that fusion being disabled in the parameters, and had to patch
  firmware. Keeping the fusion in our own process removes that whole class of
  problem.
- **Dead-reckoning through DVL dropouts.** Bumblebee report beam reflections
  and dropouts during and after high-rate manoeuvres and near pool walls and
  floors, bridged with dead-reckoning fallbacks. The lock hysteresis in
  `dvl_validation.py` is the first half of that; the coasting logic belongs to
  the estimator.
- **Quaternion attitude deltas, never Euler.** RoboSub 2026 awards style
  points for 90-degree roll and pitch through the gate, which is exactly where
  Euler differencing becomes discontinuous.
