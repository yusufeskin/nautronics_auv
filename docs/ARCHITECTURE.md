# Autonomy stack architecture

The reference for how this vehicle's software is split up and why. Read this
before adding a node; it says which layer your code belongs in and what it is
allowed to talk to.

## Principles

**Layers talk through contracts, not through each other.** Every boundary
below is a message type on a topic. A layer may be rewritten, replaced with an
off-the-shelf package, or swapped between simulation and hardware without any
other layer noticing. This is what lets two people work on estimation and
allocation in the same week.

**Standard messages wherever one exists.** Pose, twist, wrench, odometry,
IMU and range stay on `geometry_msgs` / `nav_msgs` / `sensor_msgs` so that
rviz, `robot_localization`, rosbag tooling and Nav2 keep working. `auv_core_msgs`
holds only the handful of interfaces with no standard equivalent.

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
| 2 | State estimation (odom + map correction) | — | next |
| 3 | Perception and object map | — | not started |
| 4 | Guidance and trajectory | — | not started |
| 5 | Control and thrust allocation | — | not started |
| 6 | Mission (behaviour tree) | — | not started |
| 7 | Simulation harness | `auv_description` | inherited |

## Contracts

| Boundary | Message | Frame |
|---|---|---|
| sensor → gate | driver-native | sensor |
| gate → estimator | `sensor_msgs/Imu`, `nav_msgs/Odometry` (twist), `geometry_msgs/PoseWithCovarianceStamped`, `sensor_msgs/Range` | sensor |
| gate → operators | `auv_core_msgs/SensorHealth` | — |
| perception → map | `auv_core_msgs/LandmarkObservationArray` | camera |
| estimator → control | `nav_msgs/Odometry` on `/odom/filtered` | `odom` |
| estimator → mission | `nav_msgs/Odometry` on `/pose/map`, `auv_core_msgs/NavStatus` | `map` |
| guidance → control | `geometry_msgs/TwistStamped` | `base_link` |
| control → allocation | `geometry_msgs/WrenchStamped` | `base_link` |
| allocation → hardware | `auv_core_msgs/ThrusterCommand` | — |

The two estimator outputs are separate on purpose. `odom` is continuous and
never jumps, so the controller can differentiate it; `map` is corrected by
landmark observations and *does* jump, which is fine for a mission planner
and fatal for a control loop.

## Frame conventions

Everything on a ROS topic is REP-103: **ENU** world, **FLU** body. The
autopilot and the DVL speak NED/FRD, and those are converted at exactly two
places — the DVL gate on the way in, and the autopilot bridge on the way out.
Both call `auv_core/frames.py`.

```
map ──(landmark corrector, Layer 2)──▶ odom ──(estimator, Layer 2)──▶ base_link
                                                                          │
                              (static, from config/vehicle.yaml) ─────────┤
                                                                          ├─▶ imu_link
                                                                          ├─▶ dvl_link
                                                                          ├─▶ depth_link
                                                                          └─▶ *_camera_link
```

`base_link → sensor` transforms come from `auv_core/config/vehicle.yaml` and
are published by the `static_frames` node. That file is the single source of
truth for vehicle geometry: the static TFs, the thrust allocation
configuration matrix and the simulator's URDF all have to agree with it.

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
                    └─────────────┴─────────────────────────┘──▶ /depth_gate/health
```

Run it against the simulator:

```bash
ros2 launch auv_sensors sensors_sim.launch.py
```

That starts the static frames plus all three gates with simulator topics and a
synthetic pressure source, because the Gazebo model carries no barometer. On
the real vehicle:

```bash
ros2 launch auv_core   frames.launch.py
ros2 launch auv_sensors sensors.launch.py
```

Before every run, with the vehicle floating on the surface:

```bash
ros2 service call /depth_gate/zero_surface std_srvs/srv/Trigger
```

The depth gate refuses to publish until this is done. A run started without it
carries a constant depth bias that nothing downstream can detect.

Hold the vehicle still for ~30 s at the same time so the gyro bias converges;
`/imu_gate/health` reports DEGRADED until it does.

## Tests

```bash
colcon test --packages-select auv_core auv_sensors
```

The validation logic in every gate is a plain Python class with no ROS
dependency, so it is tested directly rather than through a running graph. The
tests are the specification of what each gate lets through.

`test_vehicle_config.py` also asserts properties of the physical vehicle:
that thruster geometry has rank 6 (fully actuated, so arbitrary attitude
while translating is possible), that the vehicle is slightly positively
buoyant, and that the centre of buoyancy sits above the centre of mass. If one
of those fails, the fix is on the vehicle or in `vehicle.yaml`.

## Remaining build order

Each step is testable on its own. Steps 4 and 5 can run in parallel with 3
because the contracts between them are already fixed.

1. ~~Frames, contracts, sim harness~~ — done
2. ~~Sensor gates~~ — done
3. **System identification.** Command thruster steps, log DVL velocity, fit
   drag and added mass into `vehicle.yaml`. Feeds both control feedforward and
   simulator fidelity. Cheapest high-value step available, and the DVL makes
   it nearly free.
4. Thrust allocation and the wrench interface.
5. Estimator. Start with `robot_localization` behind the `/odom/filtered`
   contract; replace with an error-state filter if and when it earns it.
6. Velocity control loop (needs 4 and 5).
7. Position/attitude loop, trajectory generation, LOS guidance.
8. Perception and object map.
9. **Landmark re-alignment.** With no magnetometer, yaw comes from a gyro
   integral and drifts; depth and velocity do not. Bounding that drift with
   observations of known objects is the hardest and highest-value work in the
   whole stack. Start it as soon as 8 produces anything.
10. Mission behaviour tree.

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
