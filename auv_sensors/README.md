# auv_sensors

Layer 1 of the autonomy stack: one gate per sensor, sitting between the driver
and the estimator.

New to the project? Start with [docs/ONBOARDING.md](../docs/ONBOARDING.md).

## What a gate does

1. **Validates** — is this sample physically possible, fresh, and actually
   locked on?
2. **Converts** — into this stack's conventions (FLU, correct frame id,
   sensible covariance).
3. **Reports health** — publishes `auv_core_msgs/SensorHealth` at a fixed rate
   whether or not samples are arriving, so a silent sensor and a dead node
   look different.

A gate must never integrate, fuse, hold a pose, or command the autopilot.
Those belong to Layer 2 and above. Keeping them out is what makes gates
testable without a running graph.

## Nodes

| Node | In | Out |
|---|---|---|
| `dvl_gate` | `~/raw` (`marine_acoustic_msgs/Dvl`) | `~/velocity` (`nav_msgs/Odometry`, twist only), `~/altitude` (`sensor_msgs/Range`), `~/health` |
| `imu_gate` | `~/raw` (`sensor_msgs/Imu`) | `~/data` (`sensor_msgs/Imu`, no orientation), `~/gyro_bias` (`geometry_msgs/Vector3Stamped`), `~/health` |
| `depth_gate` | `~/raw` (`sensor_msgs/FluidPressure`) | `~/pose` (`geometry_msgs/PoseWithCovarianceStamped`), `~/depth` (`std_msgs/Float64`), `~/health` |
| `sim_pressure_source` | `/odom` (Gazebo ground truth) | `~/raw` (`sensor_msgs/FluidPressure`) — **simulation only** |

`depth_gate` also serves `~/zero_surface` (`std_srvs/Trigger`).

## Running

```bash
ros2 launch auv_sensors sensors_sim.launch.py    # simulator
ros2 launch auv_sensors sensors.launch.py        # vehicle
```

Raw topics are launch arguments (`dvl_raw_topic`, `imu_raw_topic`,
`pressure_raw_topic`), which is what makes simulation and hardware
interchangeable to everything above this layer.

## Before every run

```bash
# vehicle floating on the surface:
ros2 service call /depth_gate/zero_surface std_srvs/srv/Trigger
# then hold it still ~30 s and watch the gyro bias converge:
ros2 topic echo /imu_gate/health
```

`depth_gate` refuses to publish until the first is done. Both prevent errors
that are undetectable afterwards. Full checklist in
[ONBOARDING.md](../docs/ONBOARDING.md#8-running-it).

## Two design choices worth knowing

**The IMU's fused orientation is dropped by default.** It is
magnetometer-aided, and a competition pool is a steel box. The estimator
integrates the bias-corrected gyro instead. See the module docstring in
[`imu_gate.py`](auv_sensors/imu_gate.py).

**Bottom lock has hysteresis in both directions.** Four consecutive bad
samples to lose it, two good ones to regain it, so a single reflected ping
cannot tear velocity aiding away and hand it straight back.

## Structure

Each sensor is split in two: a `*_validation.py` holding the decision logic as
plain Python with no ROS import, and a `*_gate.py` node that converts messages
and delegates. The tests exercise the first directly, which is why they run in
milliseconds and need no graph.

Adding a sensor? There is a step-by-step recipe in
[ONBOARDING.md](../docs/ONBOARDING.md#10-recipe-adding-a-new-sensor).

## Tests

```bash
colcon test --packages-select auv_sensors
```

The tests are the specification of what each gate lets through.
