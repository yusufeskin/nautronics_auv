# auv_core_msgs

The interfaces that form the boundaries between layers of the autonomy stack.

This package is deliberately small. Pose, twist, wrench, odometry, IMU and
range all stay on `geometry_msgs` / `nav_msgs` / `sensor_msgs`, so that rviz,
`robot_localization`, rosbag tooling and Nav2 keep working without adapters.
Only interfaces with no adequate standard equivalent live here.

| Message | Boundary | Produced by |
|---|---|---|
| `SensorHealth` | gate → operators and estimator | every Layer 1 gate |
| `NavStatus` | estimator → mission | Layer 2 |
| `AxisDemand` | control law → output stage | Layer 4 `pose_controller` |
| `PilotDemand` | output stage → autopilot bridge (RC channels) | Layer 4 `pilot_demand` |
| `ThrusterCommand` | allocation → hardware | unused |

`ThrusterCommand` belongs to a direct thrust-allocation path this vehicle does
not use: ArduSub does the motor mixing. It is kept because that path is the
designed escape if ALT_HOLD ever becomes intolerable. Fixing contracts early
is what lets layers be built in parallel — see
[docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md#contracts).

Before adding a message here, check that no standard one fits.
