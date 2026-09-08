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
| `ThrusterCommand` | allocation → hardware | Layer 4 |

`NavStatus` and `ThrusterCommand` describe layers that do not exist yet. They
are defined now because fixing the contracts early is what lets those layers
be built in parallel — see
[docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md#contracts).

Before adding a message here, check that no standard one fits.
