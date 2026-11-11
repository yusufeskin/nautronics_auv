# AUV Description (auv_description)

Simulation and description package for the AUV: the `prototype_vehicle` robot
model, competition course models, and the Gazebo (`gz-sim`) worlds it's
spawned into.

## Contents

- **`models/prototype_vehicle/`**: the vehicle itself.
  - `prototype.urdf.xacro`: top-level robot description (links, thrusters, sensors).
  - `macros/`: one xacro macro per part — `thruster_macro`, `imu_macro`,
    `camera_macro`, `dvl_macro` (Water Linked DVL-A50), `ping1d_macro`
    (Blue Robotics Ping1D echosounder, imitated with a narrow gpu_lidar —
    see the comment in that file), `pipe_macro`, `torpedo.xacro`.
- **`models/`**: course/prop models spawned into the worlds (`pool`,
  `teknofest_pipe`, `robosub_*`, `bluerov2_heavy`, `sand_heightmap`, ...).
  Each is a standalone `model.sdf` + `model.config`, includable with
  `<include><uri>model://&lt;name&gt;</uri></include>`.
- **`worlds/`**: `teknofest_pool.world` (default) and `open_water.world`.
  Course fixtures (lane markers, waypoint markers, pipe) live inline in the
  world file; reusable static geometry (the pool itself) is a `models/`
  entry included by the world instead of being duplicated inline.
- **`launch/gazebo.launch.py`**: spawns `gz-sim` with the selected world,
  bridges Gazebo <-> ROS topics via `config/bridge.yaml`, and spawns the
  vehicle (plus the two torpedo props).

## Usage

```bash
ros2 launch auv_description gazebo.launch.py
# or a specific world:
ros2 launch auv_description gazebo.launch.py world:=open_water.world
```

## Adding a part/sensor

1. Add `models/prototype_vehicle/macros/<part>_macro.xacro` with an
   `xacro:macro` that emits the link/joint (and a `<gazebo>` sensor block if
   it's a sensor).
2. `xacro:include` it and call the macro from `prototype.urdf.xacro`.
3. If it publishes a Gazebo topic ROS needs, add the mapping to
   `config/bridge.yaml` (or, for message types `ros_gz_bridge` doesn't
   support out of the box — see the DVL — a small dedicated bridge node
   under `auv_hardware`).
