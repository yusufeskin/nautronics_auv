# auv_core

Layer 0 of the autonomy stack. Nothing here subscribes to a sensor or commands
an actuator — it holds the definitions every other layer agrees on, so that
layers can be written and replaced independently.

New to the project? Start with [docs/ONBOARDING.md](../docs/ONBOARDING.md).

## What is in it

| File | Purpose |
|---|---|
| [`auv_core/frames.py`](auv_core/frames.py) | **The only place** ENU/NED and FLU/FRD are converted. Quaternion algebra, covariance rotation, heading conventions. |
| [`config/vehicle.yaml`](config/vehicle.yaml) | Single source of truth for vehicle geometry: sensor mounts, thruster layout, mass and buoyancy. |
| [`auv_core/vehicle_config.py`](auv_core/vehicle_config.py) | Typed loader for that YAML, plus the derived thruster configuration matrix. |
| [`auv_core/health.py`](auv_core/health.py) | Rate, staleness and rejection bookkeeping shared by every sensor gate. Pure Python, no ROS. |
| [`auv_core/gate.py`](auv_core/gate.py) | `SensorGateNode`, the base class Layer 1 gates subclass. |
| [`auv_core/static_frames_node.py`](auv_core/static_frames_node.py) | Publishes `base_link → sensor` transforms from `vehicle.yaml`. |

## Running

```bash
ros2 launch auv_core frames.launch.py
```

Arguments: `vehicle_config` (path to the YAML), `use_sim_time`.

## Two rules

**Never convert a frame outside `frames.py`.** A sign flip written anywhere
else is a bug, not a shortcut, and it will present as a control tuning problem
weeks later.

**Never differentiate Euler angles.** They are degenerate at pitch ±90°, which
this vehicle deliberately passes through for RoboSub gate style points. Use
`frames.quat_delta_to_rotvec`.

## Editing vehicle.yaml

It is the authority for geometry — the static TFs, thrust allocation and the
simulator URDF all have to agree with it. If a sensor moves on the vehicle,
this file changes and nothing else does.

Values marked `MEASURE` must come from the physical vehicle, not from CAD.
Values marked `TUNE` come from pool measurements.

The `frames` block declares **no map frame**, deliberately: this vehicle has
no source of absolute position, so `odom` is the only world frame and it
drifts. Every mission must be relative.

The `hydrodynamics` block is placeholders, and each entry records its own
`source` (`placeholder` / `measured` / `cfd`). Only the translational drag
terms are realistically measurable here — constant thrust, terminal velocity,
read it off the DVL. Nothing is blocked on the rest: these numbers are
feedforward only, and the feedback loop closes on measured velocity.

## Tests

```bash
colcon test --packages-select auv_core
```

`test_vehicle_config.py` asserts properties of the *vehicle*, not of the code:
thruster geometry has rank 6 (fully actuated), the vehicle is slightly
positively buoyant, and the centre of buoyancy sits above the centre of mass.
A failure there means the fix is in `vehicle.yaml` or on the vehicle.
