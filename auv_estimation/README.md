# auv_estimation

Layer 2 of the autonomy stack. Fuses the Layer 1 gate outputs into one
continuous estimate of where the vehicle is and how fast it is going, and
reports how much that estimate is worth.

New to the project? Start with [docs/ONBOARDING.md](../docs/ONBOARDING.md).

## What runs

```
/imu_gate/data ──▶ imu_filter_madgwick ──▶ /imu/attitude ──┐
   (gyro + accel)     (use_mag: false)                     │
                                                           ▼
/dvl_gate/velocity ──────────────────────────────▶  ekf_filter_node ──▶ /odom/filtered
/depth_gate/pose ────────────────────────────────▶   (robot_localization)   + TF odom→base_link
/heading_gate/pose ──────────────────────────────▶

all */health ──────────────────────▶ nav_status ──▶ /nav_status
```

Two of those three are off-the-shelf. **The only code in this package is
`nav_status`** — everything else is configuration.

`robot_localization` and `imu_filter_madgwick` are released ROS packages
installed from apt by `rosdep` via this package's `package.xml`. They are not
vendored here and must not be added to `nautronics.repos`, which is only for
sources we build ourselves.

## Running

```bash
ros2 launch auv_estimation navigation.launch.py            # layers 0+1+2, real vehicle
ros2 launch auv_estimation navigation.launch.py sim:=true  # against the simulator
ros2 launch auv_estimation estimation.launch.py            # layer 2 only, gates already up
```

`use_heading:=true` adds the absolute heading source once the hardware exists.

## The design in one sentence

**z** comes from pressure, **roll and pitch** from gravity, **velocity** from
the DVL, and **yaw** is integrated from the gyro inside the filter with an
absolute correction from `heading_gate` when one is available. Nothing else is
fused.

Three consequences of that worth knowing before you edit `config/ekf.yaml`:

**Madgwick is in the chain for roll and pitch, not for yaw.** `imu_gate`
strips the driver's orientation, so without something observing the
accelerometer the filter would integrate roll and pitch from the gyro and
throw away a free absolute reference. Madgwick's *yaw* is a gyro integral and
is deliberately not fused (`imu0_config` index 5 is `false`) — fusing it would
make the filter cling to a drifting value and fight `heading_gate`.

**Accelerometer fusion is off.** The DVL measures velocity directly at 8 Hz,
so accelerometers mainly buy the ability to coast through DVL dropouts — and
on a low-cost IMU the accelerometer bias integrates into velocity error fast
enough to spoil that. Revisit once the IMU is chosen, by measuring how far the
estimate walks during a deliberate dropout with it off, then on.

**There is no map frame.** `world_frame` is `odom`, so this node publishes
`odom → base_link` and nothing else. `odom`'s origin is wherever the vehicle
was when the filter started, and it drifts.

## nav_status

`robot_localization` keeps publishing odometry after the DVL has gone silent,
and that odometry looks identical to odometry backed by real measurements.
`nav_status` is what tells them apart:

| Field | Meaning |
|---|---|
| `mode` | `INIT` / `AIDED` / `DEAD_RECKONING` / `FAULT` |
| `dead_reckoning_time_s` | how long since velocity aiding was last available |
| `position_sigma_xy_m` | the **worse** of the two horizontal sigmas |
| `yaw_sigma_rad` | the one that grows without bound when no heading source is fitted |

It reports; it does not decide. Whether 8 seconds of dead reckoning is too
long to keep going is a mission-layer policy, and putting it here would bury a
tactical decision inside a status node.

## Two things that will bite

**Only one node may publish `odom → base_link`.** That is `ekf_filter_node`.
`imu_filter_madgwick` runs with `publish_tf: false` for this reason — if TF
starts jittering, check that first.

**`two_d_mode` must stay `false`.** It is the classic way to get a filter that
silently pins z, roll and pitch to zero on a vehicle that has six degrees of
freedom.

## Tests

```bash
colcon test --packages-select auv_estimation
```

The `nav_status` logic is plain Python with no ROS dependency and is tested
directly. The filter configuration is not unit-testable; validating it needs a
known trajectory to compare against.
