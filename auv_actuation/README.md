# auv_actuation

Layer 4 of the autonomy stack: the control loop, and the boundary between it
and ArduSub.

New to the project? Start with [docs/ONBOARDING.md](../docs/ONBOARDING.md).
The reasoning behind the split is in
[docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md#control-ardusub-in-alt_hold-driven-like-a-joystick).

## The split

ArduSub stays in **ALT_HOLD** and is driven the way a pilot with a joystick
would drive it: one PWM value per axis, as RC channel overrides. ArduSub does
the motor mixing, holds roll and pitch level, holds depth while the throttle
is centred and holds heading while yaw is centred. This stack decides where
the vehicle should be and moves the sticks to get it there.

```
Layer 2   /odom/filtered, /nav_status ──┐
                                        ▼
Layer 4   pose_controller ──▶ AxisDemand ──▶ pilot_demand ──▶ PilotDemand (RC PWM)
             ▲      ▲     ▲                                          │
     reference  target_pose  hold                                    ▼
   (guidance)   (by hand)  (service)            auv_hardware/pixhawk_bridge2
                                             (owns the MAVLink link, rc_override)
                                                         │ RC_CHANNELS_OVERRIDE
                                                         ▼
                                             ArduSub ALT_HOLD ─▶ mixer ─▶ motors
```

**Nothing in this package speaks MAVLink.** Only one process may hold the link
to the autopilot. `pixhawk_bridge2` owns it and must run with
`command_mode:=rc_override` — `autonomy.launch.py` does that. In that mode the
bridge gives ArduSub no DVL and accepts none of the legacy target topics, so
this package is the only thing commanding the vehicle.

## What each channel means, and what the controller sends

| Ch | Axis | In ALT_HOLD | Controller |
|---|---|---|---|
| 5 | forward | thrust fraction | position → speed (P, capped) → speed vs DVL (PID) |
| 6 | lateral | thrust fraction | same |
| 3 | throttle | climb rate | depth error × `kp_depth`, capped. No PID |
| 4 | yaw | turn rate | yaw error × `kp_yaw`, capped. No PID |
| 1, 2 | pitch, roll | lean angle | always 1500 |

ArduSub already closes the climb-rate and turn-rate loops, so a PID on those
axes here would close the same loop twice. When their error reaches zero the
stick centres and ArduSub holds.

Forward and lateral are the only axes nobody else closes, so the integral
lives there — buoyancy trim pushing the vehicle along, a thruster that is
slightly stronger one way. It has clamping anti-windup: while an output is
pinned at full stick it stops integrating in that direction.

`AxisDemand` carries surge and sway as fractions in [-1, 1] and climb and yaw
as m/s and rad/s, all FLU. `pilot_demand` turns that into PWM, flipping
lateral and yaw into ArduSub's directions and skipping over ArduSub's stick
deadzones.

## Running

```bash
ros2 launch auv_bringup autonomy.launch.py sim:=true autopilot:=true
```

Nothing moves until four deliberate steps, none of which these nodes take on
their own:

```bash
ros2 service call /change_mode  auv_interfaces/srv/SetVehicleMode "{mode_name: 'ALT_HOLD'}"
ros2 service call /pilot_demand/enable     std_srvs/srv/SetBool "{data: true}"
ros2 service call /arm          std_srvs/srv/SetBool "{data: true}"
ros2 service call /pose_controller/enable std_srvs/srv/SetBool "{data: true}"
```

The output stage is enabled before arming so ArduSub sees centred sticks from
the moment it arms. Enabling the controller **holds the vehicle where it is**. Then give it
somewhere to be:

```bash
ros2 topic pub --once /pose_controller/target_pose geometry_msgs/msg/PoseStamped \
  "{header: {frame_id: odom}, pose: {position: {x: 2.0, y: 0.0, z: -1.5}}}"

ros2 service call /pose_controller/hold std_srvs/srv/Trigger    # stop here
```

Watch `/pose_controller/pose_error` shrink and `/pose_controller/at_target` go
true. A zero quaternion in the target means "keep whatever heading you have".
Targets must be in `odom`; anything else is refused.

`~/reference` (`nav_msgs/Odometry`) is the socket guidance plugs into: the
pose to be at now, plus the reference's own velocity as feedforward. If that
stream stops for `reference_timeout_s`, the controller holds at the last
reference.

## Position hold

There is no hold mode. Holding is this controller with a reference that does
not move — on enable, on `~/hold`, and at the end of a guidance trajectory.
ArduSub's own POS_HOLD is not used: it would hold against EKF3's position,
which needs the DVL fed into the autopilot, and it would fight this loop.

## Safety behaviour

| Situation | What happens |
|---|---|
| `pilot_demand` disabled | nothing sent; ArduSub's override times out |
| Enabled, no demand yet, or older than `command_timeout_s` | every stick centred |
| No odometry for `odom_timeout_s` | controller stops publishing → sticks centred |
| DVL out longer than `dvl_coast_s`, or no `/nav_status` | forward and lateral centred; depth and yaw carry on |
| Normal | mapped demand at a fixed rate |

Centred sticks in ALT_HOLD mean **hold depth, hold heading, stop pushing** —
not cut thrust, which on a positively buoyant vehicle means surfacing.

Set `RC_OVERRIDE_TIME` on the vehicle to about a second. It is what returns
the sticks to neutral if this whole computer goes quiet.

A step-by-step first test in simulation, written for someone who has never
run the stack: [docs/TESTING.md](../docs/TESTING.md).

## Tuning order, which cannot be skipped

1. **Check `config/pilot_demand.yaml` against the vehicle.** The climb and
   turn rate at full stick (`PILOT_SPEED_UP`, `PILOT_SPEED_DN`, yaw scaling)
   and the deadzones (`THR_DZ`, `RCn_DZ`). Confirm both rates by measurement.
   `pilot_demand_sim.yaml` restates them for SITL, whose defaults may differ.
2. **Depth and yaw.** Target a depth, then a heading; raise `kp_depth` /
   `kp_yaw` until it arrives briskly without overshoot.
3. **Surge and sway.** Hold, push the vehicle off station, raise `kp` until it
   returns without oscillating, then add just enough `ki` to remove the
   standing offset.
4. **Position.** Target a few metres out. `kp_position` sets how briskly it
   arrives; `max_speed_xy` sets the cruise and is the first thing to lower if
   a manoeuvre looks violent.

## What this cannot do

**Roll and pitch manoeuvres.** In ALT_HOLD those sticks command a lean angle
and the controller always leaves them level. RoboSub's 90° gate style points
need a temporary switch to MANUAL.

## Tests

```bash
colcon test --packages-select auv_actuation
```

The control law and the mapping are plain Python with no ROS dependency.
ArduSub's sticks are not one consistent frame — lateral and yaw invert
against our FLU convention, the throttle stick does not — so each direction is
spelled out and tested rather than derived from a frame rule.
