# AUV Control (auv_control)

The `auv_control` package provides the core control logic for the AUV, including thruster mixing and visual servoing.

## Nodes

### `thruster_mixer.py`

Maps high-level control commands (e.g., cmd_vel) to individual thruster PWM signals.

- **Function**: Applies a mixing matrix to translate forces and torques into thruster outputs.
- **Subscribes**: `/cmd_vel` (geometry_msgs/Twist)
- **Publishes**: PWM signals to the hardware interface.

### `visual_servoing_action.py`

Implements an action server for visual servoing, allowing the AUV to align itself with a target based on visual feedback.

- **Action Name**: `VisualServoing`
- **Goal**: Target keypoints or visual features.
- **Feedback**: Current error and alignment status.

### Movement Primitives

- **`point_follower.py`**: Logic for following a set of waypoints.
- **`yawer.py`**: Controls the yaw (heading) of the AUV to maintain a specific orientation.

## Usage

To launch the thruster mixer:

```bash
ros2 run auv_control thruster_mixer
```
