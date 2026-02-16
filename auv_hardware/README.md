# AUV Hardware (auv_hardware)

The `auv_hardware` package serves as the bridge between the high-level ROS nodes and the physical hardware components of the AUV.

## Nodes

### `pwm_router.py`

Routes PWM signals to the appropriate actuators.

### Sensors

- **`baro_publisher.py`**: Publishes pressure and depth data from the barometer.
- **`battery_node.py`**: Monitors and publishes battery status.
- **`ping_sonar.py`**: Interfaces with the acoustic altimeter/sonar.

### `state_publisher.py`

Publishes the state of the vehicle, often aggregating data from multiple sensors.

## Usage

To launch the hardware interface nodes:

```bash
ros2 launch auv_hardware hardware.launch.py
```
