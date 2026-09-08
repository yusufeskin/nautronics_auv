# Nautronics AUV

**Nautronics AUV** is a ROS 2 based Autonomous Underwater Vehicle project. This repository contains the source code for simulation, control, vision, and hardware integration.

## Table of Contents
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Packages](#packages)

## Requirements

- **ROS 2** (Humble Hawksbill recommended)
- **ArduPilot**
  - [https://github.com/ArduPilot/ardupilot](https://github.com/ArduPilot/ardupilot)
- **ArduPilot Gazebo**  
  - [https://github.com/ArduPilot/ardupilot_gazebo](https://github.com/ArduPilot/ardupilot_gazebo)
- **Gazebo Sim (Harmonic for ROS 2)**  
  - [https://gazebosim.org/docs/latest/getstarted/](https://gazebosim.org/docs/latest/getstarted/)

## Installation

Follow these steps to set up the development environment:

1. **Create a Workspace**
   ```bash
   mkdir -p ~/nautronics_ws/src
   cd ~/nautronics_ws/src
   ```

2. **Clone the Repository**
   ```bash
   git clone https://github.com/yusufeskin/nautronics_auv
   cd ~/nautronics_ws
   ```

3. **Import Dependencies**
   ```bash
   vcs import src < src/nautronics_auv/nautronics.repos
   ```

4. **Install Python Requirements**
   ```bash
   pip3 install -r src/nautronics_auv/requirements.txt
   ```

5. **Install ROS Dependencies**
   ```bash
   sudo apt update
   rosdep update
   rosdep install --from-paths src --ignore-src -r -y
   ```

6. **Build the Workspace**
   ```bash
   colcon build --symlink-install
   ```

7. **Source the Workspace**
   ```bash
   source install/setup.bash
   ```

### GZ Sim & ArduPilot Configuration

Add the necessary model paths to your `~/.bashrc` to ensure Gazebo can find the AUV models:

```bash
echo 'export GZ_SIM_RESOURCE_PATH=$HOME/ardupilot_gazebo/models:$HOME/ardupilot_gazebo/worlds:$HOME/nautronics_auv/src/auv_description/models:$HOME/nautronics_auv/src/auv_description/worlds:${GZ_SIM_RESOURCE_PATH}' >> ~/.bashrc
source ~/.bashrc
```

## Documentation

- **[docs/ONBOARDING.md](docs/ONBOARDING.md)** — start here if you are new. Explains
  the underwater navigation problem, the frame conventions, and walks through the
  code from zero.
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — the layer split, the contract
  table, and the remaining build order.

## Packages

This repository consists of several ROS 2 packages:

- **[auv_core](auv_core/README.md)**: Layer 0 — frame conventions, vehicle geometry, sensor gate base.
- **[auv_core_msgs](auv_core_msgs/README.md)**: Layer boundary contracts.
- **[auv_sensors](auv_sensors/README.md)**: Layer 1 — DVL, IMU, depth and heading sensor gates.
- **[auv_estimation](auv_estimation/README.md)**: Layer 2 — state estimation and navigation status.
- **[auv_actuation](auv_actuation/README.md)**: Layer 4 — commanded wrench to ArduSub pilot input.
- **[auv_bt](auv_bt/README.md)**: Behavior Tree implementations for mission planning.
- **[auv_cam](auv_cam/README.md)**: Camera handling and image collection nodes.
- **[auv_control](auv_control/README.md)**: Control algorithms, including thruster mixing and visual servoing.
- **[auv_description](auv_description/README.md)**: URDF/Xacro descriptions and simulation models.
- **[auv_hardware](auv_hardware/README.md)**: Hardware interfaces for sensors and actuators.
- **[auv_interfaces](auv_interfaces/README.md)**: Custom ROS 2 messages and service definitions.
- **[auv_navigation](auv_navigation/README.md)**: Navigation stack and path planning.
- **[auv_vision](auv_vision/README.md)**: Computer vision algorithms (Object detection, Keypoint detection).
- **[additional_drivers](additional_drivers/bno055/README.md)**: External drivers (e.g., BNO055 IMU).